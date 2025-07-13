import torch
import time
from typing import Tuple, Optional

def create_elastic_sparse_indices_for_validation(
    seq_len: int,
    layer_idx: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    attn_length: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    独立版本的索引创建函数，用于验证。
    Independent version of the index creation function for validation.
    """
    current_ids = torch.arange(seq_len, device=device)
    end_ids = torch.max(torch.tensor(attn_length - 1, device=device), current_ids)
    
    denominator = num_hidden_layers - 1 if num_hidden_layers > 1 else 1
    alpha = layer_idx / denominator
    
    ids_len = end_ids + 1
    position_distance = attn_length * (1 - alpha) + ids_len * alpha
    start_ids_base = (end_ids + 1 - position_distance).round().clamp(min=0).long()
    window_lengths_base = end_ids + 1 - start_ids_base
    
    spacing_base = torch.zeros_like(window_lengths_base, dtype=torch.float32)
    valid_mask = attn_length > 0
    if valid_mask:
        spacing_base = window_lengths_base.float() / attn_length

    steps_oversized = torch.arange(attn_length + 1, device=device)
    
    relative_indices_oversized = (steps_oversized[None, :] * spacing_base[:, None]).round().long()
    attention_indices_oversized = start_ids_base[:, None] + relative_indices_oversized

    base = attention_indices_oversized[:, :attn_length]
    shifted = attention_indices_oversized[:, 1:] - 1

    head_indices_tensor = torch.arange(num_attention_heads, device=device, dtype=torch.float32)
    interp_weights = head_indices_tensor / (num_attention_heads - 1) if num_attention_heads > 1 else torch.zeros_like(head_indices_tensor)

    interp_weights_expanded = interp_weights.view(num_attention_heads, 1, 1)
    attention_indices_float = base.unsqueeze(0).float() * (1 - interp_weights_expanded) + \
                              shifted.unsqueeze(0).float() * interp_weights_expanded

    attention_indices = attention_indices_float.round().long()
    attention_mask = (attention_indices <= current_ids.view(1, -1, 1)) & (attention_indices >= 0)
    
    return attention_indices, attention_mask.long()

def check_attention_holes_pytorch(
    seq_len: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    attn_length: int,
    position_id: Optional[int] = None
):
    """
    使用 PyTorch 和 CUDA 在超长序列上验证视野空洞。
    可以检查任意指定位置 (position_id) 的视野。
    Validates attention holes on ultra-long sequences using PyTorch and CUDA.
    Can check the receptive field of any specified position (position_id).
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping this test.")
        return

    device = torch.device("cuda:0")

    if position_id is None:
        target_pos = seq_len - 1
    else:
        if not (0 <= position_id < seq_len):
            raise ValueError(f"position_id must be between 0 and {seq_len - 1}, but got {position_id}")
        target_pos = position_id
        
    print(f"\n--- Checking with attn_length = {attn_length}, seq_len = {seq_len} ---")
    print(f"--- Target position to check: {target_pos} ---")

    if attn_length <= 0:
        print("Error: attn_length must be positive.")
        return
        
    start_time = time.time()

    # 我们只需要追踪从 0 到 target_pos 的可达性
    # We only need to track reachability from 0 to target_pos
    history_len = target_pos + 1
    reachable_from_target = torch.zeros(history_len, dtype=torch.bool, device=device)

    # --- 初始化：从最后一层开始 ---
    # --- Initialization: Start from the last layer ---
    last_layer_idx = num_hidden_layers - 1
    print(f"[{time.time() - start_time:.2f}s] Analyzing layer {last_layer_idx} (initialization)...")
    
    # 我们仍然需要为整个序列生成索引，因为较早的位置可能会关注较晚的位置
    # We still need to generate indices for the entire sequence, as earlier positions might attend to later ones
    # 但我们只关心 target_pos 的视野
    # But we only care about the receptive field of target_pos
    indices, mask = create_elastic_sparse_indices_for_validation(
        seq_len, last_layer_idx, num_hidden_layers, num_attention_heads, attn_length, device
    )

    target_direct_indices = indices[:, target_pos, :]
    target_direct_mask = mask[:, target_pos, :]
    
    initial_reachable_indices = target_direct_indices[target_direct_mask.bool()].unique()
    
    # 更新可达集合
    # Update the reachable set
    reachable_from_target[initial_reachable_indices] = True
    reachable_from_target[target_pos] = True
    
    print(f"[{time.time() - start_time:.2f}s] After layer {last_layer_idx}, pos {target_pos} can see {torch.sum(reachable_from_target)} tokens.")

    # --- 反向传播循环 ---
    # --- Backward propagation loop ---
    # 提前缓存所有层的索引以避免重复计算
    # Pre-cache indices for all layers to avoid recomputation
    all_indices = []
    all_masks = []
    print(f"[{time.time() - start_time:.2f}s] Pre-calculating all layer indices...")
    for layer_idx in range(num_hidden_layers - 2, -1, -1):
        indices, mask = create_elastic_sparse_indices_for_validation(
            seq_len, layer_idx, num_hidden_layers, num_attention_heads, attn_length, device
        )
        all_indices.append(indices)
        all_masks.append(mask)
    print(f"[{time.time() - start_time:.2f}s] Indices cached. Starting backward propagation...")


    for i, layer_idx in enumerate(range(num_hidden_layers - 2, -1, -1)):
        indices, mask = all_indices[i], all_masks[i]
        
        # b. 找到当前可达集合中的所有 token (这些是我们的“源” queries)
        # b. Find all tokens in the current reachable set (these are our "source" queries)
        # 注意：这里我们只关心在历史范围内的 token
        # Note: Here we only care about tokens within the history range
        source_query_indices = torch.where(reachable_from_target)[0]
        
        # c. 找到这些源 queries 在当前层能看到的所有 "新目标"
        # c. Find all "new targets" that these source queries can see at the current layer
        target_indices = indices[:, source_query_indices, :]
        target_mask = mask[:, source_query_indices, :]
        
        # d. 过滤出有效的新目标，并添加到总的可达集合中
        # d. Filter out valid new targets and add them to the total reachable set
        newly_reached = target_indices[target_mask.bool()].unique()
        
        # 我们只关心历史中的 token，所以需要clamp
        # We only care about tokens in history, so we need to clamp
        newly_reached = newly_reached[newly_reached < history_len]
        
        reachable_from_target[newly_reached] = True
        
        num_reachable = torch.sum(reachable_from_target)
        print(f"[{time.time() - start_time:.2f}s] After layer {layer_idx}, pos {target_pos} can see {num_reachable} tokens.")
        
        if num_reachable == history_len:
            print(f"[{time.time() - start_time:.2f}s] Full history reachability achieved. Stopping.")
            break

    # 4. 检查最终结果
    # 4. Check the final result
    end_time = time.time()
    total_time = end_time - start_time
    
    num_reachable_final = torch.sum(reachable_from_target)
    num_unreachable = history_len - num_reachable_final

    print(f"\n--- Test complete in {total_time:.2f} seconds ---")
    if num_unreachable == 0:
        print(f"✅ SUCCESS: Position {target_pos} has no attention holes. It can see all {history_len} previous tokens (including itself).")
    else:
        unreachable_indices_in_history = torch.where(~reachable_from_target)[0]
        print(f"❌ FAILURE: Position {target_pos} has {num_unreachable} attention holes in its history.")
        print(f"   It CANNOT see its complete history.")
        print(f"   Example unreachable indices: {unreachable_indices_in_history[:10].tolist()}")

if __name__ == "__main__":
    # --- 测试配置 ---
    # --- Test Configuration ---
    NUM_HIDDEN_LAYERS = 8
    NUM_ATTENTION_HEADS = 8
    ATTN_LENGTH = 64
    SEQ_LEN = 131072

    # --- 自定义测试 ---
    # --- Custom Test ---
    # 你可以在这里更改 position_id 来测试不同的位置
    # You can change position_id here to test different positions
    # position_id_to_test = None          # 默认检查最后一个位置 (131071) / Default: check the last position (131071)
    # position_id_to_test = 0             # 检查第一个位置 (只能看到自己) / Check the first position (can only see itself)
    # position_id_to_test = 1023          # 检查一个早期的位置 / Check an early position
    # position_id_to_test = 65537         # 检查一个中间的位置 / Check a middle position
    position_id_to_test = 131071         # 明确检查最后一个位置 / Explicitly check the last position

    check_attention_holes_pytorch(
        seq_len=SEQ_LEN,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        attn_length=ATTN_LENGTH,
        position_id=position_id_to_test
    )
    
    print("\n" + "="*50)
    print("Running another test for an early position to demonstrate functionality.")
    print("="*50)

    # 检查一个较早的位置
    # Check an earlier position
    check_attention_holes_pytorch(
        seq_len=SEQ_LEN,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        attn_length=ATTN_LENGTH,
        position_id=4095
    )

    print("\n" + "="*50)
    print("Running test with a potentially problematic smaller attn_length.")
    print("="*50)
    
    # 使用较小的 attn_length 检查中间位置
    # Check a middle position with a smaller attn_length
    check_attention_holes_pytorch(
        seq_len=SEQ_LEN,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        attn_length=16, # 这个值可能不足 / This value might be insufficient
        position_id=32767
    )
