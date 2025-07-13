import os
import math
import torch
import matplotlib.pyplot as plt

from typing import Tuple

# To make the code run independently, we create a simple class to simulate Config
# 为了让代码独立运行，我们创建一个简单的类来模拟 Config
class MockConfig:
    def __init__(self, num_hidden_layers, num_attention_heads, attention_length):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_length = attention_length


def _create_elastic_sparse_indices(
    config: MockConfig, layer_idx: int, seq_len: int, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes and returns the indices and causal mask for elastic sparse attention.
    计算并返回弹性稀疏注意力的索引和因果掩码。
    """
    attn_length = config.attention_length
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    
    current_ids = torch.arange(seq_len, device=device)
    end_ids = torch.max(torch.tensor(attn_length, device=device) - 1, current_ids)
    
    denominator = num_hidden_layers - 1 if num_hidden_layers > 1 else 1
    alpha = layer_idx / denominator
    
    ids_len = end_ids + 1
    position_distance = attn_length * (1 - alpha) + ids_len * alpha
    start_ids_base = (end_ids + 1 - position_distance).round().clamp(min=0).long()
    window_lengths_base = end_ids + 1 - start_ids_base
    spacing_base = window_lengths_base / attn_length

    steps_oversized = torch.arange(attn_length + 1, device=device)
    
    relative_indices_oversized = (steps_oversized[None, :] * spacing_base[:, None]).round().long()
    attention_indices_oversized = start_ids_base[:, None] + relative_indices_oversized

    base = attention_indices_oversized[:, :attn_length]
    shifted = attention_indices_oversized[:, 1:] - 1

    head_indices_tensor = torch.arange(num_attention_heads, device=device)
    interp_weights = head_indices_tensor / (num_attention_heads - 1) if num_attention_heads > 1 else torch.zeros_like(head_indices_tensor)

    interp_weights_expanded = interp_weights.view(num_attention_heads, 1, 1)
    attention_indices_float = base.unsqueeze(0).float() * (1 - interp_weights_expanded) + \
                                shifted.unsqueeze(0).float() * interp_weights_expanded

    # Shape: [num_heads, seq_len, attn_length] / 形状: [注意力头数, 序列长度, 注意力长度]
    attention_indices = attention_indices_float.round().long()

    # Shape: [num_heads, seq_len, attn_length] / 形状: [注意力头数, 序列长度, 注意力长度]
    attention_mask = attention_indices > current_ids.view(1, -1, 1)
    
    return attention_indices, attention_mask

def _create_full_sparse_mask(
    config: MockConfig, layer_idx: int, seq_len: int, device: str = "cpu"
) -> torch.Tensor:
    """
    Calls the sparse index generation logic and converts it into a full-size attention mask.
    In the returned mask, `True` positions represent 'attention not allowed'.
    调用稀疏索引生成逻辑，并将其转换为一个全尺寸的注意力掩码。
    返回的掩码中，True 的位置代表“不允许注意力”。
    """
    original_indices, causal_mask_for_indices = _create_elastic_sparse_indices(config, layer_idx, seq_len, device)
    
    num_attention_heads = config.num_attention_heads
    full_mask = torch.ones(num_attention_heads, seq_len, seq_len, device=device, dtype=torch.bool)

    h_indices = torch.arange(num_attention_heads, device=device).view(-1, 1, 1).expand_as(original_indices)
    q_indices = torch.arange(seq_len, device=device).view(1, -1, 1).expand_as(original_indices)
    k_indices = original_indices.clamp(min=0, max=seq_len - 1)

    # Set all positions defined in the sparse pattern to False (allowed)
    # 将所有在稀疏模式中定义的位置设为 False (允许)
    full_mask[h_indices, q_indices, k_indices] = False

    # Now, re-mark those non-causal sparse connections as True (not allowed)
    # 现在，将那些非因果的稀疏连接重新标记为 True (不允许)
    non_causal_h = h_indices[causal_mask_for_indices]
    non_causal_q = q_indices[causal_mask_for_indices]
    non_causal_k = k_indices[causal_mask_for_indices]
    full_mask[non_causal_h, non_causal_q, non_causal_k] = True
    
    # Additionally, we need to apply the standard lower-triangular causal mask, as the sparse pattern might allow future tokens
    # 同时，我们还需要应用标准的下三角因果掩码，因为稀疏模式可能允许未来的token
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    old_full_mask = full_mask
    full_mask = torch.logical_or(full_mask, causal_mask.unsqueeze(0))
    assert torch.all(full_mask == old_full_mask)

    return full_mask


# ==============================================================================
# Visualization Functions
# 可视化函数
# ==============================================================================

def visualize_attention_pattern(config: MockConfig, layer_idx: int, seq_len: int):
    """
    Generates and displays the attention patterns for all heads in a specified layer.
    为指定的层生成并显示所有头的注意力模式。
    """
    print(f"Generating visualization for Layer {layer_idx}/{config.num_hidden_layers - 1}...")
    
    # Create the 'theory/figs' directory if it doesn't exist
    # 如果 'theory/figs' 文件夹不存在，则创建它
    output_dir = "theory/figs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate the full sparse attention mask
    # full_mask shape is [num_heads, seq_len, seq_len]
    # True means Masked (attention not allowed), False means Unmasked (attention allowed)
    # 1. 生成完整的稀疏注意力掩码
    # full_mask 的形状是 [num_heads, seq_len, seq_len]
    # True 表示 Masked (不允许关注), False 表示 Unmasked (允许关注)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_mask = _create_full_sparse_mask(config, layer_idx, seq_len, device=device)
    
    num_heads = config.num_attention_heads
    
    # 2. Prepare for plotting
    # Dynamically calculate the number of rows and columns for subplots
    # 2. 准备绘图
    # 动态计算子图的行列数
    if num_heads == 1:
        ncols = 1
        nrows = 1
    elif num_heads <= 4:
        ncols = num_heads
        nrows = 1
    else:
        ncols = 4
        nrows = math.ceil(num_heads / ncols)
        
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False)
    fig.suptitle(f'Elastic Sparse Attention Pattern - Layer {layer_idx}\n(Black = Masked , White = Allowed)', fontsize=16)

    # 3. Iterate through each head and plot its attention pattern
    # 3. 遍历每个头并绘制其注意力模式
    for head_idx in range(num_heads):
        row = head_idx // ncols
        col = head_idx % ncols
        ax = axes[row, col]

        # Get the mask for the current head [seq_len, seq_len]
        # 获取当前头的掩码 [seq_len, seq_len]
        head_mask = full_mask[head_idx].cpu().numpy()

        # Plot the mask
        # cmap='gray' means grayscale, True(1) is white, False(0) is black
        # 绘制掩码
        # cmap='gray' 表示灰度，True(1)为白, False(0)为黑
        ax.imshow(~head_mask, cmap='gray', interpolation='nearest')

        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

    # Hide unused subplots
    # 隐藏多余的子图
    for i in range(num_heads, nrows * ncols):
        row = i // ncols
        col = i % ncols
        fig.delaxes(axes[row][col])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- Save and Show Figure ---
    # --- 保存并显示图像 ---

    # Construct a descriptive filename
    # 构建一个描述性的文件名
    filename = (
        f"{output_dir}/layer_{layer_idx}_seq_{seq_len}_heads_{config.num_attention_heads}"
        f"_k_{config.attention_length}.png"
    )
    
    # Save the figure to the figs directory
    # 将图像保存到 figs 文件夹
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Image saved to: {filename}")
    
    # Display the plot
    # 显示图像
    plt.show()

    # Close the figure to free up memory
    # 关闭图形以释放内存
    plt.close(fig)


# ==============================================================================
# Main Program Entry
# 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # --- Modify parameters here ---
    # --- 在这里修改参数 ---
    SEQ_LEN = 512         # Sequence length / 序列长度
    NUM_LAYERS = 12       # Total number of layers in the model / 模型总层数
    NUM_HEADS = 8         # Number of attention heads / 注意力头的数量
    ATTENTION_LENGTH = 32 # Base length of the sparse attention window (k) / 稀疏注意力窗口的基本长度 (k)
    
    # Instantiate the mock config
    # 实例化模拟配置
    config = MockConfig(
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        attention_length=ATTENTION_LENGTH
    )

    # --- Select layers to visualize ---
    # You can call this function multiple times to compare different layers
    # --- 选择要可视化的层 ---
    # 你可以多次调用这个函数来比较不同层
    
    # Visualize the first layer (layer_idx = 0)
    # 可视化第一层 (layer_idx = 0)
    print("="*40)
    visualize_attention_pattern(config, layer_idx=0, seq_len=SEQ_LEN)
    
    # Visualize a middle layer
    # 可视化中间某一层
    print("="*40)
    visualize_attention_pattern(config, layer_idx=NUM_LAYERS // 2, seq_len=SEQ_LEN)
    
    # Visualize the last layer (layer_idx = NUM_LAYERS - 1)
    # 可视化最后一层 (layer_idx = NUM_LAYERS - 1)
    print("="*40)
    visualize_attention_pattern(config, layer_idx=NUM_LAYERS - 1, seq_len=SEQ_LEN)
