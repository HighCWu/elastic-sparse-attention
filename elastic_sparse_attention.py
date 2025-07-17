from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing necessary utilities from transformers, with a fallback to standard libraries.
# 尝试从 transformers.utils 导入必要的工具，如果环境没有 transformers，则回退到标准库。
try:
    from transformers.utils import logging
    from transformers.utils.import_utils import is_triton_available
except ImportError:
    import logging
    # A simple placeholder for is_triton_available if transformers is not installed.
    # 如果 transformers 未安装，提供一个简单的 is_triton_available 占位符。
    def is_triton_available():
        try:
            import triton
            return True
        except ImportError:
            return False

# Get a logger instance for this module.
# 获取此模块的日志记录器实例。
logger = logging.get_logger(__name__)

# ==============================================================================
#           BEGIN: OPTIMIZED & UNIFIED Triton Kernel Implementation
# ==============================================================================
if is_triton_available():
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_A': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_A': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_A': 256}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_A': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_A': 128}, num_warps=8, num_stages=2),
        ],
        key=['ATTN_LENGTH', 'HEAD_DIM'],
    )
    @triton.jit
    def _elastic_sparse_attention_kernel(
        # --- Tensors ---
        Q, K, V,
        sparse_indices, sparse_mask,
        sm_scale,
        L, M,
        O,
        # --- Backward Pass Tensors ---
        dO, dQ, dK, dV,

        # --- Strides ---
        stride_q_b, stride_q_h, stride_q_s,
        stride_k_b, stride_k_h, stride_k_s,
        stride_v_b, stride_v_h, stride_v_s,
        stride_idx_h, stride_idx_s,
        stride_mask_h, stride_mask_s,
        stride_o_b, stride_o_h, stride_o_s,
        stride_lm_b, stride_lm_h, stride_lm_s,
        
        # --- Backward Pass Strides ---
        stride_do_b, stride_do_h, stride_do_s,
        stride_dq_b, stride_dq_h, stride_dq_s,
        stride_dk_b, stride_dk_h, stride_dk_s,
        stride_dv_b, stride_dv_h, stride_dv_s,
        
        # --- Other metadata ---
        kv_seq_len: int,

        # --- Compile-time constants ---
        HEAD_DIM: tl.constexpr,
        ATTN_LENGTH: tl.constexpr,
        BLOCK_A: tl.constexpr,
        IS_BWD: tl.constexpr,
        # --- GQA Support ---
        NUM_KV_GROUPS: tl.constexpr,
    ):
        # Each program instance computes the attention for one query token in one head.
        # 每个程序实例计算一个头中一个查询 token 的注意力。
        q_pos = tl.program_id(0)      # Query token position / 查询 token 的位置
        head_idx = tl.program_id(1)   # Head index / 注意力头的索引
        batch_idx = tl.program_id(2)  # Batch index / 批次的索引

        # GQA: Map query head index to its corresponding key/value head index.
        # GQA: 将查询头索引映射到其对应的键/值头索引。
        kv_head_idx = head_idx // NUM_KV_GROUPS

        # Pointers to the current query, key, value, indices, and mask.
        # 指向当前 query, key, value, indices, 和 mask 的指针。
        q_offset = batch_idx * stride_q_b + head_idx * stride_q_h + q_pos * stride_q_s
        offs_d = tl.arange(0, HEAD_DIM)
        
        # GQA: Use kv_head_idx for K and V pointers.
        # GQA: 对 K 和 V 指针使用 kv_head_idx。
        k_base_ptr = K + batch_idx * stride_k_b + kv_head_idx * stride_k_h
        v_base_ptr = V + batch_idx * stride_v_b + kv_head_idx * stride_v_h
        
        indices_base_ptr = sparse_indices + head_idx * stride_idx_h + q_pos * stride_idx_s
        mask_base_ptr = sparse_mask + head_idx * stride_mask_h + q_pos * stride_mask_s
        
        lm_offset = batch_idx * stride_lm_b + head_idx * stride_lm_h + q_pos * stride_lm_s

        # Initialize accumulator, max score, and sum of exponentials for softmax.
        # 初始化累加器、最大分数和 softmax 的指数和。
        l_i = 0.0
        m_i = -float("inf")
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # Load the current query vector.
        # 加载当前的查询向量。
        q = tl.load(Q + q_offset + offs_d).to(tl.float32)

        # --- Forward Pass ---
        if not IS_BWD:
            # Iterate over the sparse key/value pairs in blocks.
            # 分块遍历稀疏的键/值对。
            for start_a in range(0, ATTN_LENGTH, BLOCK_A):
                offs_a = start_a + tl.arange(0, BLOCK_A)
                block_mask = offs_a < ATTN_LENGTH
                
                # Load sparse indices and the corresponding causal mask.
                # 加载稀疏索引和对应的因果掩码。
                k_indices = tl.load(indices_base_ptr + offs_a, mask=block_mask, other=0)
                causal_mask = tl.load(mask_base_ptr + offs_a, mask=block_mask, other=0).to(tl.int1)
                
                # Create a mask for safely gathering keys (within sequence bounds and causally valid).
                # 创建一个安全收集键的掩码（在序列边界内且因果有效）。
                k_gather_mask = block_mask[:, None] & causal_mask[:, None] & (k_indices[:, None] >= 0) & (k_indices[:, None] < kv_seq_len)
                k_offs = k_indices[:, None] * stride_k_s + offs_d[None, :]
                k = tl.load(k_base_ptr + k_offs, mask=k_gather_mask, other=0.0)
                
                # Compute dot-product scores.
                # 计算点积分数。
                scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
                scores = tl.where(block_mask & causal_mask, scores, -float("inf"))

                # Online softmax update.
                # 在线 softmax 更新。
                m_i_new = tl.maximum(m_i, tl.max(scores, axis=0))
                p = tl.exp(scores - m_i_new)
                alpha = tl.exp(m_i - m_i_new)
                
                acc = acc * alpha
                l_i = l_i * alpha + tl.sum(p, axis=0)
                m_i = m_i_new
                
                # Update accumulator with weighted values.
                # 用加权后的值更新累加器。
                v_offs = k_indices[:, None] * stride_v_s + offs_d[None, :]
                v = tl.load(v_base_ptr + v_offs, mask=k_gather_mask, other=0.0)
                acc += tl.sum(p[:, None] * v, axis=0)

            # Finalize attention output.
            # 完成注意力输出的计算。
            acc = acc / l_i
            
            # Store the output, and the L, M values for the backward pass.
            # 存储输出，以及用于反向传播的 L, M 值。
            o_ptr = O + batch_idx * stride_o_b + head_idx * stride_o_h + q_pos * stride_o_s
            tl.store(o_ptr + offs_d, acc.to(O.dtype.element_ty))
            
            tl.store(L + lm_offset, l_i)
            tl.store(M + lm_offset, m_i)

        # --- Backward Pass ---
        if IS_BWD:
            # GQA: Use kv_head_idx for dK and dV pointers.
            # GQA: 对 dK 和 dV 指针使用 kv_head_idx。
            dk_base_ptr = dK + batch_idx * stride_dk_b + kv_head_idx * stride_dk_h
            dv_base_ptr = dV + batch_idx * stride_dv_b + kv_head_idx * stride_dv_h

            # Load do, l, m, and o from the forward pass.
            # 从前向传播加载 do, l, m, 和 o。
            do_ptr = dO + batch_idx * stride_do_b + head_idx * stride_do_h + q_pos * stride_do_s
            do = tl.load(do_ptr + offs_d).to(tl.float32)
            
            l = tl.load(L + lm_offset)
            m = tl.load(M + lm_offset)
            
            o_ptr = O + batch_idx * stride_o_b + head_idx * stride_o_h + q_pos * stride_o_s
            o = tl.load(o_ptr + offs_d).to(tl.float32)
            
            do_dot_o = tl.sum(do * o, axis=0)
            dq_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

            # Re-compute attention scores to get gradients.
            # 重新计算注意力分数以获得梯度。
            for start_a in range(0, ATTN_LENGTH, BLOCK_A):
                offs_a = start_a + tl.arange(0, BLOCK_A)
                block_mask = offs_a < ATTN_LENGTH
                
                k_indices = tl.load(indices_base_ptr + offs_a, mask=block_mask, other=0)
                causal_mask = tl.load(mask_base_ptr + offs_a, mask=block_mask, other=0).to(tl.int1)
                
                k_gather_mask = block_mask[:, None] & causal_mask[:, None] & (k_indices[:, None] >= 0) & (k_indices[:, None] < kv_seq_len)
                k_offs = k_indices[:, None] * stride_k_s + offs_d[None, :]
                k = tl.load(k_base_ptr + k_offs, mask=k_gather_mask, other=0.0)
                
                scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
                p = tl.exp(scores - m) / l
                p = tl.where(block_mask & causal_mask, p, 0.0)

                v_offs = k_indices[:, None] * stride_v_s + offs_d[None, :]
                v = tl.load(v_base_ptr + v_offs, mask=k_gather_mask, other=0.0)

                # Compute dV: dV = P^T * dO
                # 计算 dV：dV = P^T * dO
                dv = p[:, None] * do[None, :]
                dv_offs = k_indices[:, None] * stride_dv_s + offs_d[None, :]
                tl.atomic_add(dv_base_ptr + dv_offs, dv, mask=k_gather_mask)
                
                # Compute dS = P * (dO^T * V - sum(dO * O))
                # 计算 dS = P * (dO^T * V - sum(dO * O))
                dp = tl.sum(do[None, :] * v, axis=1)
                ds = p * (dp - do_dot_o) * sm_scale

                # Compute dQ: dQ = dS * K
                # 计算 dQ: dQ = dS * K
                dq_acc += tl.sum(ds[:, None] * k, axis=0)
                
                # Compute dK: dK = dS^T * Q
                # 计算 dK: dK = dS^T * Q
                dk = ds[:, None] * q[None, :]
                dk_offs = k_indices[:, None] * stride_dk_s + offs_d[None, :]
                tl.atomic_add(dk_base_ptr + dk_offs, dk, mask=k_gather_mask)

            # Store the final dQ gradient.
            # 存储最终的 dQ 梯度。
            dq_ptr = dQ + batch_idx*stride_dq_b + head_idx*stride_dq_h + q_pos*stride_dq_s
            tl.store(dq_ptr + offs_d, dq_acc.to(dQ.dtype.element_ty))

    class ElasticSparseAttentionFunction(torch.autograd.Function):
        """
        A custom autograd function to wrap the Triton kernel for elastic sparse attention.
        一个自定义的 autograd 函数，用于封装弹性稀疏注意力的 Triton 内核。
        """
        @staticmethod
        def forward(ctx, q, k, v, sparse_indices, sparse_mask, sm_scale, num_kv_groups):
            # Ensure tensors are contiguous for kernel performance.
            # 确保张量是连续的，以保证内核性能。
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            sparse_indices, sparse_mask = sparse_indices.contiguous(), sparse_mask.contiguous()

            # Get tensor shapes and parameters.
            # 获取张量形状和参数。
            batch_size, num_heads, q_seq_len, head_dim = q.shape
            kv_seq_len = k.shape[2]
            attn_length = sparse_indices.shape[-1]
            
            # Create output and intermediate tensors.
            # 创建输出和中间张量。
            o = torch.empty_like(q)
            l = torch.empty((batch_size, num_heads, q_seq_len), device=q.device, dtype=torch.float32)
            m = torch.empty((batch_size, num_heads, q_seq_len), device=q.device, dtype=torch.float32)
            
            # Define the grid for launching the Triton kernel.
            # 定义用于启动 Triton 内核的网格。
            grid = (q_seq_len, num_heads, batch_size)
            
            _elastic_sparse_attention_kernel[grid](
                Q=q, K=k, V=v,
                sparse_indices=sparse_indices, sparse_mask=sparse_mask,
                sm_scale=sm_scale,
                L=l, M=m, O=o,
                # Gradients are not needed in forward pass.
                # 前向传播不需要梯度。
                dO=None, dQ=None, dK=None, dV=None,
                # Pass tensor strides to the kernel.
                # 将张量步幅传递给内核。
                stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_s=q.stride(2),
                stride_k_b=k.stride(0), stride_k_h=k.stride(1), stride_k_s=k.stride(2),
                stride_v_b=v.stride(0), stride_v_h=v.stride(1), stride_v_s=v.stride(2),
                stride_idx_h=sparse_indices.stride(0), stride_idx_s=sparse_indices.stride(1),
                stride_mask_h=sparse_mask.stride(0), stride_mask_s=sparse_mask.stride(1),
                stride_o_b=o.stride(0), stride_o_h=o.stride(1), stride_o_s=o.stride(2),
                stride_lm_b=l.stride(0), stride_lm_h=l.stride(1), stride_lm_s=l.stride(2),
                # Strides for gradients are zero as they are not used.
                # 梯度的步幅为零，因为未使用。
                stride_do_b=0, stride_do_h=0, stride_do_s=0, stride_dq_b=0, stride_dq_h=0, stride_dq_s=0,
                stride_dk_b=0, stride_dk_h=0, stride_dk_s=0, stride_dv_b=0, stride_dv_h=0, stride_dv_s=0,
                # Pass metadata and compile-time constants.
                # 传递元数据和编译时常量。
                kv_seq_len=kv_seq_len, HEAD_DIM=head_dim, ATTN_LENGTH=attn_length, IS_BWD=False,
                NUM_KV_GROUPS=num_kv_groups,
            )
            
            # Save tensors for the backward pass.
            # 为反向传播保存张量。
            ctx.save_for_backward(q, k, v, o, l, m, sparse_indices, sparse_mask)
            ctx.sm_scale = sm_scale
            ctx.head_dim = head_dim
            ctx.attn_length = attn_length
            ctx.num_kv_groups = num_kv_groups
            return o

        @staticmethod
        def backward(ctx, do):
            do = do.contiguous()
            q, k, v, o, l, m, sparse_indices, sparse_mask = ctx.saved_tensors
            batch_size, num_heads, q_seq_len, _ = q.shape

            # Initialize gradient tensors. dK and dV are initialized to zero for atomic adds.
            # 初始化梯度张量。dK 和 dV 初始化为零，以便进行原子加法。
            dq = torch.empty_like(q)
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            
            grid = (q_seq_len, num_heads, batch_size)
            
            _elastic_sparse_attention_kernel[grid](
                # Pass all tensors required for both forward recomputation and backward calculation.
                # 传递前向重计算和反向计算所需的所有张量。
                Q=q, K=k, V=v,
                sparse_indices=sparse_indices, sparse_mask=sparse_mask,
                sm_scale=ctx.sm_scale,
                L=l, M=m, O=o,
                dO=do, dQ=dq, dK=dk, dV=dv,
                # Pass all tensor strides.
                # 传递所有张量步幅。
                stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_s=q.stride(2),
                stride_k_b=k.stride(0), stride_k_h=k.stride(1), stride_k_s=k.stride(2),
                stride_v_b=v.stride(0), stride_v_h=v.stride(1), stride_v_s=v.stride(2),
                stride_idx_h=sparse_indices.stride(0), stride_idx_s=sparse_indices.stride(1),
                stride_mask_h=sparse_mask.stride(0), stride_mask_s=sparse_mask.stride(1),
                stride_o_b=o.stride(0), stride_o_h=o.stride(1), stride_o_s=o.stride(2),
                stride_lm_b=l.stride(0), stride_lm_h=l.stride(1), stride_lm_s=l.stride(2),
                stride_do_b=do.stride(0), stride_do_h=do.stride(1), stride_do_s=do.stride(2),
                stride_dq_b=dq.stride(0), stride_dq_h=dq.stride(1), stride_dq_s=dq.stride(2),
                stride_dk_b=dk.stride(0), stride_dk_h=dk.stride(1), stride_dk_s=dk.stride(2),
                stride_dv_b=dv.stride(0), stride_dv_h=dv.stride(1), stride_dv_s=dv.stride(2),
                kv_seq_len=k.shape[2], HEAD_DIM=ctx.head_dim, ATTN_LENGTH=ctx.attn_length, IS_BWD=True,
                NUM_KV_GROUPS=ctx.num_kv_groups,
            )
            # Return gradients for q, k, v. Other inputs don't require gradients.
            # 返回 q, k, v 的梯度。其他输入不需要梯度。
            return dq, dk, dv, None, None, None, None

# ==============================================================================
#            END: OPTIMIZED & UNIFIED Triton Kernel Implementation
# ==============================================================================

class ElasticSparseAttention(nn.Module):
    """
    A standalone Elastic Sparse Attention module, with support for Grouped-Query Attention (GQA).
    一个独立的弹性稀疏注意力（Elastic Sparse Attention）模块，支持分组查询注意力（GQA）。

    This module is abstracted from a full attention mechanism, focusing on the core
    computation of elastic sparse attention. It does not include any linear projection layers,
    normalization layers, or positional encodings, and assumes that the input Q, K, V
    tensors are already prepared.
    该模块从完整的注意力机制中抽离，专注于弹性稀疏注意力的核心计算。
    它不包含任何线性投射层、归一化层或位置编码，并假定输入的 Q, K, V
    张量已经准备就绪。

    It allows choosing between three implementations:
    可以在三种实现之间进行选择：
    1. 'triton': A highly optimized Triton kernel for best performance.
       'triton': 高度优化的 Triton 内核，性能最佳。
    2. 'naive': A pure PyTorch implementation using `torch.take_along_dim` to gather sparse keys and values.
       'naive': 一个纯 PyTorch 实现，使用 `torch.take_along_dim` 来收集稀疏的键和值。
    3. 'dense': Simulates the sparse pattern by constructing a full attention mask and using PyTorch 2.x's
               `scaled_dot_product_attention` for computation, mainly for debugging and validation.
       'dense': 通过构建一个完整的注意力掩码来模拟稀疏模式，并使用 PyTorch 2.x
               的 `scaled_dot_product_attention` 进行计算，主要用于调试和验证。
               
    Caching Mechanism: The module pre-computes and caches the sparse indices and masks for all positions
    up to `max_seq_len` during initialization. These are then sliced directly during the `forward`
    pass to improve performance.
    缓存机制: 模块在初始化时会根据 `max_seq_len` 预先计算并缓存
    所有位置的稀疏索引和掩码，在 `forward` 传递中直接切片使用，以提高性能。
    """
    def __init__(
        self,
        layer_idx: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        head_dim: int,
        attention_length: int,
        max_seq_len: int,
        num_key_value_heads: Optional[int] = None,
        attention_dropout: float = 0.0,
        implementation: str = "triton",
    ):
        """
        Initializes the ElasticSparseAttention module.
        初始化 ElasticSparseAttention 模块。

        Args:
            layer_idx (`int`): The index of the current layer (starting from 0). // 当前层的索引 (从 0 开始)。
            num_hidden_layers (`int`): The total number of layers in the model. // 模型中的总层数。
            num_attention_heads (`int`): The number of attention heads for queries (Q). // 查询（Q）的注意力头数量。
            head_dim (`int`): The dimension of each attention head. // 每个注意力头的维度。
            attention_length (`int`): The window length of the sparse attention pattern (w). // 稀疏注意力模式的窗口长度 (w)。
            max_seq_len (`int`): The maximum sequence length supported by the model, used for pre-computing and caching indices/masks. // 模型支持的最大序列长度，用于预计算和缓存索引/掩码。
            num_key_value_heads (`Optional[int]`, *optional*):
                The number of attention heads for keys and values (K, V). If not provided, defaults to `num_attention_heads` (MHA).
                For GQA, `num_attention_heads` must be divisible by `num_key_value_heads`.
                键（K）和值（V）的注意力头数量。如果未提供，则默认为 `num_attention_heads`（MHA）。
                对于 GQA，`num_attention_heads` 必须能被 `num_key_value_heads` 整除。
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout probability applied to the attention weights. // 应用于注意力权重的 dropout 概率。
            implementation (`str`, *optional*, defaults to "triton"):
                The implementation to use. Options are "triton", "naive", "dense".
                If "triton" is chosen but not available, it will automatically fall back to "naive".
                要使用的实现。可选值为 "triton", "naive", "dense"。
                如果选择 "triton" 但环境不可用，将自动回退到 "naive"。
        """
        super().__init__()
        valid_implementations = ["triton", "naive", "dense"]
        if implementation not in valid_implementations:
            raise ValueError(
                f"Invalid implementation '{implementation}'. Must be one of {valid_implementations}."
                f" // 无效的实现 '{implementation}'。必须是 {valid_implementations} 中的一个。"
            )
        if implementation == "triton" and not is_triton_available():
            logger.warning(
                "Triton is not available, falling back to 'naive' implementation."
                " // Triton 不可用，回退到 'naive' 实现。"
            )
            implementation = "naive"

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.attention_length = attention_length
        self.attention_dropout = attention_dropout
        self.implementation = implementation
        self.max_seq_len = max_seq_len
        
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"`num_attention_heads` ({self.num_attention_heads}) must be divisible by "
                f"`num_key_value_heads` ({self.num_key_value_heads}) for GQA."
                f" // 对于 GQA, `num_attention_heads` ({self.num_attention_heads}) 必须能被 "
                f"`num_key_value_heads` ({self.num_key_value_heads}) 整除。"
            )
        self.num_kv_groups = self.num_attention_heads // self.num_key_value_heads
        
        self.scaling = self.head_dim**-0.5

        # --- Pre-caching indices and masks ---
        # Pre-compute all possible indices and masks at initialization and register them as buffers.
        # Buffers are moved to the correct device with the model (e.g., model.to('cuda')).
        # We generate them on the CPU to avoid needing a GPU during initialization.
        # --- 预缓存索引和掩码 ---
        # 在初始化时预先计算所有可能的索引和掩码，并将其注册为 buffer。
        # buffer 会随模型移动到正确的设备 (e.g., model.to('cuda'))。
        # 我们在 CPU 上生成，以避免在初始化时需要 GPU。
        device = torch.device("cpu")
        if implementation == "dense":
            # For 'dense' mode, pre-compute the full boolean attention mask.
            # 对于 'dense' 模式，预计算完整的布尔注意力掩码。
            full_mask = self._create_full_sparse_mask(self.max_seq_len, device=device)
            self.register_buffer("full_attention_mask", full_mask, persistent=False)
            self.register_buffer("sparse_attention_indices", None, persistent=False)
            self.register_buffer("sparse_causal_mask", None, persistent=False)
        else:
            # For 'triton' and 'naive' modes, pre-compute sparse indices and causal masks.
            # 对于 'triton' 和 'naive' 模式，预计算稀疏索引和因果掩码。
            indices, mask = self._create_elastic_sparse_indices(self.max_seq_len, device=device)
            self.register_buffer("sparse_attention_indices", indices, persistent=False)
            self.register_buffer("sparse_causal_mask", mask, persistent=False)
            self.register_buffer("full_attention_mask", None, persistent=False)

    def _create_elastic_sparse_indices(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes and returns the indices and mask for elastic sparse attention.
        The pattern is generated per query head.
        计算并返回弹性稀疏注意力的索引和掩码。
        稀疏模式是为每个查询头生成的。
        """
        attn_length = self.attention_length
        num_hidden_layers = self.num_hidden_layers
        
        # Base position IDs for each query position.
        # 每个查询位置的基础位置 ID。
        current_ids = torch.arange(seq_len, device=device)
        end_ids = torch.max(torch.tensor(attn_length, device=device) - 1, current_ids)
        
        # Calculate alpha, the interpolation factor based on layer depth.
        # alpha is 0 for the first layer, 1 for the last layer.
        # 计算 alpha, 基于层深度的插值因子。第一层为 0，最后一层为 1。
        denominator = num_hidden_layers - 1 if num_hidden_layers > 1 else 1
        alpha = self.layer_idx / denominator
        
        # Linearly interpolate the attention window size based on alpha.
        # Early layers have a larger, more local window. Later layers have a smaller, more dilated window.
        # 根据 alpha 线性插值注意力窗口大小。
        # 早期层有更大、更局部的窗口。后期层有更小、更扩展的窗口。
        ids_len = end_ids + 1
        position_distance = attn_length * (1 - alpha) + ids_len * alpha
        start_ids_base = (end_ids + 1 - position_distance).round().clamp(min=0).long()
        window_lengths_base = end_ids + 1 - start_ids_base
        spacing_base = window_lengths_base / attn_length

        # To avoid floating point inaccuracies, we create an oversized set of indices and then interpolate.
        # 为避免浮点不精确，我们创建一个超大尺寸的索引集然后进行插值。
        steps_oversized = torch.arange(attn_length + 1, device=device)
        
        relative_indices_oversized = (steps_oversized[None, :] * spacing_base[:, None]).round().long()
        attention_indices_oversized = start_ids_base[:, None] + relative_indices_oversized

        # The 'base' indices and 'shifted' indices for interpolation between heads.
        # 用于在不同头之间插值的 'base' 索引和 'shifted' 索引。
        base = attention_indices_oversized[:, :attn_length]
        shifted = attention_indices_oversized[:, 1:] - 1

        # Create interpolation weights for each head. Head 0 uses 'base', last head uses 'shifted'.
        # 为每个头创建插值权重。头 0 使用 'base'，最后一个头使用 'shifted'。
        head_indices_tensor = torch.arange(self.num_attention_heads, device=device)
        interp_weights = head_indices_tensor / (self.num_attention_heads - 1) if self.num_attention_heads > 1 else torch.zeros_like(head_indices_tensor)

        # Interpolate between base and shifted indices to get the final indices for each head.
        # 在 base 和 shifted 索引之间插值，得到每个头的最终索引。
        interp_weights_expanded = interp_weights.view(self.num_attention_heads, 1, 1)
        attention_indices_float = base.unsqueeze(0).float() * (1 - interp_weights_expanded) + \
                                  shifted.unsqueeze(0).float() * interp_weights_expanded

        # Final indices are rounded and cast to long. Shape: [num_heads, seq_len, attn_length]
        # 最终索引四舍五入并转换为 long。形状: [num_heads, seq_len, attn_length]
        attention_indices = attention_indices_float.round().long()
        
        # Create a causal mask: an attended key position must be <= the query position.
        # Shape: [num_heads, seq_len, attn_length]
        # 创建因果掩码：被注意的键的位置必须 <= 查询的位置。
        # 形状: [num_heads, seq_len, attn_length]
        attention_mask = attention_indices <= current_ids.view(1, -1, 1)
        attention_indices.masked_fill_(~attention_mask, 0) # Set invalid indices to 0 for safety.
        
        return attention_indices, attention_mask.long()

    def _create_full_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Calls the sparse index generation logic and converts it into a full-sized boolean attention mask.
        The mask is created with `num_attention_heads` as SDPA expects a mask matching the query heads for GQA.
        调用稀疏索引生成逻辑，并将其转换为一个全尺寸的布尔注意力掩码。
        掩码的头数量与 `num_attention_heads` 匹配，因为 SDPA 在 GQA 模式下期望掩码与查询头对齐。
        """
        original_indices, causal_mask_for_indices = self._create_elastic_sparse_indices(seq_len, device)
        
        # Create a boolean mask where True means attention is allowed.
        # 创建布尔掩码，True 表示允许注意力。
        full_mask = torch.zeros(self.num_attention_heads, seq_len, seq_len, device=device, dtype=torch.bool)

        # Use advanced indexing to set the sparse locations to True.
        # 使用高级索引将稀疏模式指定的位置设置为 True。
        h_indices = torch.arange(self.num_attention_heads, device=device).view(-1, 1, 1).expand_as(original_indices)
        q_indices = torch.arange(seq_len, device=device).view(1, -1, 1).expand_as(original_indices)
        k_indices = original_indices.clamp(min=0, max=seq_len - 1)
        
        # The causal_mask_for_indices tells us which of the sparse connections are valid.
        # causal_mask_for_indices 告诉我们哪些稀疏连接是有效的。
        is_causal_mask = causal_mask_for_indices.bool()

        # Use the causal mask to select only the valid indices before assigning them.
        # This avoids populating the full_mask with non-causal locations in the first place.
        # 使用因果掩码在赋值前只选择有效的索引。
        # 这从一开始就避免了用非因果位置填充 full_mask。
        h_indices_valid = h_indices[is_causal_mask]
        q_indices_valid = q_indices[is_causal_mask]
        k_indices_valid = k_indices[is_causal_mask]

        full_mask[h_indices_valid, q_indices_valid, k_indices_valid] = True
        
        # Return with a batch dimension to match SDPA's expected format.
        # 返回时增加一个批次维度，以匹配 SDPA 的期望格式。
        return full_mask.unsqueeze(0)

    def _get_prepared_indices_and_mask(self, q_seq_len: int, kv_seq_len: int):
        """
        Slices the required indices and mask for the current sequence length from the pre-computed cache.
        从预计算的缓存中切片出当前序列长度所需的索引和掩码。
        """
        if kv_seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({kv_seq_len}) exceeds the maximum length initialized "
                f"in the module ({self.max_seq_len})."
                f" // 输入序列长度 ({kv_seq_len}) 超过了模块初始化的最大长度 ({self.max_seq_len})。"
            )
        
        # Slice the pre-computed tensors.
        # 切片预计算的张量。
        indices = self.sparse_attention_indices[:, :q_seq_len, :]
        mask = self.sparse_causal_mask[:, :q_seq_len, :]
        return indices, mask

    def _triton_forward(self, query_states, key_states, value_states, cache_position):
        """Forward pass using the Triton implementation."""
        """使用 Triton 实现的前向传播。"""
        q_seq_len = query_states.shape[2]
        kv_seq_len = key_states.shape[2]
        
        # Get indices and mask from cache instead of dynamic generation.
        # 从缓存中获取索引和掩码，而不是动态生成。
        attention_indices, attention_mask = self._get_prepared_indices_and_mask(q_seq_len, kv_seq_len)

        if cache_position is not None:
            # For decoding, we only need the indices and mask for the current position.
            # 对于解码，我们只需要当前位置的索引和掩码。
            # For decoding, we only need the indices and mask for the current position.
            # The slicing here is correct because cache_position corresponds to the q_seq_len dimension.
            attention_indices = attention_indices[:, cache_position, :]
            attention_mask = attention_mask[:, cache_position, :]
        else:
            attention_indices = attention_indices[:, :q_seq_len]
            attention_mask = attention_mask[:, :q_seq_len]

        # Clamp indices to prevent out-of-bounds access in case of kv_cache length mismatch.
        # 防止 kv cache 长度和索引不匹配的边缘情况，对索引进行钳位。
        attention_indices = torch.clamp(attention_indices, min=0, max=kv_seq_len - 1)
        
        attn_output = ElasticSparseAttentionFunction.apply(
            query_states, key_states, value_states, attention_indices, attention_mask, self.scaling, self.num_kv_groups
        )
        # The Triton kernel does not return attention weights to save computation and memory.
        # Triton 内核不返回注意力权重以节省计算和内存。
        return attn_output, None

    def _naive_forward(self, query_states, key_states, value_states, cache_position):
        """Forward pass using the 'naive' PyTorch implementation."""
        """使用 'naive' PyTorch 实现的前向传播。"""
        batch_size, num_heads, q_seq_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[2]
        
        # GQA: Repeat K and V heads to match Q heads
        # GQA: 重复 K 和 V 的头，使其数量与 Q 的头匹配
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # Get indices and mask from cache.
        # 从缓存中获取索引和掩码。
        attention_indices, attention_mask = self._get_prepared_indices_and_mask(q_seq_len, kv_seq_len)
        
        if cache_position is not None:
            # For decoding, we only need the indices and mask for the current position.
            # 对于解码，我们只需要当前位置的索引和掩码。
            # For decoding, we only need the indices and mask for the current position.
            # The slicing here is correct because cache_position corresponds to the q_seq_len dimension.
            attention_indices = attention_indices[:, cache_position, :]
            attention_mask = attention_mask[:, cache_position, :]
        else:
            attention_indices = attention_indices[:, :q_seq_len]
            attention_mask = attention_mask[:, :q_seq_len]
        current_indices = attention_indices
        current_mask = attention_mask

        # Expand indices to gather along the head_dim dimension.
        # 扩展索引以便沿 head_dim 维度进行收集。
        idx = current_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1, head_dim)
        
        # Gather the sparse key and value vectors based on the computed indices.
        # 根据计算出的索引收集稀疏的键和值向量。
        gathered_keys = torch.take_along_dim(key_states.unsqueeze(2), idx, dim=3)
        gathered_values = torch.take_along_dim(value_states.unsqueeze(2), idx, dim=3)

        # Compute attention scores.
        # 计算注意力分数。
        scores = torch.matmul(query_states.unsqueeze(3), gathered_keys.transpose(-1, -2)).squeeze(3)
        scores *= self.scaling
        
        # Apply the causal mask.
        # 应用因果掩码。
        scores.masked_fill_(current_mask.unsqueeze(0) < 1, torch.finfo(scores.dtype).min)

        # Compute softmax and apply dropout.
        # 计算 softmax 并应用 dropout。
        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(query_states)
        attn_weights = torch.dropout(attn_weights, p=self.attention_dropout, train=self.training)
        
        # Compute attention output.
        # 计算注意力输出。
        attn_out = torch.matmul(attn_weights.unsqueeze(-2), gathered_values).squeeze(-2)
        
        return attn_out, attn_weights

    def _dense_forward(self, query_states, key_states, value_states, cache_position):
        """
        Forward pass using the 'dense' implementation with PyTorch's SDPA.
        No change is needed for GQA as F.scaled_dot_product_attention handles it automatically.
        使用 'dense' 实现和 PyTorch 的 SDPA 进行前向传播。
        GQA 无需修改，因为 F.scaled_dot_product_attention 会自动处理。
        """
        q_seq_len = query_states.shape[2]
        kv_seq_len = key_states.shape[2]

        # The max length check for dense is done inside the _get_prepared_... function.
        # Re-checking here for clarity, though it might be redundant.
        # dense 实现的最大长度检查已在 _create_full_sparse_mask 中隐式完成，但为清晰起见在此重申。
        if kv_seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({kv_seq_len}) exceeds the maximum length initialized "
                f"in the module ({self.max_seq_len})."
                f" // 输入序列长度 ({kv_seq_len}) 超过了模块初始化的最大长度 ({self.max_seq_len})。"
            )
        
        # Slice the required portion of the pre-computed mask.
        # Mask shape is [1, num_attention_heads, max_seq_len, max_seq_len].
        # 从预计算的掩码中切片出需要的部分。
        # 掩码形状为 [1, num_attention_heads, max_seq_len, max_seq_len]。
        attention_mask = self.full_attention_mask[:, :, :q_seq_len, :kv_seq_len]
        
        if cache_position is not None:
            # Slice for decoding mode.
            # 为解码模式进行切片。
            attention_mask = attention_mask[:, :, cache_position, :]

        # Call the native SDPA function.
        # 调用原生的 SDPA 函数。
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scaling,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            enable_gqa=True,
        )
        # SDPA does not return attention weights by default when a mask is provided.
        # 当提供掩码时，SDPA 默认不返回注意力权重。
        return attn_output, None

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs the forward pass of elastic sparse attention.
        执行弹性稀疏注意力的前向传播。

        Args:
            query_states (`torch.Tensor`): Query tensor of shape (bsz, num_heads, q_len, head_dim). // 查询张量，形状为 (bsz, num_heads, q_len, head_dim)。
            key_states (`torch.Tensor`): Key tensor of shape (bsz, num_kv_heads, kv_len, head_dim). // 键张量，形状为 (bsz, num_kv_heads, kv_len, head_dim)。
            value_states (`torch.Tensor`): Value tensor of shape (bsz, num_kv_heads, kv_len, head_dim). // 值张量，形状为 (bsz, num_kv_heads, kv_len, head_dim)。
            cache_position (`Optional[torch.LongTensor]`): The positions of the query tokens, used for decoding. // 查询 token 的位置，用于解码。

        Returns:
            `tuple(torch.Tensor, Optional[torch.Tensor])`:
            - Attention output tensor of shape (bsz, q_len, hidden_dim). // 注意力输出张量，形状为 (bsz, q_len, hidden_dim)。
            - Attention weights tensor (if supported by the implementation). 'triton' and 'dense' (with SDPA) return None. // 注意力权重张量（如果实现支持）。'triton' 和 'dense' (使用SDPA) 实现返回 None。
        """
        # Dispatch to the selected implementation.
        # 分派到选定的实现。
        if self.implementation == "triton":
            attn_output, attn_weights = self._triton_forward(query_states, key_states, value_states, cache_position)
        elif self.implementation == "naive":
            attn_output, attn_weights = self._naive_forward(query_states, key_states, value_states, cache_position)
        else:  # dense
            attn_output, attn_weights = self._dense_forward(query_states, key_states, value_states, cache_position)
        
        # Reshape the output back to (bsz, q_len, hidden_dim) to match standard transformer block outputs.
        # 将输出 reshape 回 (bsz, q_len, hidden_dim) 以匹配标准 Transformer 块的输出。
        batch_size, _, q_len, head_dim = query_states.shape
        hidden_dim = self.num_attention_heads * head_dim
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, q_len, hidden_dim)

        return attn_output, attn_weights
