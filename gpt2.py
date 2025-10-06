import math

from max import nn
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Linear
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged, fused_qkv_ragged_matmul
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import LayerList
from max.nn.norm import LayerNorm


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_embd: int,
        kv_params: KVCacheParams,
        device: DeviceRef | None = None,
        dtype: DType = DType.float32,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.kv_params = kv_params
        self.device = device or DeviceRef.CPU()
        self.scale = 1.0 / math.sqrt(n_embd // n_head)
        self.wqkv = Weight(
            name="c_attn.weight",
            dtype=dtype,
            shape=[3 * n_embd, n_embd],
            device=self.device,
        )
        self.wqkv_bias = Weight(
            name="c_attn.bias",
            dtype=dtype,
            shape=[3 * n_embd],
            device=self.device,
        )
        self.c_proj = Linear(
            in_dim=n_embd,
            out_dim=n_embd,
            has_bias=True,
            dtype=dtype,
            device=self.device,
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        total_seq_len = x.shape[0]
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            input_row_offsets=input_row_offsets,
            wqkv=self.wqkv,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_head,
            bias=self.wqkv_bias,
        )

        xq = xq.reshape((-1, self.n_head, self.n_embd // self.n_head))
        attn_out = flash_attention_ragged(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            layer_idx,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        y = self.c_proj(attn_out)
        return y


class MLP(nn.Module):
    def __init__(
        self, n_embd: int, device: DeviceRef | None = None, dtype: DType = DType.float32
    ):
        super().__init__()
        self.device = device or DeviceRef.CPU()
        self.c_fc = Linear(
            in_dim=n_embd,
            out_dim=n_embd * 4,
            has_bias=True,
            dtype=dtype,
            device=self.device,
        )
        self.c_proj = Linear(
            in_dim=n_embd * 4,
            out_dim=n_embd,
            has_bias=True,
            dtype=dtype,
            device=self.device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        y = ops.gelu(self.c_fc(x))
        y = self.c_proj(y)
        return y


class GPT2Block(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_embd: int,
        layer_norm_epsilon: float,
        kv_params: KVCacheParams,
        device: DeviceRef | None = None,
        dtype: DType = DType.float32,
    ):
        super().__init__()
        device = device or DeviceRef.CPU()
        self.attn = CausalSelfAttention(n_head, n_embd, kv_params, device, dtype)
        self.mlp = MLP(n_embd, device, dtype)
        self.ln_1 = LayerNorm(n_embd, device, dtype, eps=layer_norm_epsilon)
        self.ln_2 = LayerNorm(n_embd, device, dtype, eps=layer_norm_epsilon)

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        x = x + self.attn(layer_idx, self.ln_1(x), kv_collection, input_row_offsets)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        layer_norm_epsilon: float,
        kv_params: KVCacheParams,
        device: DeviceRef | None = None,
        dtype: DType = DType.float32,
    ):
        super().__init__()
        self.device = device or DeviceRef.CPU()
        self.n_ctx = n_ctx
        self.dtype = dtype
        self.wte = nn.Embedding(vocab_size, n_embd, dtype, device=self.device)
        self.wpe = nn.Embedding(n_ctx, n_embd, dtype, device=self.device)
        self.h = LayerList(
            [
                GPT2Block(
                    n_head, n_embd, layer_norm_epsilon, kv_params, self.device, dtype
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = LayerNorm(n_embd, self.device, dtype)
        self.lm_head = Linear(n_embd, vocab_size, dtype, self.device)
        self.lm_head.set_shared_weight("weight", self.wte.weight)

    def __call__(
        self,
        token_ids: TensorValue,
        pos: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        token_embd = self.wte(token_ids)
        pos_embd = self.wpe(pos)
        h = token_embd + pos_embd
        for i, layer in enumerate(self.h):
            h = layer(
                ops.constant(i, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                input_row_offsets,
            )
        last_h = ops.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self.lm_head(self.ln_f(last_h))
        return last_logits
