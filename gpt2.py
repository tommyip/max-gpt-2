import math

from max import nn
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Linear
from max.nn.attention import mask_config
from max.nn.kernels import flash_attention_gpu
from max.nn.norm import LayerNorm
from max.nn.layer import LayerList


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_embd: int,
        device: DeviceRef | None = None,
        dtype: DType = DType.float32,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.device = device or DeviceRef.CPU()
        self.c_attn = Linear(
            in_dim=n_embd,
            out_dim=n_embd * 3,
            has_bias=True,
            dtype=dtype,
            device=self.device,
        )
        self.c_proj = Linear(
            in_dim=n_embd,
            out_dim=n_embd,
            has_bias=True,
            dtype=dtype,
            device=self.device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        B, T, C = x.shape  # batch size, seq len, embedding dim
        assert C == self.n_embd

        qkv = self.c_attn(x)  # B, T, 3*n_embd
        q, k, v = ops.split(qkv, [self.n_embd] * 3, axis=2)  # (B, T, n_embd)
        k = ops.reshape(
            k, [B, T, self.n_head, self.n_embd // self.n_head]
        )  # (B, T, nh, hs)
        q = ops.reshape(
            q, [B, T, self.n_head, self.n_embd // self.n_head]
        )  # (B, T, nh, hs)
        v = ops.reshape(
            v, [B, T, self.n_head, self.n_embd // self.n_head]
        )  # (B, T, nh, hs)

        y = flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=mask_config.MHAMaskVariant.CAUSAL_MASK,
            scale=1 / math.sqrt(int(k.shape[-1])),
        )  # (B, T, nh, hs)
        y = y.reshape((B, T, C))
        y = self.c_proj(y)
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
        device: DeviceRef | None = None,
        dtype: DType = DType.float32,
    ):
        super().__init__()
        device = device or DeviceRef.CPU()
        self.attn = CausalSelfAttention(n_head, n_embd, device, dtype)
        self.mlp = MLP(n_embd, device, dtype)
        self.ln_1 = LayerNorm(n_embd, device, dtype, eps=layer_norm_epsilon)
        self.ln_2 = LayerNorm(n_embd, device, dtype, eps=layer_norm_epsilon)

    def __call__(self, x: TensorValue) -> TensorValue:
        x = x + self.attn(self.ln_1(x))
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
                GPT2Block(n_head, n_embd, layer_norm_epsilon, self.device, dtype)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = LayerNorm(n_embd, self.device, dtype)
        self.lm_head = Linear(n_embd, vocab_size, dtype, self.device, name="wte")
        # Tie weights
        self.lm_head.weight = self.wte.weight

    def __call__(self, token_ids: TensorValue) -> TensorValue:
        B, T = token_ids.shape
        pos = ops.range(0, T, out_dim=T, dtype=DType.int64, device=self.device)

        token_embd = self.wte(token_ids)  # (B, T, n_embd)
        pos_embd = self.wpe(pos)
        x = token_embd + pos_embd
        for layer in self.h:
            x = layer(x)
        x = self.ln_f(x)

        logits = self.lm_head(x[:, -1, :])
        return logits
