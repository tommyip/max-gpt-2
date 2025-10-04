from dataclasses import dataclass
from typing import Any, Sequence, cast, override

import numpy as np
from max.engine.api import InferenceSession, Model
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParams,
    PagedKVCacheManager,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.lib import (
    PipelineModel,
    PipelineConfig,
    ModelInputs,
    ModelOutputs,
    SupportedEncoding,
    KVCacheConfig,
    KVCacheMixin,
    MAXModelConfigBase,
    MAXModelConfig,
    upper_bounded_default,
)
from max.pipelines.core import TextContext
from max.driver import Device, Tensor
from max.dtype import DType
from max.graph import Graph, DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from transformers import AutoConfig
from safetensors import safe_open

from .gpt2 import GPT2


@dataclass
class GPT2ConfigBase(MAXModelConfigBase):
    bos_token_id: int
    eos_token_id: int
    initializer_range: float
    layer_norm_epsilon: float
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int
    n_positions: int
    vocab_size: int


@dataclass
class GPT2Config(MAXModelConfig, GPT2ConfigBase):
    @staticmethod
    def generate(huggingface_config: AutoConfig):
        return GPT2ConfigBase(
            bos_token_id=huggingface_config.bos_token_id,
            eos_token_id=huggingface_config.eos_token_id,
            initializer_range=huggingface_config.initializer_range,
            layer_norm_epsilon=huggingface_config.layer_norm_epsilon,
            n_ctx=huggingface_config.n_ctx,
            n_embd=huggingface_config.n_embd,
            n_head=huggingface_config.n_head,
            n_layer=huggingface_config.n_layer,
            n_positions=huggingface_config.n_positions,
            vocab_size=huggingface_config.vocab_size,
        )

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.n_head,
            head_dim=huggingface_config.n_embd // huggingface_config.n_head,
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            n_devices=n_devices,
            data_parallel_degree=1,
        )


class GPT2Inputs(ModelInputs):
    tokens: Tensor
    input_row_offsets: Tensor

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
    ):
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.kv_cache_inputs = kv_cache_inputs


class GPT2Model(PipelineModel[TextContext], KVCacheMixin):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None,
        return_logits: ReturnLogits,
    ):
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        self.devices = devices
        self.model = self.load_model(session)

    def load_model(self, session: InferenceSession) -> Model:
        max_batch_size = self.pipeline_config.max_batch_size
        assert max_batch_size
        self._input_row_offsets = Tensor.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        assert self.adapter
        state_dict = self.adapter(
            dict(self.weights.items()),
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
        )
        self.config = config = GPT2Config.generate(self.huggingface_config)
        device_spec = self.pipeline_config.model_config.device_specs[0]
        device = DeviceRef(device_spec.device_type, device_spec.id)
        model_spec = GPT2(
            n_ctx=config.n_ctx,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            layer_norm_epsilon=config.layer_norm_epsilon,
            dtype=DType.float32,
            device=device,
        )
        model_spec.load_state_dict(state_dict, weight_alignment=1)
        input_type = TensorType(
            DType.int64, shape=["batch_size", "total_seq_len"], device=device
        )
        with Graph("gpt2", input_types=(input_type,)) as graph:
            token_ids = graph.inputs[0]
            output = model_spec(token_ids)
            graph.output(output)

        model = session.load(graph, weights_registry=state_dict)

        return model

    @classmethod
    @override
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.n_ctx,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for GPT-2, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_seq_len ({huggingface_config.max_seq_len})."
            ) from e

    @override
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, GPT2Inputs)

        (logits,) = self.model.execute(model_inputs.tokens)

        assert isinstance(logits, Tensor)
        return ModelOutputs(logits=logits)

    @override
    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        assert kv_cache_inputs is not None
        kv_cache_inputs = cast(KVCacheInputsSequence, kv_cache_inputs)

        input_row_offsets_np = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch], dtype=np.uint32
        )
        input_row_offsets = Tensor.from_numpy(input_row_offsets_np).to(self.devices[0])

        tokens_np = np.stack([ctx.next_tokens for ctx in context_batch])
        tokens = Tensor.from_numpy(tokens_np).to(self.devices[0])

        print(
            f"prepare_initial_token_inputs tokens={tokens_np} input_row_offsets={input_row_offsets_np}"
        )

        return GPT2Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            kv_cache_inputs=kv_cache_inputs,
        )

    @override
    def prepare_next_token_inputs(
        self, next_tokens: Tensor, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        assert isinstance(prev_model_inputs, GPT2Inputs)
        prev_tokens = prev_model_inputs.tokens.to_numpy()
        tokens_np = np.hstack(
            [prev_tokens, np.expand_dims(next_tokens.to_numpy(), axis=1)]
        )
        tokens = Tensor.from_numpy(tokens_np).to(self.devices[0])

        # row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        # next_row_offsets = self._input_row_offsets[:row_offsets_size]
        seq_lengths = [tokens_np.shape[1]] * tokens_np.shape[0]
        next_row_offsets_np = np.cumsum([0] + seq_lengths, dtype=np.uint32)
        next_row_offsets = Tensor.from_numpy(next_row_offsets_np).to(self.devices[0])

        print(
            f"prepare_next_token_inputs tokens={tokens_np} input_row_offsets={next_row_offsets.to_numpy()}"
        )

        return GPT2Inputs(
            tokens=tokens,
            input_row_offsets=next_row_offsets,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    @override
    def load_kv_manager(
        self, session: InferenceSession, available_cache_memory: int | None
    ) -> PagedKVCacheManager:
        return load_kv_manager(
            params=GPT2Config.get_kv_params(
                self.huggingface_config,
                len(self.devices),
                self.kv_cache_config,
                self.encoding.dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, self.huggingface_config
            ),
            num_layers=self.get_num_layers(self.huggingface_config),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )

    @classmethod
    @override
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return GPT2Config.get_kv_params(
            huggingface_config, n_devices, kv_cache_config, cache_dtype
        )

    @classmethod
    @override
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return huggingface_config.n_layer

    @classmethod
    @override
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        return estimate_kv_cache_size(
            params=GPT2Config.get_kv_params(
                huggingface_config, len(devices), kv_cache_config, cache_dtype
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=cls.get_num_layers(huggingface_config),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )


def convert_state_dict(state_dict: dict[str, Any]):
    for key, tensor in list(state_dict.items()):
        if (
            key.endswith("c_attn.weight")
            or key.endswith("c_proj.weight")
            or key.endswith("c_fc.weight")
        ):
            state_dict[key] = np.ascontiguousarray(tensor.transpose())
        if key.endswith(".attn.bias"):
            del state_dict[key]
    state_dict["lm_head.weight"] = state_dict["wte.weight"]
    return state_dict


if __name__ == "__main__":
    import numpy as np
    from max.engine.api import InferenceSession
    from max.driver import Accelerator, Tensor, CPU
    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, ops

    np.random.seed(0)

    device_ref = DeviceRef.GPU()
    device = Accelerator()

    with safe_open(
        "/home/thomas/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors",
        framework="numpy",
    ) as f:
        weights = {k: f.get_tensor(k) for k in f.keys()}
        state_dict = convert_state_dict(weights)

    gpt2 = GPT2(
        n_ctx=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        layer_norm_epsilon=1e-5,
        device=device_ref,
    )
    gpt2.load_state_dict(state_dict)
    input_type = TensorType(dtype=DType.int64, shape=(1, 1000), device=DeviceRef.CPU())
    with Graph("gpt_2", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        y = gpt2(x)
        graph.output(y)

    session = InferenceSession(devices=[CPU(), device])
    model = session.load(graph, weights_registry=state_dict)
    for tensor in model.input_metadata:
        print(f"name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}")
    for tensor in model.output_metadata:
        print(f"out name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}")

    x = np.random.randint(0, 50257, size=(1, 1000))
    x = Tensor.from_numpy(x)
    print(x.dtype)
    output = model.execute(x)[0]
    assert isinstance(output, Tensor)
    result = output.to_numpy()
    print(result)
