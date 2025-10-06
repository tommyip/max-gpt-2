from dataclasses import dataclass
from typing import Sequence, cast, override

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine.api import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParams,
    PagedCacheValues,
    PagedKVCacheManager,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    KVCacheMixin,
    MAXModelConfig,
    MAXModelConfigBase,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    upper_bounded_default,
)
from transformers import AutoConfig

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
    """Tensor containing the input token IDs."""
    pos: Tensor
    """Tensor containing the token positions for indexing the position embeddings."""
    input_row_offsets: Tensor
    """Tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence."""

    def __init__(
        self,
        tokens: Tensor,
        pos: Tensor,
        input_row_offsets: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
    ):
        self.tokens = tokens
        self.pos = pos
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

    @override
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, GPT2Inputs)

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.pos,
            model_inputs.input_row_offsets,
            *curr_kv_cache_inputs,
        )

        logits = model_outputs[0]
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

        tokens_np = np.concatenate([ctx.next_tokens for ctx in context_batch])
        tokens = Tensor.from_numpy(tokens_np).to(self.devices[0])

        self.batch_seq_len = np.array([ctx.current_length for ctx in context_batch])
        batch_token_offsets = np.concatenate(
            [np.arange(ctx.active_length, dtype=np.int64) for ctx in context_batch]
        )
        batch_start_offsets = np.repeat(
            [ctx.start_idx for ctx in context_batch],
            [ctx.active_length for ctx in context_batch],
        )
        pos_np = batch_start_offsets + batch_token_offsets
        pos = Tensor.from_numpy(pos_np).to(self.devices[0])

        input_row_offsets_np = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch], dtype=np.uint32
        )
        input_row_offsets = Tensor.from_numpy(input_row_offsets_np).to(self.devices[0])

        return GPT2Inputs(
            tokens=tokens,
            pos=pos,
            input_row_offsets=input_row_offsets,
            kv_cache_inputs=kv_cache_inputs,
        )

    @override
    def prepare_next_token_inputs(
        self, next_tokens: Tensor, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        assert isinstance(prev_model_inputs, GPT2Inputs)

        next_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:next_offsets_size]

        self.batch_seq_len += 1
        pos = Tensor.from_numpy(self.batch_seq_len).to(self.devices[0])

        return GPT2Inputs(
            tokens=next_tokens,
            pos=pos,
            input_row_offsets=next_row_offsets,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    def graph_inputs(self) -> Sequence[TensorType]:
        device_ref = DeviceRef.from_device(self.devices[0])

        kv_inputs = self.kv_manager.input_symbols()
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        pos_type = TensorType(DType.int64, shape=["total_seq_len"], device=device_ref)
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return (tokens_type, pos_type, input_row_offsets_type, *kv_inputs[0])

    def load_model(self, session: InferenceSession) -> Model:
        max_batch_size = self.pipeline_config.max_batch_size
        assert max_batch_size
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        assert self.adapter
        state_dict = self.adapter(
            dict(self.weights.items()),
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
        )
        self.config = config = GPT2Config.generate(self.huggingface_config)
        device_ref = DeviceRef.from_device(self.devices[0])
        kv_params = GPT2Config.get_kv_params(
            self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.dtype,
        )

        graph_inputs = self.graph_inputs()

        model_spec = GPT2(
            n_ctx=config.n_ctx,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            layer_norm_epsilon=config.layer_norm_epsilon,
            kv_params=kv_params,
            dtype=DType.float32,
            device=device_ref,
        )
        model_spec.load_state_dict(state_dict, weight_alignment=1)

        with Graph("gpt2", input_types=graph_inputs) as graph:
            tokens, pos, input_row_offsets, *kv_cache_inputs = graph.inputs
            kv_collection = PagedCacheValues(
                kv_blocks=kv_cache_inputs[0].buffer,
                cache_lengths=kv_cache_inputs[1].tensor,
                lookup_table=kv_cache_inputs[2].tensor,
                max_lengths=kv_cache_inputs[3].tensor,
            )
            output = model_spec(
                tokens.tensor, pos.tensor, kv_collection, input_row_offsets.tensor
            )
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
