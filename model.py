from dataclasses import dataclass
from typing import Any

from max.engine.api import InferenceSession, Model
from max.pipelines.lib import (
    PipelineModel,
    PipelineConfig,
    SupportedEncoding,
    KVCacheConfig,
    MAXModelConfigBase,
    MAXModelConfig,
)
from max.pipelines.core import TextContext
from max.driver import Device
from max.dtype import DType
from max.graph import Graph, DeviceRef
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


class GPT2Model(PipelineModel[TextContext]):
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

    def load_model(self, session: InferenceSession) -> Model:
        assert self.adapter
        state_dict = self.adapter(
            dict(weights.items()),
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
        )
        config = GPT2Config.generate(self.huggingface_config)
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
        model_spec.load_state_dict(state_dict)
        input_type = TensorType(DType.int64, shape=["total_seq_len"], device=device)
        with Graph("gpt2", input_types=(input_type,)) as graph:
            token_ids = graph.inputs[0]
            output = model_spec(token_ids)
            graph.output(output)

        model = session.load(graph, weights_registry=state_dict)

        return model


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
