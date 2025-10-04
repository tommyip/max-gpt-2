from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.lib import (
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from .model import GPT2Model
from .weight_adapters import convert_safetensor_state_dict

gpt2_arch = SupportedArchitecture(
    name="GPT2LMHeadModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["openai-community/gpt2"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=SupportedEncoding.float32,
    supported_encodings={SupportedEncoding.float32: [KVCacheStrategy.PAGED]},
    pipeline_model=GPT2Model,
    tokenizer=TextTokenizer,
    weight_adapters={WeightsFormat.safetensors: convert_safetensor_state_dict},
)
