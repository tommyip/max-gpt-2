import numpy as np
from max.graph.weights import WeightData, Weights


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}
    for key, value in state_dict.items():
        if not key.endswith(".attn.bias"):
            new_state_dict[key] = value.data()

    for key, value in new_state_dict.items():
        if (
            key.endswith("c_attn.weight")
            or key.endswith("c_proj.weight")
            or key.endswith("c_fc.weight")
        ):
            np_array = np.from_dlpack(value.data)
            new_state_dict[key] = np.ascontiguousarray(np_array.transpose())

    return new_state_dict
