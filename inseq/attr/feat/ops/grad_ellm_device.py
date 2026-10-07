"""Device placement at the extension boundary, without patching Inseq."""

from contextlib import contextmanager

import torch


def model_input_device(model) -> torch.device:
    """Return the embedding device; reject unsupported sharding/offloading."""
    device = model.get_input_embeddings().weight.device
    devices = {parameter.device for parameter in model.parameters()}
    if device.type == "meta" or any(value.type == "meta" for value in devices):
        raise NotImplementedError("Meta/offloaded weights are unsupported; load the complete model on one device.")
    device_map = getattr(model, "hf_device_map", None) or {}
    offloaded = any(getattr(getattr(module, "_hf_hook", None), "offload", False) for module in model.modules())
    if len(devices) > 1 or offloaded or any(value == "disk" for value in device_map.values()):
        raise NotImplementedError("Multi-device/offloaded models are unsupported; load the model on one CPU or GPU.")
    return device


@contextmanager
def attribution_device(device):
    """Keep loading AND attribution in this scope when using Inseq on CUDA.

    Inseq 0.6/0.7 validates backend names, so this context yields 'cuda' while
    preserving the chosen CUDA index for every allocation until scope exit.
    """
    selected = torch.device(device)
    if selected.type == "cpu":
        yield "cpu"
    elif selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment.")
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA index {index} is not visible; device_count={torch.cuda.device_count()}.")
        with torch.cuda.device(index):
            yield "cuda"
    else:
        raise NotImplementedError("This extension validates CPU and CUDA only.")
