import os
import torch
import functools
import gc
from loguru import logger
from collections import defaultdict
import torch
import torch.nn.functional as F

DEFAULT_EVAL_CUDA_DEVICE = "0"


def ensure_single_gpu_evaluation(
    preferred_gpu: str = DEFAULT_EVAL_CUDA_DEVICE, env: dict | None = None
) -> str:
    """Ensure evaluation runs on exactly one visible GPU.

    If `CUDA_VISIBLE_DEVICES` is unset, default to the preferred idle GPU.
    If multiple GPUs or multi-process Accelerate variables are configured,
    raise an error instead of silently running multi-GPU evaluation.
    """
    env = os.environ if env is None else env

    visible_devices = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = preferred_gpu
        visible_devices = preferred_gpu
        logger.info(
            f"CUDA_VISIBLE_DEVICES was unset; defaulting evaluation to GPU {preferred_gpu}."
        )

    normalized_devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
    if len(normalized_devices) != 1:
        raise RuntimeError(
            "Evaluation must run on exactly one GPU. "
            f"Set CUDA_VISIBLE_DEVICES to a single device (preferred {preferred_gpu}), "
            f"got {visible_devices!r}."
        )

    world_size = int(env.get("WORLD_SIZE", "1"))
    local_world_size = int(env.get("LOCAL_WORLD_SIZE", "1"))
    if world_size != 1 or local_world_size != 1:
        raise RuntimeError(
            "Multi-process evaluation is disabled. "
            f"Expected WORLD_SIZE=1 and LOCAL_WORLD_SIZE=1, got WORLD_SIZE={world_size}, "
            f"LOCAL_WORLD_SIZE={local_world_size}."
        )

    return normalized_devices[0]


def ensure_single_process_accelerator(accelerator, context: str = "evaluation") -> None:
    """Reject multi-process Accelerate setups for single-GPU evaluation."""
    if accelerator.num_processes != 1:
        raise RuntimeError(
            f"{context} must use exactly one process / one GPU, "
            f"but Accelerator reported num_processes={accelerator.num_processes}."
        )


def create_mask_after_token(
    input_ids: torch.Tensor, special_token_id: int, offset: int = 2
) -> torch.Tensor:
    """Create a boolean mask for positions at/after the first occurrence of special_token_id + offset.

    Args:
        input_ids: Token ids of shape (batch_size, seq_len).
        special_token_id: Token id to locate (e.g., answer start token).
        offset: Number of positions to skip after the special token (default 2).

    Returns:
        Boolean tensor of shape (batch_size, seq_len), True where positions are part of the answer.
    """
    bs, seq_len = input_ids.shape
    device = input_ids.device
    token_indices = torch.argmax((input_ids == special_token_id).int(), dim=1)
    token_found = torch.any(input_ids == special_token_id, dim=1)
    effective_indices = torch.where(token_found, token_indices, seq_len)
    start_masking_indices = effective_indices + offset
    col_indices = torch.arange(seq_len, device=device)
    mask = col_indices >= start_masking_indices.unsqueeze(1)
    return mask


def create_mask_after_last_token(
    input_ids: torch.Tensor, special_token_id: int, offset: int = 2
) -> torch.Tensor:
    """Create a boolean mask for positions at/after the last occurrence of special_token_id + offset.

    Args:
        input_ids: Token ids of shape (batch_size, seq_len).
        special_token_id: Token id to locate (e.g., answer start token).
        offset: Number of positions to skip after the last special token (default 2).

    Returns:
        Boolean tensor of shape (batch_size, seq_len), True where positions are part of the answer.
    """
    bs, seq_len = input_ids.shape
    device = input_ids.device
    is_special_token = (input_ids == special_token_id).int()
    flipped_matches = torch.flip(is_special_token, dims=[1])
    flipped_indices = torch.argmax(flipped_matches, dim=1)
    token_indices = (seq_len - 1) - flipped_indices
    token_found = torch.any(is_special_token, dim=1)
    effective_indices = torch.where(token_found, token_indices, seq_len)
    start_masking_indices = effective_indices + offset
    col_indices = torch.arange(seq_len, device=device)
    mask = col_indices >= start_masking_indices.unsqueeze(1)
    return mask


def create_mask_from_prompt_attention(
    full_attention_mask: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Create a mask covering completion tokens in left-padded prompt+completion batches."""
    full_lengths = full_attention_mask.sum(dim=1)
    prompt_lengths = prompt_attention_mask.sum(dim=1)
    answer_lengths = torch.clamp(full_lengths - prompt_lengths, min=0)
    seq_len = full_attention_mask.shape[1]
    col_indices = torch.arange(seq_len, device=full_attention_mask.device)
    start_indices = seq_len - answer_lengths
    mask = col_indices.unsqueeze(0) >= start_indices.unsqueeze(1)
    return mask & full_attention_mask.bool()


def to_cpu_detached(x):
    """Recursively move tensors to CPU and detach from the computation graph.

    Args:
        x: A tensor, list, tuple, dict, or other object (non-tensors returned as-is).

    Returns:
        Same structure as x with tensors moved to CPU and detached.
    """
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu_detached(t) for t in x)
    if isinstance(x, dict):
        return {k: to_cpu_detached(v) for k, v in x.items()}
    return x


def to_device(x, device):
    """Recursively move tensors to the specified device.

    Args:
        x: A tensor, list, tuple, dict, or other object (non-tensors returned as-is).
        device: Target device (e.g., 'cuda:0', 'cpu').

    Returns:
        Same structure as x with tensors on the target device.
    """
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def is_tensor(x):
    """Check if x is a PyTorch tensor.

    Args:
        x: Object to check.

    Returns:
        True if x is a torch.Tensor, False otherwise.
    """
    return isinstance(x, torch.Tensor)


class StopForwardException(Exception):
    pass


class DataSaverHook:
    """Forward hook to optionally store inputs, outputs, and stop forward pass."""

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.rec = defaultdict(list)

    def __call__(self, m, args, kwargs, output):
        """Process forward pass: optionally store input/output and raise StopForwardException.

        Args:
            m: The module being hooked.
            args: Positional arguments passed to forward.
            kwargs: Keyword arguments passed to forward.
            output: Output tensor from forward.
        """
        # assert args is None, "DataSaverHook only support kwargs input"
        # logger.info(f"DataSaverHook input kwargs: {kwargs.keys()}")
        # import ipdb; ipdb.set_trace()
        if self.store_input:
            self.rec["input"].append(
                {
                    **{k: to_cpu_detached(v) for k, v in kwargs.items()},
                }
            )
        if self.store_output:
            self.rec["output"].append(to_cpu_detached(output)[0])
        if self.stop_forward:
            raise StopForwardException

    def clear(self):
        """Clear recorded data and free GPU cache."""
        gc.collect()
        self.rec = defaultdict(list)


class GetLayerInpOut:
    """Utility to capture input and output of a specific layer via forward hook."""

    def __init__(self, model) -> None:
        self.model = model
        self.data_saver = DataSaverHook(
            store_input=True, store_output=True, stop_forward=True
        )

    def __call__(self, layer, from_model_start=False, **kwargs):
        """Run forward and return the input and output of the given layer.

        Args:
            layer: The layer module to hook.
            from_model_start: If True, run full model forward; else run only the layer.
            **kwargs: Forward pass arguments.

        Returns:
            Tuple of (list of input dicts, list of output tensors).
        """
        self.layer = layer
        handle = self.layer.register_forward_hook(self.data_saver, with_kwargs=True)
        with torch.no_grad():
            try:
                if from_model_start:
                    _ = self.model(**kwargs)
                else:
                    _ = self.layer(**kwargs)
            except StopForwardException:
                pass

        handle.remove()
        return self.data_saver.rec["input"], self.data_saver.rec["output"]

    def clear(self):
        """Clear DataSaverHook state and free cache."""
        self.data_saver.clear()
