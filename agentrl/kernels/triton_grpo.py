"""Optional Triton kernels for the sampled-token GRPO objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:  # pragma: no cover - exercised on CUDA runners with Triton installed.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - default local/dev path without Triton.
    triton = None
    tl = None


@dataclass(slots=True)
class TritonGRPOResult:
    policy_loss_tensor: torch.Tensor
    mean_ratio_tensor: torch.Tensor
    clip_ratio_region_mean_tensor: torch.Tensor
    clip_ratio_low_mean_tensor: torch.Tensor
    clip_ratio_high_mean_tensor: torch.Tensor


def triton_clipped_grpo_objective(
    *,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    sampled_token_mask: torch.Tensor,
    epsilon: float,
    clip_range: float,
) -> TritonGRPOResult | None:
    """Return a Triton fused GRPO objective result when the runtime supports it.

    The kernel covers the beta=0 sampled-token clipped objective. Callers keep
    the PyTorch path as the reference and fallback for CPU, missing Triton,
    unsupported dtypes/devices, and KL-enabled runs.
    """

    if triton is None or tl is None:
        return None
    if not current_logprobs.is_cuda:
        return None
    if current_logprobs.ndim != 2 or old_logprobs.shape != current_logprobs.shape:
        return None
    if sampled_token_mask.shape != current_logprobs.shape:
        return None
    if advantages.ndim != 1 or advantages.shape[0] != current_logprobs.shape[0]:
        return None
    if current_logprobs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return None

    current = current_logprobs.contiguous()
    old = old_logprobs.to(device=current.device, dtype=current.dtype).contiguous()
    adv = advantages.to(device=current.device, dtype=current.dtype).contiguous()
    mask = sampled_token_mask.to(device=current.device).contiguous()

    flat_count = current.numel()
    if flat_count == 0:
        return None

    block_size = 256
    grid = (triton.cdiv(flat_count, block_size),)
    partial_policy = torch.empty(grid[0], device=current.device, dtype=torch.float32)
    partial_ratio = torch.empty_like(partial_policy)
    partial_count = torch.empty_like(partial_policy)
    partial_clip_region = torch.empty_like(partial_policy)
    partial_clip_low = torch.empty_like(partial_policy)
    partial_clip_high = torch.empty_like(partial_policy)
    grad_numer = torch.empty_like(current)

    _grpo_forward_kernel[grid](
        current,
        old,
        adv,
        mask,
        grad_numer,
        partial_policy,
        partial_ratio,
        partial_count,
        partial_clip_region,
        partial_clip_low,
        partial_clip_high,
        flat_count,
        current.shape[1],
        float(epsilon),
        float(clip_range),
        BLOCK_SIZE=block_size,
    )

    (
        policy_loss,
        mean_ratio,
        clip_region,
        clip_low,
        clip_high,
    ) = _TritonGRPOObjective.apply(
        partial_policy,
        partial_ratio,
        partial_count,
        partial_clip_region,
        partial_clip_low,
        partial_clip_high,
        grad_numer,
        current,
    )
    return TritonGRPOResult(
        policy_loss_tensor=policy_loss,
        mean_ratio_tensor=mean_ratio,
        clip_ratio_region_mean_tensor=clip_region,
        clip_ratio_low_mean_tensor=clip_low,
        clip_ratio_high_mean_tensor=clip_high,
    )


class _TritonGRPOObjective(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        partial_policy: torch.Tensor,
        partial_ratio: torch.Tensor,
        partial_count: torch.Tensor,
        partial_clip_region: torch.Tensor,
        partial_clip_low: torch.Tensor,
        partial_clip_high: torch.Tensor,
        grad_numer: torch.Tensor,
        current_logprobs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del current_logprobs
        token_count = partial_count.sum().clamp(min=1.0)
        ctx.save_for_backward(grad_numer, token_count)
        policy_loss = partial_policy.sum() / token_count
        mean_ratio = partial_ratio.sum() / token_count
        clip_region = partial_clip_region.sum() / token_count
        clip_low = partial_clip_low.sum() / token_count
        clip_high = partial_clip_high.sum() / token_count
        return policy_loss, mean_ratio, clip_region, clip_low, clip_high

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_policy_loss: torch.Tensor | None,
        grad_mean_ratio: torch.Tensor | None,
        grad_clip_region: torch.Tensor | None,
        grad_clip_low: torch.Tensor | None,
        grad_clip_high: torch.Tensor | None,
    ) -> tuple[None, None, None, None, None, None, None, torch.Tensor]:
        del grad_mean_ratio, grad_clip_region, grad_clip_low, grad_clip_high
        grad_numer, token_count = ctx.saved_tensors
        if grad_policy_loss is None:
            grad_current = torch.zeros_like(grad_numer)
        else:
            grad_current = grad_policy_loss.to(device=grad_numer.device, dtype=grad_numer.dtype) * (
                grad_numer / token_count.to(device=grad_numer.device, dtype=grad_numer.dtype)
            )
        return None, None, None, None, None, None, None, grad_current


if triton is not None and tl is not None:  # pragma: no cover - requires Triton.

    @triton.jit
    def _grpo_forward_kernel(
        current_ptr,
        old_ptr,
        advantages_ptr,
        mask_ptr,
        grad_numer_ptr,
        partial_policy_ptr,
        partial_ratio_ptr,
        partial_count_ptr,
        partial_clip_region_ptr,
        partial_clip_low_ptr,
        partial_clip_high_ptr,
        n_elements: tl.constexpr,
        tokens_per_row: tl.constexpr,
        epsilon: tl.constexpr,
        clip_range: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements

        current = tl.load(current_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        old = tl.load(old_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        token_mask = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
        row_ids = offsets // tokens_per_row
        adv = tl.load(advantages_ptr + row_ids, mask=valid, other=0.0).to(tl.float32)
        active = valid & token_mask

        delta = current - old
        delta_clamped = tl.maximum(tl.minimum(delta, clip_range), -clip_range)
        ratio = tl.exp(delta_clamped)
        low = 1.0 - epsilon
        high = 1.0 + epsilon
        clipped_ratio = tl.maximum(tl.minimum(ratio, high), low)

        unclipped = ratio * adv
        clipped = clipped_ratio * adv
        use_unclipped = unclipped <= clipped
        selected = tl.minimum(unclipped, clipped)
        policy_terms = -selected

        clip_low = ratio < low
        clip_high = ratio > high
        clip_region = clip_low | clip_high
        unclamped_delta = (delta >= -clip_range) & (delta <= clip_range)
        grad_numer = tl.where(active & use_unclipped & unclamped_delta, -(adv * ratio), 0.0)

        tl.store(grad_numer_ptr + offsets, grad_numer, mask=valid)
        zeros = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        tl.store(partial_policy_ptr + program_id, tl.sum(tl.where(active, policy_terms, zeros), axis=0))
        tl.store(partial_ratio_ptr + program_id, tl.sum(tl.where(active, ratio, zeros), axis=0))
        tl.store(partial_count_ptr + program_id, tl.sum(tl.where(active, 1.0, 0.0), axis=0))
        tl.store(partial_clip_region_ptr + program_id, tl.sum(tl.where(active & clip_region, 1.0, 0.0), axis=0))
        tl.store(partial_clip_low_ptr + program_id, tl.sum(tl.where(active & clip_low, 1.0, 0.0), axis=0))
        tl.store(partial_clip_high_ptr + program_id, tl.sum(tl.where(active & clip_high, 1.0, 0.0), axis=0))

else:
    _grpo_forward_kernel = None
