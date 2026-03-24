"""GPU bootstrap helpers.

The V2 platform is intentionally GPU-first. Every app and trainer must fail early
with a clear error message if CUDA is not available.
"""

from __future__ import annotations


def require_gpu() -> None:
    """Require a CUDA-capable GPU at process startup."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for GPU validation but could not be imported."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is mandatory for this platform, but no CUDA-capable GPU was detected."
        )
