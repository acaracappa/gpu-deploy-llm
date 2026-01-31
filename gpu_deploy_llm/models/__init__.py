"""LLM model registry and requirements calculator."""

from .registry import (
    ModelSpec,
    Quantization,
    MODEL_REGISTRY,
    get_model_spec,
    list_models,
)
from .requirements import (
    GPURequirements,
    calculate_requirements,
)

__all__ = [
    "ModelSpec",
    "Quantization",
    "MODEL_REGISTRY",
    "get_model_spec",
    "list_models",
    "GPURequirements",
    "calculate_requirements",
]
