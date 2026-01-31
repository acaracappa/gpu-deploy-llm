"""LLM model specifications and VRAM requirements.

Only open-weight models are supported (no HuggingFace token required).
Gated models (Llama-2, Llama-3, etc.) are not supported to avoid
HuggingFace token complexity.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List


class Quantization(str, Enum):
    """Supported quantization methods."""

    NONE = "none"  # FP16
    AWQ = "awq"
    GPTQ = "gptq"


@dataclass
class ModelSpec:
    """Specification for an LLM model."""

    model_id: str
    name: str
    vram_fp16_gb: float  # VRAM required for FP16
    vram_4bit_gb: float  # VRAM required for 4-bit quantization
    min_gpus: int = 1
    supports_awq: bool = True
    supports_gptq: bool = True
    description: str = ""

    def get_vram(self, quantization: Quantization = Quantization.NONE) -> float:
        """Get VRAM requirement for specified quantization."""
        if quantization in (Quantization.AWQ, Quantization.GPTQ):
            return self.vram_4bit_gb
        return self.vram_fp16_gb


# Registry of supported open-weight models
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": ModelSpec(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        name="TinyLlama 1.1B",
        vram_fp16_gb=4,
        vram_4bit_gb=2,
        min_gpus=1,
        description="Smallest model, great for testing",
    ),
    "microsoft/phi-2": ModelSpec(
        model_id="microsoft/phi-2",
        name="Phi-2",
        vram_fp16_gb=6,
        vram_4bit_gb=3,
        min_gpus=1,
        description="Microsoft's efficient 2.7B model",
    ),
    "mistralai/Mistral-7B-Instruct-v0.2": ModelSpec(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        name="Mistral 7B Instruct",
        vram_fp16_gb=16,
        vram_4bit_gb=8,
        min_gpus=1,
        description="High-quality 7B instruction-tuned model",
    ),
    "Qwen/Qwen2-7B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2-7B-Instruct",
        name="Qwen2 7B Instruct",
        vram_fp16_gb=16,
        vram_4bit_gb=8,
        min_gpus=1,
        description="Alibaba's Qwen2 7B model",
    ),
    "google/gemma-7b-it": ModelSpec(
        model_id="google/gemma-7b-it",
        name="Gemma 7B IT",
        vram_fp16_gb=16,
        vram_4bit_gb=8,
        min_gpus=1,
        description="Google's Gemma 7B instruction-tuned",
    ),
    # Smaller variants for quick testing
    "microsoft/phi-1_5": ModelSpec(
        model_id="microsoft/phi-1_5",
        name="Phi-1.5",
        vram_fp16_gb=4,
        vram_4bit_gb=2,
        min_gpus=1,
        description="Microsoft's compact 1.3B model",
    ),
    "Qwen/Qwen2-1.5B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2-1.5B-Instruct",
        name="Qwen2 1.5B Instruct",
        vram_fp16_gb=4,
        vram_4bit_gb=2,
        min_gpus=1,
        description="Alibaba's compact Qwen2 1.5B model",
    ),
    "google/gemma-2b-it": ModelSpec(
        model_id="google/gemma-2b-it",
        name="Gemma 2B IT",
        vram_fp16_gb=6,
        vram_4bit_gb=3,
        min_gpus=1,
        description="Google's compact Gemma 2B instruction-tuned",
    ),
}

# Aliases for common model names
MODEL_ALIASES: Dict[str, str] = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi-2": "microsoft/phi-2",
    "phi-1.5": "microsoft/phi-1_5",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
    "qwen": "Qwen/Qwen2-7B-Instruct",
    "gemma-7b": "google/gemma-7b-it",
    "gemma-2b": "google/gemma-2b-it",
    "gemma": "google/gemma-7b-it",
}


def get_model_spec(model_id: str) -> Optional[ModelSpec]:
    """Get model specification by ID or alias.

    Args:
        model_id: Full model ID or alias

    Returns:
        ModelSpec if found, None otherwise
    """
    # Check if it's an alias
    if model_id.lower() in MODEL_ALIASES:
        model_id = MODEL_ALIASES[model_id.lower()]

    return MODEL_REGISTRY.get(model_id)


def list_models() -> List[ModelSpec]:
    """Get list of all supported models.

    Returns:
        List of ModelSpec objects
    """
    return list(MODEL_REGISTRY.values())


def resolve_model_id(model_id: str) -> str:
    """Resolve model alias to full model ID.

    Args:
        model_id: Full model ID or alias

    Returns:
        Full model ID
    """
    if model_id.lower() in MODEL_ALIASES:
        return MODEL_ALIASES[model_id.lower()]
    return model_id
