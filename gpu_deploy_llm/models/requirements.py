"""GPU requirement calculator for LLM deployments."""

from dataclasses import dataclass
from typing import Optional

from .registry import ModelSpec, Quantization, get_model_spec, resolve_model_id


@dataclass
class GPURequirements:
    """GPU requirements for deploying a model."""

    model_id: str
    quantization: Quantization
    min_vram_gb: float
    min_gpu_count: int

    # Overhead for vLLM runtime (~2GB recommended)
    VLLM_OVERHEAD_GB: float = 2.0

    @property
    def total_vram_needed(self) -> float:
        """Total VRAM needed including overhead."""
        return self.min_vram_gb + self.VLLM_OVERHEAD_GB

    def meets_requirements(self, vram_gb: float, gpu_count: int = 1) -> bool:
        """Check if given GPU specs meet requirements."""
        return vram_gb >= self.total_vram_needed and gpu_count >= self.min_gpu_count


def calculate_requirements(
    model_id: str,
    quantization: Quantization = Quantization.NONE,
) -> GPURequirements:
    """Calculate GPU requirements for a model.

    Args:
        model_id: Model ID or alias
        quantization: Quantization method

    Returns:
        GPURequirements object

    Raises:
        ValueError: If model is not in registry
    """
    # Resolve aliases
    full_model_id = resolve_model_id(model_id)

    spec = get_model_spec(full_model_id)
    if not spec:
        raise ValueError(
            f"Model '{model_id}' not found in registry. "
            f"Use 'gpu-deploy-llm list-models' to see available models."
        )

    # Check quantization support
    if quantization == Quantization.AWQ and not spec.supports_awq:
        raise ValueError(f"Model '{model_id}' does not support AWQ quantization")
    if quantization == Quantization.GPTQ and not spec.supports_gptq:
        raise ValueError(f"Model '{model_id}' does not support GPTQ quantization")

    vram = spec.get_vram(quantization)

    return GPURequirements(
        model_id=full_model_id,
        quantization=quantization,
        min_vram_gb=vram,
        min_gpu_count=spec.min_gpus,
    )


def select_best_offer(
    offers: list,
    requirements: GPURequirements,
    max_price: Optional[float] = None,
):
    """Select the best (cheapest suitable) offer.

    Args:
        offers: List of GPU offers (GPUOffer objects or dicts)
        requirements: GPU requirements to meet
        max_price: Maximum acceptable hourly price

    Returns:
        Best offer or None if no suitable offer
    """
    suitable = []

    for offer in offers:
        # Get attributes - support both Pydantic models and dicts
        if hasattr(offer, "vram_gb"):
            vram = offer.vram_gb
            gpu_count = offer.gpu_count
            price = offer.price_per_hour
        else:
            vram = offer.get("vram_gb", 0)
            gpu_count = offer.get("gpu_count", 1)
            price = offer.get("price_per_hour", 0)

        # Check VRAM
        if vram < requirements.total_vram_needed:
            continue

        # Check GPU count
        if gpu_count < requirements.min_gpu_count:
            continue

        # Check price
        if max_price is not None and price > max_price:
            continue

        suitable.append(offer)

    if not suitable:
        return None

    # Sort by price and return cheapest
    def get_price(o):
        if hasattr(o, "price_per_hour"):
            return o.price_per_hour
        return o.get("price_per_hour", float("inf"))

    suitable.sort(key=get_price)
    return suitable[0]
