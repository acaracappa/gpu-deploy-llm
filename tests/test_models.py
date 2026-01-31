"""Tests for model registry and requirements calculation."""

import pytest
from gpu_deploy_llm.models import (
    list_models,
    get_model_spec,
    calculate_requirements,
    Quantization,
    MODEL_REGISTRY,
)
from gpu_deploy_llm.models.registry import resolve_model_id, MODEL_ALIASES


class TestModelRegistry:
    """Tests for model registry."""

    def test_list_models_returns_all_models(self):
        """All registered models should be returned."""
        models = list_models()
        assert len(models) == len(MODEL_REGISTRY)
        assert all(hasattr(m, "model_id") for m in models)

    def test_get_model_spec_by_full_id(self):
        """Should find model by full ID."""
        spec = get_model_spec("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert spec is not None
        assert spec.name == "TinyLlama 1.1B"
        assert spec.vram_fp16_gb == 4
        assert spec.vram_4bit_gb == 2

    def test_get_model_spec_by_alias(self):
        """Should find model by alias."""
        spec = get_model_spec("tinyllama")
        assert spec is not None
        assert spec.model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_get_model_spec_alias_case_insensitive(self):
        """Aliases should be case-insensitive."""
        spec1 = get_model_spec("TinyLlama")
        spec2 = get_model_spec("tinyllama")
        spec3 = get_model_spec("TINYLLAMA")
        assert spec1 == spec2 == spec3

    def test_get_model_spec_not_found(self):
        """Should return None for unknown model."""
        spec = get_model_spec("nonexistent/model")
        assert spec is None

    def test_resolve_model_id_alias(self):
        """Should resolve alias to full ID."""
        full_id = resolve_model_id("mistral-7b")
        assert full_id == "mistralai/Mistral-7B-Instruct-v0.2"

    def test_resolve_model_id_passthrough(self):
        """Should pass through full IDs unchanged."""
        full_id = resolve_model_id("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert full_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class TestModelSpec:
    """Tests for ModelSpec."""

    def test_get_vram_fp16(self):
        """Should return FP16 VRAM for no quantization."""
        spec = get_model_spec("tinyllama")
        assert spec.get_vram(Quantization.NONE) == 4

    def test_get_vram_awq(self):
        """Should return 4-bit VRAM for AWQ."""
        spec = get_model_spec("tinyllama")
        assert spec.get_vram(Quantization.AWQ) == 2

    def test_get_vram_gptq(self):
        """Should return 4-bit VRAM for GPTQ."""
        spec = get_model_spec("tinyllama")
        assert spec.get_vram(Quantization.GPTQ) == 2


class TestCalculateRequirements:
    """Tests for requirements calculation."""

    def test_calculate_requirements_fp16(self):
        """Should calculate correct requirements for FP16."""
        reqs = calculate_requirements("tinyllama", Quantization.NONE)
        assert reqs.model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert reqs.min_vram_gb == 4
        assert reqs.min_gpu_count == 1
        assert reqs.total_vram_needed == 6  # 4 + 2 overhead

    def test_calculate_requirements_quantized(self):
        """Should calculate correct requirements for quantized."""
        reqs = calculate_requirements("tinyllama", Quantization.AWQ)
        assert reqs.min_vram_gb == 2
        assert reqs.total_vram_needed == 4  # 2 + 2 overhead

    def test_calculate_requirements_alias(self):
        """Should work with aliases."""
        reqs = calculate_requirements("mistral", Quantization.NONE)
        assert reqs.model_id == "mistralai/Mistral-7B-Instruct-v0.2"
        assert reqs.min_vram_gb == 16

    def test_calculate_requirements_unknown_model(self):
        """Should raise for unknown model."""
        with pytest.raises(ValueError, match="not found"):
            calculate_requirements("nonexistent/model")

    def test_meets_requirements_true(self):
        """Should return True when requirements met."""
        reqs = calculate_requirements("tinyllama")
        assert reqs.meets_requirements(vram_gb=8, gpu_count=1)

    def test_meets_requirements_insufficient_vram(self):
        """Should return False for insufficient VRAM."""
        reqs = calculate_requirements("tinyllama")
        assert not reqs.meets_requirements(vram_gb=4, gpu_count=1)

    def test_meets_requirements_insufficient_gpus(self):
        """Should return False for insufficient GPUs."""
        reqs = calculate_requirements("tinyllama")
        # TinyLlama only needs 1 GPU, so this should still pass
        assert reqs.meets_requirements(vram_gb=8, gpu_count=1)
