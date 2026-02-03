"""Benchmark result storage and analysis.

Stores benchmark results with session metadata for:
- Historical comparison
- Cost/performance analysis
- Model/GPU type recommendations
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Default storage location
DEFAULT_STORE_PATH = Path.home() / ".gpu-deploy-llm" / "benchmarks"


@dataclass
class BenchmarkRecord:
    """Complete benchmark record with session context."""

    # Identifiers
    id: str  # Unique benchmark ID
    session_id: str
    timestamp: str  # ISO format

    # Model info
    model_id: str
    quantization: str  # none, awq, gptq

    # Hardware info
    provider: str  # vastai, tensordock
    gpu_type: str  # RTX 4090, A100, etc.
    gpu_count: int
    gpu_vram_mb: int

    # Cost info
    price_per_hour: float

    # Performance metrics
    avg_tokens_per_second: float
    avg_time_to_first_token_ms: float
    throughput_tokens_per_second: float
    total_tokens_generated: int

    # Quality metrics
    prompts_tested: int
    prompts_passed: int
    match_rate: float

    # Per-prompt results
    prompt_results: List[Dict[str, Any]]

    # Computed metrics
    tokens_per_dollar: float = 0.0  # tokens/sec per $/hr
    cost_per_million_tokens: float = 0.0

    def __post_init__(self):
        # Calculate cost efficiency metrics
        if self.price_per_hour > 0 and self.avg_tokens_per_second > 0:
            self.tokens_per_dollar = self.avg_tokens_per_second / self.price_per_hour
            # Cost per 1M tokens (assuming sustained throughput)
            tokens_per_hour = self.avg_tokens_per_second * 3600
            if tokens_per_hour > 0:
                self.cost_per_million_tokens = (self.price_per_hour / tokens_per_hour) * 1_000_000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkRecord":
        return cls(**data)


class BenchmarkStore:
    """Persistent storage for benchmark results."""

    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or DEFAULT_STORE_PATH
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.store_path / "index.json"
        self._index: List[Dict[str, Any]] = []
        self._load_index()

    def _load_index(self):
        """Load the benchmark index."""
        if self.index_file.exists():
            try:
                with open(self.index_file) as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load benchmark index: {e}")
                self._index = []

    def _save_index(self):
        """Save the benchmark index."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save benchmark index: {e}")

    def save(self, record: BenchmarkRecord) -> str:
        """Save a benchmark record.

        Args:
            record: BenchmarkRecord to save

        Returns:
            Record ID
        """
        # Save full record
        record_file = self.store_path / f"{record.id}.json"
        with open(record_file, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

        # Update index with summary
        index_entry = {
            "id": record.id,
            "session_id": record.session_id,
            "timestamp": record.timestamp,
            "model_id": record.model_id,
            "provider": record.provider,
            "gpu_type": record.gpu_type,
            "price_per_hour": record.price_per_hour,
            "avg_tokens_per_second": record.avg_tokens_per_second,
            "tokens_per_dollar": record.tokens_per_dollar,
            "cost_per_million_tokens": record.cost_per_million_tokens,
            "match_rate": record.match_rate,
        }

        # Remove existing entry if re-running
        self._index = [e for e in self._index if e["id"] != record.id]
        self._index.insert(0, index_entry)  # Newest first
        self._save_index()

        logger.info(f"Saved benchmark {record.id} to {record_file}")
        return record.id

    def get(self, benchmark_id: str) -> Optional[BenchmarkRecord]:
        """Get a benchmark record by ID."""
        record_file = self.store_path / f"{benchmark_id}.json"
        if not record_file.exists():
            return None

        with open(record_file) as f:
            return BenchmarkRecord.from_dict(json.load(f))

    def list(
        self,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        gpu_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List benchmark summaries with optional filters.

        Args:
            model_id: Filter by model
            provider: Filter by provider
            gpu_type: Filter by GPU type
            limit: Max results

        Returns:
            List of benchmark summaries
        """
        results = self._index

        if model_id:
            results = [r for r in results if r["model_id"] == model_id]
        if provider:
            results = [r for r in results if r["provider"] == provider]
        if gpu_type:
            results = [r for r in results if r["gpu_type"] == gpu_type]

        return results[:limit]

    def get_best_config(
        self,
        model_id: str,
        optimize_for: str = "cost",  # "cost", "speed", "balanced"
        max_price: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get the best configuration for a model based on historical data.

        Args:
            model_id: Model to find best config for
            optimize_for: Optimization target
            max_price: Maximum price constraint

        Returns:
            Best configuration recommendation or None
        """
        results = [r for r in self._index if r["model_id"] == model_id]

        if max_price:
            results = [r for r in results if r["price_per_hour"] <= max_price]

        if not results:
            return None

        if optimize_for == "cost":
            # Best tokens per dollar
            best = max(results, key=lambda r: r.get("tokens_per_dollar", 0))
        elif optimize_for == "speed":
            # Fastest tokens/sec
            best = max(results, key=lambda r: r.get("avg_tokens_per_second", 0))
        else:  # balanced
            # Score = tokens_per_second * tokens_per_dollar (normalized)
            max_tps = max(r.get("avg_tokens_per_second", 1) for r in results)
            max_tpd = max(r.get("tokens_per_dollar", 1) for r in results)
            best = max(
                results,
                key=lambda r: (
                    (r.get("avg_tokens_per_second", 0) / max_tps) *
                    (r.get("tokens_per_dollar", 0) / max_tpd)
                )
            )

        return {
            "recommendation": {
                "provider": best["provider"],
                "gpu_type": best["gpu_type"],
                "price_per_hour": best["price_per_hour"],
            },
            "expected_performance": {
                "tokens_per_second": best["avg_tokens_per_second"],
                "tokens_per_dollar": best["tokens_per_dollar"],
                "cost_per_million_tokens": best["cost_per_million_tokens"],
            },
            "based_on_benchmark": best["id"],
            "optimize_for": optimize_for,
        }

    def get_comparison(
        self,
        model_id: str,
    ) -> Dict[str, Any]:
        """Get performance comparison across GPU types for a model.

        Args:
            model_id: Model to compare

        Returns:
            Comparison data grouped by GPU type
        """
        results = [r for r in self._index if r["model_id"] == model_id]

        if not results:
            return {"model_id": model_id, "comparisons": [], "sample_count": 0}

        # Group by GPU type
        by_gpu: Dict[str, List] = {}
        for r in results:
            gpu = r["gpu_type"]
            if gpu not in by_gpu:
                by_gpu[gpu] = []
            by_gpu[gpu].append(r)

        comparisons = []
        for gpu_type, records in by_gpu.items():
            avg_tps = sum(r["avg_tokens_per_second"] for r in records) / len(records)
            avg_price = sum(r["price_per_hour"] for r in records) / len(records)
            avg_tpd = sum(r.get("tokens_per_dollar", 0) for r in records) / len(records)

            comparisons.append({
                "gpu_type": gpu_type,
                "sample_count": len(records),
                "avg_tokens_per_second": round(avg_tps, 2),
                "avg_price_per_hour": round(avg_price, 4),
                "avg_tokens_per_dollar": round(avg_tpd, 2),
                "providers": list(set(r["provider"] for r in records)),
            })

        # Sort by tokens per dollar (best value first)
        comparisons.sort(key=lambda x: x["avg_tokens_per_dollar"], reverse=True)

        return {
            "model_id": model_id,
            "comparisons": comparisons,
            "sample_count": len(results),
        }


def create_benchmark_record(
    session_id: str,
    model_id: str,
    quantization: str,
    provider: str,
    gpu_type: str,
    gpu_count: int,
    gpu_vram_mb: int,
    price_per_hour: float,
    benchmark_result,  # BenchmarkResult from benchmark.py
) -> BenchmarkRecord:
    """Create a BenchmarkRecord from a BenchmarkResult and session info.

    Args:
        session_id: Session ID
        model_id: Model ID
        quantization: Quantization method
        provider: Cloud provider
        gpu_type: GPU type
        gpu_count: Number of GPUs
        gpu_vram_mb: GPU VRAM in MB
        price_per_hour: Hourly price
        benchmark_result: BenchmarkResult from running benchmark

    Returns:
        BenchmarkRecord ready to save
    """
    import uuid

    return BenchmarkRecord(
        id=f"bench-{uuid.uuid4().hex[:8]}",
        session_id=session_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_id=model_id,
        quantization=quantization,
        provider=provider,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        gpu_vram_mb=gpu_vram_mb,
        price_per_hour=price_per_hour,
        avg_tokens_per_second=benchmark_result.avg_tokens_per_second,
        avg_time_to_first_token_ms=benchmark_result.avg_time_to_first_token_ms,
        throughput_tokens_per_second=benchmark_result.throughput_tokens_per_second,
        total_tokens_generated=benchmark_result.total_tokens_generated,
        prompts_tested=benchmark_result.total_prompts,
        prompts_passed=benchmark_result.successful_prompts,
        match_rate=benchmark_result.match_rate,
        prompt_results=benchmark_result.to_dict().get("prompt_results", []),
    )
