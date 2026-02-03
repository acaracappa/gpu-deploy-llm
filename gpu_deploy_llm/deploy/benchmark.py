"""Benchmarking for vLLM deployments.

Measures:
- Tokens per second (generation speed)
- Time to first token (latency)
- Response quality via test prompts

Uses SSH-based curl to access vLLM on remote instances where
direct HTTP access isn't available (e.g., Vast.ai SSH proxy).
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from gpu_deploy_llm.ssh.connection import SSHConnection

logger = logging.getLogger(__name__)

# Test prompts for benchmarking and response comparison
TEST_PROMPTS = [
    {
        "id": "math_simple",
        "category": "reasoning",
        "prompt": "What is 15 + 27? Answer with just the number.",
        "expected_contains": ["42"],
        "max_tokens": 20,
    },
    {
        "id": "code_simple",
        "category": "coding",
        "prompt": "Write a Python function that returns the square of a number. Just the function, no explanation.",
        "expected_contains": ["def", "return", "**2", "* x"],
        "max_tokens": 100,
    },
    {
        "id": "knowledge",
        "category": "knowledge",
        "prompt": "What is the capital of France? Answer in one word.",
        "expected_contains": ["Paris"],
        "max_tokens": 20,
    },
    {
        "id": "creative",
        "category": "creative",
        "prompt": "Write a haiku about programming.",
        "expected_contains": None,  # No specific expected content
        "max_tokens": 100,
    },
    {
        "id": "instruction",
        "category": "instruction",
        "prompt": "List 3 primary colors, separated by commas.",
        "expected_contains": ["red", "blue", "yellow"],
        "max_tokens": 50,
    },
]

# Throughput test prompt (longer generation)
THROUGHPUT_PROMPT = {
    "id": "throughput",
    "prompt": "Explain the concept of machine learning in detail, covering supervised learning, unsupervised learning, and reinforcement learning.",
    "max_tokens": 500,
}


@dataclass
class PromptResult:
    """Result of a single prompt test."""

    prompt_id: str
    category: str
    prompt: str
    response: str
    tokens_generated: int
    time_seconds: float
    time_to_first_token_ms: float
    tokens_per_second: float
    expected_contains: Optional[List[str]]
    matches_expected: Optional[bool]  # None if no expected content
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""

    success: bool
    prompt_results: List[PromptResult] = field(default_factory=list)

    # Aggregate metrics
    total_prompts: int = 0
    successful_prompts: int = 0
    failed_prompts: int = 0

    # Performance metrics
    avg_tokens_per_second: float = 0.0
    avg_time_to_first_token_ms: float = 0.0
    total_tokens_generated: int = 0
    total_time_seconds: float = 0.0

    # Response quality metrics
    prompts_with_expected: int = 0
    prompts_matching_expected: int = 0
    match_rate: float = 0.0

    # Throughput test
    throughput_tokens_per_second: float = 0.0

    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "total_prompts": self.total_prompts,
            "successful_prompts": self.successful_prompts,
            "failed_prompts": self.failed_prompts,
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
            "avg_time_to_first_token_ms": round(self.avg_time_to_first_token_ms, 2),
            "total_tokens_generated": self.total_tokens_generated,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "prompts_with_expected": self.prompts_with_expected,
            "prompts_matching_expected": self.prompts_matching_expected,
            "match_rate": round(self.match_rate * 100, 1),
            "throughput_tokens_per_second": round(self.throughput_tokens_per_second, 2),
            "prompt_results": [
                {
                    "id": r.prompt_id,
                    "category": r.category,
                    "prompt": r.prompt[:50] + "..." if len(r.prompt) > 50 else r.prompt,
                    "response": r.response[:100] + "..." if len(r.response) > 100 else r.response,
                    "tokens": r.tokens_generated,
                    "tps": round(r.tokens_per_second, 2),
                    "ttft_ms": round(r.time_to_first_token_ms, 2),
                    "matches": r.matches_expected,
                    "error": r.error,
                }
                for r in self.prompt_results
            ],
            "error": self.error,
        }


class Benchmarker:
    """Runs benchmarks against a vLLM deployment via SSH.

    Uses SSH-based curl to access vLLM on remote instances where
    direct HTTP access isn't available through SSH proxy hosts.
    """

    def __init__(
        self,
        ssh: SSHConnection,
        api_key: str,
        model_id: Optional[str] = None,
        timeout: float = 120.0,
        port: int = 8000,
    ):
        """Initialize benchmarker with SSH connection.

        Args:
            ssh: SSH connection to the remote instance
            api_key: API key for vLLM authentication
            model_id: Model ID (fetched from API if not provided)
            timeout: Request timeout in seconds
            port: vLLM port on remote instance (default 8000)
        """
        self.ssh = ssh
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout
        self.port = port

    async def __aenter__(self):
        # Fetch model ID if not provided
        if not self.model_id:
            await self._fetch_model_id()
        return self

    async def __aexit__(self, *args):
        pass  # No cleanup needed for SSH-based approach

    async def _fetch_model_id(self) -> None:
        """Fetch model ID from the API via SSH."""
        try:
            result = await self.ssh.run(
                f"curl -s -H 'Authorization: Bearer {self.api_key}' "
                f"http://localhost:{self.port}/v1/models",
                timeout=int(self.timeout),
            )
            if result.success and result.stdout.strip():
                data = json.loads(result.stdout)
                models = data.get("data", [])
                if models:
                    self.model_id = models[0].get("id")
                    logger.info(f"Using model: {self.model_id}")
        except Exception as e:
            logger.warning(f"Could not fetch model ID: {e}")

    async def run_benchmark(
        self,
        include_throughput: bool = True,
        prompts: Optional[List[Dict]] = None,
    ) -> BenchmarkResult:
        """Run full benchmark suite.

        Args:
            include_throughput: Whether to run throughput test
            prompts: Custom prompts (uses TEST_PROMPTS if not provided)

        Returns:
            BenchmarkResult with all metrics
        """
        result = BenchmarkResult(success=False)
        prompts = prompts or TEST_PROMPTS

        try:
            # Run test prompts
            logger.info(f"Running {len(prompts)} test prompts...")

            for prompt_config in prompts:
                prompt_result = await self._run_prompt(prompt_config)
                result.prompt_results.append(prompt_result)

                if prompt_result.error:
                    result.failed_prompts += 1
                else:
                    result.successful_prompts += 1

            result.total_prompts = len(prompts)

            # Run throughput test
            if include_throughput:
                logger.info("Running throughput test...")
                throughput_result = await self._run_prompt(THROUGHPUT_PROMPT)
                if not throughput_result.error:
                    result.throughput_tokens_per_second = throughput_result.tokens_per_second

            # Calculate aggregate metrics
            successful_results = [r for r in result.prompt_results if not r.error]

            if successful_results:
                result.avg_tokens_per_second = sum(r.tokens_per_second for r in successful_results) / len(successful_results)
                result.avg_time_to_first_token_ms = sum(r.time_to_first_token_ms for r in successful_results) / len(successful_results)
                result.total_tokens_generated = sum(r.tokens_generated for r in successful_results)
                result.total_time_seconds = sum(r.time_seconds for r in successful_results)

            # Calculate response quality metrics
            results_with_expected = [r for r in result.prompt_results if r.expected_contains is not None]
            result.prompts_with_expected = len(results_with_expected)
            result.prompts_matching_expected = sum(1 for r in results_with_expected if r.matches_expected)

            if result.prompts_with_expected > 0:
                result.match_rate = result.prompts_matching_expected / result.prompts_with_expected

            result.success = result.failed_prompts == 0

        except Exception as e:
            result.error = str(e)
            logger.error(f"Benchmark failed: {e}")

        return result

    async def _run_prompt(self, prompt_config: Dict) -> PromptResult:
        """Run a single prompt and measure performance via SSH curl."""
        prompt_id = prompt_config.get("id", "unknown")
        category = prompt_config.get("category", "general")
        prompt = prompt_config["prompt"]
        max_tokens = prompt_config.get("max_tokens", 100)
        expected_contains = prompt_config.get("expected_contains")

        result = PromptResult(
            prompt_id=prompt_id,
            category=category,
            prompt=prompt,
            response="",
            tokens_generated=0,
            time_seconds=0.0,
            time_to_first_token_ms=0.0,
            tokens_per_second=0.0,
            expected_contains=expected_contains,
            matches_expected=None,
        )

        try:
            # Build request payload - escape for shell
            payload = json.dumps({
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            })
            # Escape single quotes in payload for shell
            payload_escaped = payload.replace("'", "'\\''")

            # Use curl with timing to measure TTFT (time_starttransfer)
            # time_starttransfer is time until first byte received
            start_time = time.perf_counter()

            curl_result = await self.ssh.run(
                f"curl -s -w '\\n__TIMING__:%{{time_starttransfer}}:%{{time_total}}' "
                f"-X POST -H 'Authorization: Bearer {self.api_key}' "
                f"-H 'Content-Type: application/json' "
                f"-d '{payload_escaped}' http://localhost:{self.port}/v1/chat/completions",
                timeout=int(self.timeout),
            )

            end_time = time.perf_counter()

            if not curl_result.success:
                result.error = f"curl failed: {curl_result.stderr}"
                return result

            # Parse response and timing
            output = curl_result.stdout
            timing_line = ""
            response_json = ""

            if "__TIMING__:" in output:
                parts = output.rsplit("__TIMING__:", 1)
                response_json = parts[0].strip()
                timing_line = parts[1].strip() if len(parts) > 1 else ""
            else:
                response_json = output

            # Parse timing
            ttft_seconds = 0.0
            total_seconds = end_time - start_time

            if timing_line:
                try:
                    timing_parts = timing_line.split(":")
                    if len(timing_parts) >= 2:
                        ttft_seconds = float(timing_parts[0])
                        total_seconds = float(timing_parts[1])
                except (ValueError, IndexError):
                    pass

            # Parse response JSON
            if response_json:
                try:
                    data = json.loads(response_json)
                    choices = data.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        response_text = message.get("content", "")
                        result.response = response_text

                        # Estimate token count from usage or response length
                        usage = data.get("usage", {})
                        tokens = usage.get("completion_tokens", 0)
                        if tokens == 0:
                            # Rough estimate: ~4 chars per token
                            tokens = max(1, len(response_text) // 4)

                        result.tokens_generated = tokens
                        result.time_seconds = total_seconds
                        result.time_to_first_token_ms = ttft_seconds * 1000

                        if total_seconds > 0:
                            result.tokens_per_second = tokens / total_seconds

                        # Check expected content
                        if expected_contains is not None:
                            response_lower = response_text.lower()
                            result.matches_expected = any(
                                exp.lower() in response_lower for exp in expected_contains
                            )

                        logger.debug(
                            f"Prompt '{prompt_id}': {tokens} tokens in {total_seconds:.2f}s "
                            f"({result.tokens_per_second:.1f} tps, TTFT: {result.time_to_first_token_ms:.0f}ms)"
                        )
                except json.JSONDecodeError as e:
                    result.error = f"Invalid JSON: {str(e)[:50]}"
            else:
                result.error = "Empty response"

        except Exception as e:
            result.error = str(e)
            logger.error(f"Prompt '{prompt_id}' failed: {e}")

        return result


async def run_benchmark(
    ssh: SSHConnection,
    api_key: str,
    model_id: Optional[str] = None,
    port: int = 8000,
) -> BenchmarkResult:
    """Convenience function to run benchmark via SSH.

    Args:
        ssh: SSH connection to remote instance
        api_key: API key for vLLM
        model_id: Optional model ID
        port: vLLM port (default 8000)

    Returns:
        BenchmarkResult
    """
    async with Benchmarker(ssh, api_key, model_id, port=port) as benchmarker:
        return await benchmarker.run_benchmark()
