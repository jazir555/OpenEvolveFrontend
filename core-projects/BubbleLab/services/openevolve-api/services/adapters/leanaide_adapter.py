"""
LeanAide Service Adapter

Connects OpenEvolve to BubbleLab LeanAide API
Provides Lean 4 theorem proving capabilities

LeanAide API: http://localhost:3001/api/leanaide
"""

import httpx
import structlog
from typing import Dict, Any, Optional
from ..config import settings

logger = structlog.get_logger()


class LeanAideAdapter:
    """Adapter for BubbleLab LeanAide API"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 120.0,  # Longer timeout for theorem proving
    ):
        """
        Initialize LeanAide adapter

        Args:
            base_url: LeanAide API base URL (default from settings)
            timeout: Request timeout in seconds (default 120s for proving)
        """
        self.base_url = (base_url or settings.LEANAIDE_API_URL).rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def health_check(self) -> bool:
        """
        Check if LeanAide service is healthy

        Returns:
            True if service is healthy
        """
        try:
            # Try to get models (lightweight health check)
            response = await self.client.get(f"{self.base_url}/models")
            is_healthy = response.status_code == 200
            logger.info(
                "leanaide_health_check",
                healthy=is_healthy,
                status_code=response.status_code,
            )
            return is_healthy
        except Exception as e:
            logger.error("leanaide_health_check_failed", error=str(e))
            return False

    async def generate_proof(
        self,
        proposition: str,
        tactic: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a Lean 4 proof for a proposition

        Args:
            proposition: Mathematical proposition to prove
            tactic: Optional tactic to use
            context: Optional additional context

        Returns:
            Proof generation result with Lean 4 code
        """
        try:
            payload = {
                "task": "prove_for_formalization",
                "proposition": proposition,
            }

            if tactic:
                payload["tactic"] = tactic
            if context:
                payload["context"] = context

            logger.debug(
                "leanaide_generate_proof_request",
                proposition_length=len(proposition),
                tactic=tactic,
            )

            response = await self.client.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()
            result = response.json()

            logger.info(
                "leanaide_generate_proof_success",
                has_proof=bool(result.get("proof")),
                proof_length=len(result.get("proof", "")),
            )

            return result

        except httpx.HTTPStatusError as e:
            logger.error(
                "leanaide_generate_proof_http_error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error("leanaide_generate_proof_failed", error=str(e))
            raise

    async def verify_proof(
        self,
        proof: str,
        proposition: str,
    ) -> Dict[str, Any]:
        """
        Verify a Lean 4 proof

        Args:
            proof: Lean 4 proof code
            proposition: Proposition being proved

        Returns:
            Verification result with success status and errors
        """
        try:
            payload = {
                "task": "elaborate",
                "proof": proof,
                "proposition": proposition,
            }

            logger.debug(
                "leanaide_verify_proof_request",
                proof_length=len(proof),
                proposition_length=len(proposition),
            )

            response = await self.client.post(
                f"{self.base_url}/verify",
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()
            result = response.json()

            logger.info(
                "leanaide_verify_proof_success",
                is_valid=result.get("is_valid", False),
                has_errors=bool(result.get("errors")),
            )

            return result

        except httpx.HTTPStatusError as e:
            logger.error(
                "leanaide_verify_proof_http_error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error("leanaide_verify_proof_failed", error=str(e))
            raise

    async def get_models(self) -> Dict[str, Any]:
        """
        Get available Lean 4 models

        Returns:
            List of available models
        """
        try:
            logger.debug("leanaide_get_models_request")

            response = await self.client.get(f"{self.base_url}/models")

            response.raise_for_status()
            result = response.json()

            logger.info(
                "leanaide_get_models_success",
                model_count=len(result.get("models", [])),
            )

            return result

        except Exception as e:
            logger.error("leanaide_get_models_failed", error=str(e))
            raise

    async def run_benchmark(
        self,
        benchmark_name: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run a Lean 4 proving benchmark

        Args:
            benchmark_name: Name of benchmark to run
            timeout: Optional timeout for benchmark

        Returns:
            Benchmark results
        """
        try:
            payload = {
                "benchmark": benchmark_name,
            }

            if timeout:
                payload["timeout"] = timeout

            logger.debug(
                "leanaide_run_benchmark_request",
                benchmark_name=benchmark_name,
            )

            response = await self.client.post(
                f"{self.base_url}/benchmark/run",
                json=payload,
                timeout=timeout or self.timeout,
            )

            response.raise_for_status()
            result = response.json()

            logger.info(
                "leanaide_run_benchmark_success",
                benchmark_name=benchmark_name,
                result_count=len(result.get("results", [])),
            )

            return result

        except Exception as e:
            logger.error("leanaide_run_benchmark_failed", error=str(e))
            raise

    async def get_benchmark_results(
        self,
        benchmark_name: str,
    ) -> Dict[str, Any]:
        """
        Get results from a previously run benchmark

        Args:
            benchmark_name: Name of benchmark

        Returns:
            Benchmark results
        """
        try:
            logger.debug(
                "leanaide_get_benchmark_results_request",
                benchmark_name=benchmark_name,
            )

            response = await self.client.get(
                f"{self.base_url}/benchmark/{benchmark_name}/results",
            )

            response.raise_for_status()
            result = response.json()

            logger.info(
                "leanaide_get_benchmark_results_success",
                benchmark_name=benchmark_name,
                result_count=len(result.get("results", [])),
            )

            return result

        except Exception as e:
            logger.error("leanaide_get_benchmark_results_failed", error=str(e))
            raise


# Singleton instance
_leanaide_adapter: Optional[LeanAideAdapter] = None


def get_leanaide_adapter() -> LeanAideAdapter:
    """Get or create LeanAide adapter singleton"""
    global _leanaide_adapter
    if _leanaide_adapter is None:
        _leanaide_adapter = LeanAideAdapter()
    return _leanaide_adapter
