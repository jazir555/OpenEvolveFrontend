"""
Mutate Service Adapter

Connects OpenEvolve to BubbleLab Evolution Mutate API
Performs code mutations for evolutionary algorithms

Mutate API: http://localhost:3001/api/evolution-mutate
"""

import httpx
import structlog
from typing import Dict, Any, List, Optional
from ..config import settings

logger = structlog.get_logger()


class MutateAdapter:
    """Adapter for BubbleLab Evolution Mutate API"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize Mutate adapter

        Args:
            base_url: Mutate API base URL (default from settings)
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or settings.MUTATE_API_URL).rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def health_check(self) -> bool:
        """
        Check if Mutate service is healthy

        Returns:
            True if service is healthy
        """
        try:
            response = await self.client.get(f"{self.base_url}/health")
            is_healthy = response.status_code == 200
            logger.info(
                "mutate_health_check",
                healthy=is_healthy,
                status_code=response.status_code,
            )
            return is_healthy
        except Exception as e:
            logger.error("mutate_health_check_failed", error=str(e))
            return False

    async def mutate(
        self,
        code: str,
        mutation_type: str = "point",
        mutation_rate: float = 0.1,
        target_loc: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform mutation on code

        Args:
            code: Original code to mutate
            mutation_type: Type of mutation (point, crossover, etc.)
            mutation_rate: Probability of mutation per token
            target_loc: Optional target location for mutation

        Returns:
            Mutated code with metadata
        """
        try:
            payload = {
                "code": code,
                "mutation_type": mutation_type,
                "mutation_rate": mutation_rate,
                "target_loc": target_loc,
            }

            logger.debug(
                "mutate_request",
                code_length=len(code),
                mutation_type=mutation_type,
                mutation_rate=mutation_rate,
            )

            response = await self.client.post(
                f"{self.base_url}/mutate",
                json=payload,
            )

            response.raise_for_status()
            result = response.json()

            logger.info(
                "mutate_success",
                original_length=len(code),
                mutated_length=len(result.get("mutated_code", "")),
                mutations_count=result.get("mutations_count", 0),
            )

            return result

        except httpx.HTTPStatusError as e:
            logger.error(
                "mutate_http_error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error("mutate_failed", error=str(e))
            raise

    async def mutate_batch(
        self,
        codes: List[str],
        mutation_type: str = "point",
        mutation_rate: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Perform mutations on multiple code samples

        Args:
            codes: List of original code samples
            mutation_type: Type of mutation
            mutation_rate: Probability of mutation per token

        Returns:
            List of mutated code results
        """
        try:
            payload = {
                "codes": codes,
                "mutation_type": mutation_type,
                "mutation_rate": mutation_rate,
            }

            logger.debug(
                "mutate_batch_request",
                code_count=len(codes),
                mutation_type=mutation_type,
                mutation_rate=mutation_rate,
            )

            response = await self.client.post(
                f"{self.base_url}/mutate-batch",
                json=payload,
            )

            response.raise_for_status()
            results = response.json()

            logger.info(
                "mutate_batch_success",
                result_count=len(results),
                total_mutations=sum(r.get("mutations_count", 0) for r in results),
            )

            return results

        except httpx.HTTPStatusError as e:
            logger.error(
                "mutate_batch_http_error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error("mutate_batch_failed", error=str(e))
            raise

    async def crossover(
        self,
        code1: str,
        code2: str,
        num_points: int = 1,
    ) -> Dict[str, Any]:
        """
        Perform crossover between two code samples

        Args:
            code1: First parent code
            code2: Second parent code
            num_points: Number of crossover points

        Returns:
            Child code from crossover
        """
        try:
            payload = {
                "code1": code1,
                "code2": code2,
                "num_points": num_points,
            }

            logger.debug(
                "crossover_request",
                code1_length=len(code1),
                code2_length=len(code2),
                num_points=num_points,
            )

            response = await self.client.post(
                f"{self.base_url}/crossover",
                json=payload,
            )

            response.raise_for_status()
            result = response.json()

            logger.info(
                "crossover_success",
                child_length=len(result.get("child_code", "")),
                crossover_points=result.get("crossover_points", []),
            )

            return result

        except Exception as e:
            logger.error("crossover_failed", error=str(e))
            raise


# Singleton instance
_mutate_adapter: Optional[MutateAdapter] = None


def get_mutate_adapter() -> MutateAdapter:
    """Get or create Mutate adapter singleton"""
    global _mutate_adapter
    if _mutate_adapter is None:
        _mutate_adapter = MutateAdapter()
    return _mutate_adapter
