"""
Judge Service Adapter

Connects OpenEvolve to BubbleLab Evolution Judge API
Evaluates generated code using visual LLM judge

Judge API: http://localhost:3001/api/evolution-judge
"""

import httpx
import structlog
from typing import Dict, Any, List, Optional
from ..config import settings

logger = structlog.get_logger()


class JudgeAdapter:
    """Adapter for BubbleLab Evolution Judge API"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize Judge adapter

        Args:
            base_url: Judge API base URL (default from settings)
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or settings.JUDGE_API_URL).rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def health_check(self) -> bool:
        """
        Check if Judge service is healthy

        Returns:
            True if service is healthy
        """
        try:
            response = await self.client.get(f"{self.base_url}/health")
            is_healthy = response.status_code == 200
            logger.info(
                "judge_health_check",
                healthy=is_healthy,
                status_code=response.status_code,
            )
            return is_healthy
        except Exception as e:
            logger.error("judge_health_check_failed", error=str(e))
            return False

    async def evaluate(
        self,
        code: str,
        problem_statement: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate generated code using visual LLM judge

        Args:
            code: Generated code to evaluate
            problem_statement: Original problem statement
            weights: Optional weights for different criteria

        Returns:
            Evaluation results with scores and feedback
        """
        try:
            payload = {
                "input": {
                    "code": code,
                    "problem": problem_statement,
                },
                "weights": weights or {
                    "correctness": 0.4,
                    "efficiency": 0.3,
                    "style": 0.2,
                    "documentation": 0.1,
                },
            }

            logger.debug(
                "judge_evaluate_request",
                code_length=len(code),
                problem_length=len(problem_statement),
            )

            response = await self.client.post(
                f"{self.base_url}/judge",
                json=payload,
            )

            response.raise_for_status()
            result = response.json()

            logger.info(
                "judge_evaluate_success",
                score=result.get("overall_score"),
                criteria_count=len(result.get("criteria", [])),
            )

            return result

        except httpx.HTTPStatusError as e:
            logger.error(
                "judge_evaluate_http_error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error("judge_evaluate_failed", error=str(e))
            raise

    async def evaluate_batch(
        self,
        codes: List[str],
        problem_statement: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple code samples in batch

        Args:
            codes: List of generated code samples
            problem_statement: Original problem statement
            weights: Optional weights for different criteria

        Returns:
            List of evaluation results
        """
        try:
            payload = {
                "inputs": [
                    {
                        "code": code,
                        "problem": problem_statement,
                    }
                    for code in codes
                ],
                "weights": weights or {
                    "correctness": 0.4,
                    "efficiency": 0.3,
                    "style": 0.2,
                    "documentation": 0.1,
                },
            }

            logger.debug(
                "judge_evaluate_batch_request",
                code_count=len(codes),
                problem_length=len(problem_statement),
            )

            response = await self.client.post(
                f"{self.base_url}/judge-batch",
                json=payload,
            )

            response.raise_for_status()
            results = response.json()

            logger.info(
                "judge_evaluate_batch_success",
                result_count=len(results),
                avg_score=sum(r.get("overall_score", 0) for r in results) / len(results),
            )

            return results

        except httpx.HTTPStatusError as e:
            logger.error(
                "judge_evaluate_batch_http_error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error("judge_evaluate_batch_failed", error=str(e))
            raise


# Singleton instance
_judge_adapter: Optional[JudgeAdapter] = None


def get_judge_adapter() -> JudgeAdapter:
    """Get or create Judge adapter singleton"""
    global _judge_adapter
    if _judge_adapter is None:
        _judge_adapter = JudgeAdapter()
    return _judge_adapter
