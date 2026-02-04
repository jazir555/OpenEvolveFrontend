"""
RESE Deep Exploration Engine Adapter

This adapter wraps the DEE library and provides:
- RESTful API interface
- Request validation
- Response transformation to canonical format
- Error handling with DLQ support

Following CLAUDE.md principles:
- Law of Configuration Explicitness: Env vars required
- Law of Idempotency: UPSERT logic
- Circuit Breaker: Pattern recognition failures
- Structured Logging: JSON with correlation_id
- Timeout: All operations bounded
"""

import os
import sys
import json
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import asdict

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "schemas"))

try:
    from rese_dee import (
        DeepExplorationEngine,
        HypothesisGenerator,
        PatternRecognizer,
        MCTSExplainer,
        DEELogger,
        CircuitBreaker,
        retry_with_backoff,
    )
    from rese_schemas import (
        ExplorationConfig,
        MCTSSearchResult,
        Hypothesis,
        HypothesisStatus,
    )
except ImportError:
    # Fallback imports
    from glue.lib.rese_dee import (
        DeepExplorationEngine,
        HypothesisGenerator,
        PatternRecognizer,
        MCTSExplainer,
        DEELogger,
        CircuitBreaker,
        retry_with_backoff,
    )
    from glue.schemas.rese_schemas import (
        ExplorationConfig,
        MCTSSearchResult,
        Hypothesis,
        HypothesisStatus,
    )


# ============================================================================
# DEAD LETTER QUEUE (DLQ)
# ============================================================================

class DeadLetterQueue:
    """
    Dead Letter Queue for failed exploration requests.

    Stores failed requests for later analysis and retry.
    """

    def __init__(self, logger: Optional[DEELogger] = None):
        self.logger = logger or DEELogger()
        self.failed_requests: List[Dict[str, Any]] = []
        self.max_size = int(os.getenv("DLQ_MAX_SIZE", "1000"))

    def add(self, request: Dict[str, Any], error: str, error_type: str):
        """
        Add failed request to DLQ.

        Args:
            request: The failed request
            error: Error message
            error_type: Type of error (transient, logic, system)
        """
        if len(self.failed_requests) >= self.max_size:
            self.logger.warning("DLQ full, dropping oldest request")
            self.failed_requests.pop(0)

        dlq_entry = {
            "request": request,
            "error": error,
            "error_type": error_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dlq_id": str(uuid.uuid4())
        }
        self.failed_requests.append(dlq_entry)

        self.logger.error(
            "Request added to DLQ",
            dlq_id=dlq_entry["dlq_id"],
            error_type=error_type,
            error=error
        )

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all failed requests."""
        return self.failed_requests.copy()

    def clear(self):
        """Clear the DLQ."""
        self.failed_requests.clear()
        self.logger.info("DLQ cleared")

    def size(self) -> int:
        """Get DLQ size."""
        return len(self.failed_requests)


# ============================================================================
# DEE ADAPTER
# ============================================================================

class DEEAdapter:
    """
    Adapter for RESE Deep Exploration Engine.

    Provides:
    - Request validation
    - Exploration execution
    - Response transformation
    - Error handling with DLQ
    """

    def __init__(self):
        # Validate configuration at startup (Law of Configuration Explicitness)
        self.config = self._validate_config()
        self.logger = DEELogger()
        self.dlq = DeadLetterQueue(self.logger)
        self.circuit_breaker = CircuitBreaker(logger=self.logger)

        # Initialize DEE engine
        self.engine = DeepExplorationEngine(self.config, self.logger)

        self.logger.info(
            "DEE Adapter initialized",
            config=self.config.to_dict()
        )

    def _validate_config(self) -> ExplorationConfig:
        """
        Validate configuration from environment.

        Crashes immediately if required vars are missing (Law of Configuration Explicitness).
        """
        try:
            config = ExplorationConfig.from_env()
            return config
        except Exception as e:
            print(f"FATAL: Configuration validation failed: {e}")
            print("Required environment variables:")
            print("  - EXPLORATION_DEPTH")
            print("  - MCTS_ITERATIONS")
            print("  - MCTS_EXPLORATION_CONSTANT")
            print("  - CONVERGENCE_THRESHOLD")
            print("  - EXPLORATION_TIMEOUT_MS")
            print("  - MAX_HYPOTHESES")
            print("  - PATTERN_RECOGNITION_THRESHOLD")
            sys.exit(1)

    def explore(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform deep exploration.

        Args:
            request: Exploration request with keys:
                - problem_statement (str, required)
                - domain (str, required)
                - context (dict, optional)
                - correlation_id (str, optional)

        Returns:
            Exploration result in canonical format

        Raises:
            ValueError: If request validation fails
        """
        correlation_id = request.get("correlation_id") or str(uuid.uuid4())
        self.logger = DEELogger(correlation_id)

        # Validate request
        validation_errors = self._validate_request(request)
        if validation_errors:
            error_msg = f"Request validation failed: {validation_errors}"
            self.logger.error(error_msg, errors=validation_errors)
            self.dlq.add(request, error_msg, "logic")
            raise ValueError(error_msg)

        problem_statement = request["problem_statement"]
        domain = request["domain"]
        context = request.get("context")

        self.logger.info(
            "Exploration request received",
            domain=domain,
            problem_length=len(problem_statement)
        )

        try:
            # Execute exploration with circuit breaker
            result = self.circuit_breaker.call(
                self._execute_exploration,
                problem_statement,
                domain,
                context
            )

            # Transform to canonical format
            canonical_result = self._to_canonical_format(result)

            self.logger.info(
                "Exploration complete",
                search_id=canonical_result["search_id"],
                best_confidence=canonical_result.get("best_confidence", 0.0)
            )

            return canonical_result

        except Exception as e:
            error_msg = f"Exploration failed: {str(e)}"
            self.logger.error(error_msg, error=str(e))

            # Add to DLQ based on error type
            error_type = self._classify_error(e)
            self.dlq.add(request, error_msg, error_type)

            # Re-raise for client handling
            raise

    def _execute_exploration(
        self,
        problem_statement: str,
        domain: str,
        context: Optional[Dict[str, Any]]
    ) -> MCTSSearchResult:
        """Execute exploration (wrapped by circuit breaker)."""
        return self.engine.explore(problem_statement, domain, context)

    def batch_explore(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform batch deep exploration.

        Args:
            request: Batch exploration request with keys:
                - problems (list of dict, required)
                    Each dict has: problem_statement, domain
                - context (dict, optional)
                - correlation_id (str, optional)

        Returns:
            Batch exploration result
        """
        correlation_id = request.get("correlation_id") or str(uuid.uuid4())
        self.logger = DEELogger(correlation_id)

        # Validate request
        if "problems" not in request or not isinstance(request["problems"], list):
            error_msg = "Request must contain 'problems' list"
            self.logger.error(error_msg)
            self.dlq.add(request, error_msg, "logic")
            raise ValueError(error_msg)

        problems_data = request["problems"]
        context = request.get("context")

        # Transform to list of tuples
        problems = []
        for p in problems_data:
            if "problem_statement" not in p or "domain" not in p:
                error_msg = "Each problem must have 'problem_statement' and 'domain'"
                self.logger.error(error_msg)
                self.dlq.add(request, error_msg, "logic")
                raise ValueError(error_msg)
            problems.append((p["problem_statement"], p["domain"]))

        self.logger.info(
            "Batch exploration request received",
            problem_count=len(problems)
        )

        try:
            # Execute batch exploration
            results = self.engine.batch_explore(problems, context)

            # Transform to canonical format
            canonical_results = [
                self._to_canonical_format(r)
                for r in results
            ]

            self.logger.info(
                "Batch exploration complete",
                total_problems=len(problems),
                successful_results=len(canonical_results)
            )

            return {
                "correlation_id": correlation_id,
                "total_problems": len(problems),
                "successful_results": len(canonical_results),
                "results": canonical_results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            error_msg = f"Batch exploration failed: {str(e)}"
            self.logger.error(error_msg, error=str(e))
            self.dlq.add(request, error_msg, "system")
            raise

    def _validate_request(self, request: Dict[str, Any]) -> List[str]:
        """Validate exploration request."""
        errors = []

        if "problem_statement" not in request:
            errors.append("Missing required field: problem_statement")
        elif not isinstance(request["problem_statement"], str):
            errors.append("problem_statement must be a string")

        if "domain" not in request:
            errors.append("Missing required field: domain")
        elif not isinstance(request["domain"], str):
            errors.append("domain must be a string")

        if "context" in request and not isinstance(request["context"], dict):
            errors.append("context must be a dictionary")

        return errors

    def _to_canonical_format(self, result: MCTSSearchResult) -> Dict[str, Any]:
        """
        Transform MCTS search result to canonical format.

        Canonical format ensures consistent API responses.
        """
        return {
            "search_id": result.search_id,
            "root_hypothesis": result.root_hypothesis.to_dict() if result.root_hypothesis else None,
            "best_hypothesis": result.best_hypothesis.to_dict() if result.best_hypothesis else None,
            "best_confidence": result.best_hypothesis.confidence if result.best_hypothesis else 0.0,
            "tree_statistics": {
                "iterations": result.iterations,
                "convergence_reached": result.convergence_reached,
                "convergence_iteration": result.convergence_iteration,
                "total_nodes": result.total_nodes,
                "max_depth": result.max_depth,
            },
            "execution_time_ms": result.execution_time_ms,
            "strategy": result.strategy.value if isinstance(result.strategy, ExplorationStrategy) else result.strategy,
            "patterns": result.metadata.get("patterns", []),
            "timestamp": result.created_at.isoformat() if isinstance(result.created_at, datetime) else result.created_at,
        }

    def _classify_error(self, error: Exception) -> str:
        """
        Classify error type for DLQ.

        Returns:
            "transient", "logic", or "system"
        """
        error_type = str(type(error).__name__)

        # Transient errors (network, timeout)
        if any(t in error_type.lower() for t in ["timeout", "connection", "network"]):
            return "transient"

        # Logic errors (validation, bad data)
        if any(t in error_type.lower() for t in ["value", "validation", "key"]):
            return "logic"

        # System errors (circuit breaker, etc.)
        return "system"

    def get_health(self) -> Dict[str, Any]:
        """Get adapter health status."""
        return {
            "status": "healthy" if self.circuit_breaker.state == "CLOSED" else "degraded",
            "circuit_breaker_state": self.circuit_breaker.state,
            "dlq_size": self.dlq.size(),
            "config": self.config.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_dlq_contents(self) -> List[Dict[str, Any]]:
        """Get contents of Dead Letter Queue."""
        return self.dlq.get_all()

    def clear_dlq(self):
        """Clear Dead Letter Queue."""
        self.dlq.clear()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface for DEE adapter."""
    import argparse

    parser = argparse.ArgumentParser(description="RESE Deep Exploration Engine Adapter")
    parser.add_argument("--problem", type=str, help="Problem statement to explore")
    parser.add_argument("--domain", type=str, help="Domain of the problem")
    parser.add_argument("--config", action="store_true", help="Show configuration and exit")
    parser.add_argument("--health", action="store_true", help="Show health status and exit")
    parser.add_argument("--dlq", action="store_true", help="Show Dead Letter Queue contents")

    args = parser.parse_args()

    # Initialize adapter
    adapter = DEEAdapter()

    if args.config:
        print("Configuration:")
        print(json.dumps(adapter.config.to_dict(), indent=2))
        return

    if args.health:
        print("Health Status:")
        print(json.dumps(adapter.get_health(), indent=2))
        return

    if args.dlq:
        print("Dead Letter Queue:")
        print(json.dumps(adapter.get_dlq_contents(), indent=2))
        return

    if args.problem and args.domain:
        print("Exploring...")
        result = adapter.explore({
            "problem_statement": args.problem,
            "domain": args.domain
        })
        print("\nResult:")
        print(json.dumps(result, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
