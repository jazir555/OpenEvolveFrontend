"""
RESE Phase III Adapter

Adapter for Phase III MCTS Search Executor.
Provides RESTful API interface and integrates with DEE and LLTL.

Following CLAUDE.md principles:
- Law of Configuration Explicitness: Env vars required
- Law of Idempotency: UPSERT logic
- Circuit Breaker: Detect search failures
- Structured Logging: JSON with correlation_id
- Timeout: All operations timeout (default 30000ms)

Author: RESE Team
Created: 2026-02-04
Phase: III - Monte Carlo Refinement
"""

import os
import sys
import json
import uuid
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "schemas"))

try:
    from rese_schemas import (
        Hypothesis,
        MCTSSearchResult,
        HypothesisStatus,
        ExplorationStrategy,
    )
    from rese_dee import DEELogger, CircuitBreaker
except ImportError:
    # Fallback imports
    from glue.schemas.rese_schemas import (
        Hypothesis,
        MCTSSearchResult,
        HypothesisStatus,
        ExplorationStrategy,
    )
    from glue.lib.rese_dee import DEELogger, CircuitBreaker

try:
    from phase3_executor import (
        MCTSSearchExecutor,
        Phase3Config,
        ValidationMetrics,
    )
except ImportError:
    # Use importlib for dynamic import with hyphenated package name
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "phase3_executor",
        os.path.join(os.path.dirname(__file__), "phase3_executor.py")
    )
    phase3_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase3_module)
    MCTSSearchExecutor = phase3_module.MCTSSearchExecutor
    Phase3Config = phase3_module.Phase3Config
    ValidationMetrics = phase3_module.ValidationMetrics


# ============================================================================
# PHASE III ADAPTER
# ============================================================================

class Phase3Adapter:
    """
    Adapter for RESE Phase III MCTS Search.

    Provides simplified interface for MC-NEST search while maintaining
    all CLAUDE.md compliance requirements.
    """

    def __init__(self, config: Optional[Phase3Config] = None):
        """
        Initialize Phase III adapter.

        Args:
            config: Optional configuration dict (overrides env vars)

        Raises:
            RuntimeError: If configuration validation fails
        """
        # Load configuration from environment (Law of Configuration Explicitness)
        self.config = config or Phase3Config.from_env()
        self.logger = DEELogger(self.config.correlation_id)

        # Initialize executor
        self.executor = MCTSSearchExecutor(self.config, self.logger)

        self.logger.info(
            "Phase III Adapter initialized",
            config=self.config.__dict__
        )

    def search(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform MCTS search.

        Args:
            request: Search request with keys:
                - root_hypothesis (dict, required): Root hypothesis data
                - hypothesis_generator_config (dict, optional): Config for generation
                - reward_function_config (dict, optional): Config for reward function
                - correlation_id (str, optional): For tracing

        Returns:
            Search result in canonical format

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
            raise ValueError(error_msg)

        # Parse root hypothesis
        root_hypothesis_data = request["root_hypothesis"]
        root_hypothesis = Hypothesis.from_dict(root_hypothesis_data)

        self.logger.info(
            "Phase III search request received",
            hypothesis_id=root_hypothesis.hypothesis_id,
            domain=root_hypothesis.domain
        )

        try:
            # Create hypothesis generator (default implementation)
            def hypothesis_generator() -> List[Hypothesis]:
                """Generate child hypotheses (default implementation)."""
                # In production, this would use DEE's HypothesisGenerator
                num_children = request.get("num_children", 5)
                children = []

                for i in range(num_children):
                    child = Hypothesis(
                        statement=f"Child hypothesis {i} from {root_hypothesis.hypothesis_id}",
                        type=root_hypothesis.type,
                        domain=root_hypothesis.domain,
                        confidence=0.5,
                        source_hypotheses=[root_hypothesis.hypothesis_id],
                    )
                    children.append(child)

                return children

            # Create reward function (default implementation)
            def reward_function(hypothesis: Hypothesis) -> float:
                """Calculate reward for hypothesis (default implementation)."""
                # In production, this would use LLTL for constraint-based evaluation
                base_reward = hypothesis.confidence

                # Add noise for exploration
                import random
                noise = random.uniform(-0.1, 0.1)

                return max(0.0, min(1.0, base_reward + noise))

            # Execute search
            start_time = time.time()
            search_result, error = self.executor.execute_search(
                root_hypothesis=root_hypothesis,
                hypothesis_generator=hypothesis_generator,
                reward_function=reward_function,
            )

            if error:
                self.logger.error("Search failed", error=error)
                return {
                    "success": False,
                    "error": error,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # Transform to canonical format
            canonical_result = self._to_canonical_format(search_result)

            self.logger.info(
                "Search complete",
                search_id=search_result.search_id,
                best_confidence=canonical_result.get("best_confidence", 0.0),
                execution_time_ms=canonical_result.get("execution_time_ms", 0.0)
            )

            return canonical_result

        except Exception as e:
            error_msg = f"Search execution failed: {str(e)}"
            self.logger.error(error_msg, error=str(e))
            return {
                "success": False,
                "error": error_msg,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def validate_hypothesis(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a single hypothesis.

        Args:
            request: Validation request with keys:
                - hypothesis (dict, required): Hypothesis to validate
                - rewards (list, required): Reward values from simulations
                - correlation_id (str, optional): For tracing

        Returns:
            Validation result
        """
        correlation_id = request.get("correlation_id") or str(uuid.uuid4())
        self.logger = DEELogger(correlation_id)

        # Validate request
        if "hypothesis" not in request:
            error_msg = "Missing required field: hypothesis"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

        if "rewards" not in request:
            error_msg = "Missing required field: rewards"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

        try:
            # Parse hypothesis
            hypothesis = Hypothesis.from_dict(request["hypothesis"])
            rewards = request["rewards"]

            # Validate
            validation_metrics, error = self.executor.hypothesis_validator.validate(
                hypothesis,
                rewards
            )

            if error:
                self.logger.error("Validation failed", error=error)
                return {
                    "success": False,
                    "error": error,
                    "correlation_id": correlation_id,
                }

            self.logger.info(
                "Hypothesis validated",
                hypothesis_id=hypothesis.hypothesis_id,
                is_valid=validation_metrics.is_valid,
                confidence=validation_metrics.confidence
            )

            return {
                "success": True,
                "validation_result": validation_metrics.to_dict(),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            self.logger.error(error_msg, error=str(e))
            return {
                "success": False,
                "error": error_msg,
                "correlation_id": correlation_id,
            }

    def check_convergence(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if convergence has been reached.

        Args:
            request: Convergence check request with keys:
                - iteration (int, required): Current iteration
                - best_confidence (float, required): Best hypothesis confidence
                - best_reward (float, required): Best reward value
                - correlation_id (str, optional): For tracing

        Returns:
            Convergence check result
        """
        correlation_id = request.get("correlation_id") or str(uuid.uuid4())
        self.logger = DEELogger(correlation_id)

        # Validate request
        required_fields = ["iteration", "best_confidence", "best_reward"]
        missing_fields = [f for f in required_fields if f not in request]

        if missing_fields:
            error_msg = f"Missing required fields: {missing_fields}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

        try:
            # Update convergence detector
            self.executor.convergence_detector.update(
                request["iteration"],
                request["best_confidence"],
                request["best_reward"]
            )

            # Check convergence
            is_converged, aci_value = self.executor.convergence_detector.check_convergence()

            self.logger.info(
                "Convergence check complete",
                is_converged=is_converged,
                aci_value=aci_value,
                iteration=request["iteration"]
            )

            return {
                "success": True,
                "is_converged": is_converged,
                "aci_value": aci_value,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            error_msg = f"Convergence check failed: {str(e)}"
            self.logger.error(error_msg, error=str(e))
            return {
                "success": False,
                "error": error_msg,
                "correlation_id": correlation_id,
            }

    def get_health(self) -> Dict[str, Any]:
        """Get adapter health status."""
        return {
            "status": "healthy" if self.executor.circuit_breaker.state == "CLOSED" else "degraded",
            "circuit_breaker_state": self.executor.circuit_breaker.state,
            "dlq_size": self.executor.dlq.size(),
            "config": self.config.__dict__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_dlq_contents(self) -> List[Dict[str, Any]]:
        """Get contents of Dead Letter Queue."""
        return self.executor.dlq.get_all()

    def clear_dlq(self):
        """Clear Dead Letter Queue."""
        self.executor.dlq.clear()

    def _validate_request(self, request: Dict[str, Any]) -> List[str]:
        """Validate search request."""
        errors = []

        if "root_hypothesis" not in request:
            errors.append("Missing required field: root_hypothesis")
        elif not isinstance(request["root_hypothesis"], dict):
            errors.append("root_hypothesis must be a dictionary")

        optional_fields = ["hypothesis_generator_config", "reward_function_config", "correlation_id"]
        for field in optional_fields:
            if field in request and not isinstance(request[field], dict):
                errors.append(f"{field} must be a dictionary")

        return errors

    def _to_canonical_format(self, result: MCTSSearchResult) -> Dict[str, Any]:
        """Transform MCTS search result to canonical format."""
        return {
            "success": True,
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
            "metadata": result.metadata,
            "timestamp": result.created_at.isoformat() if isinstance(result.created_at, datetime) else result.created_at,
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_adapter(config: Optional[Phase3Config] = None) -> Phase3Adapter:
    """
    Factory function to create Phase III adapter.

    Args:
        config: Optional configuration

    Returns:
        Phase3Adapter instance

    Raises:
        RuntimeError: If configuration validation fails
    """
    return Phase3Adapter(config)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface for Phase III adapter."""
    import argparse

    parser = argparse.ArgumentParser(description="RESE Phase III MCTS Search Adapter")
    parser.add_argument("--config", action="store_true", help="Show configuration and exit")
    parser.add_argument("--health", action="store_true", help="Show health status and exit")
    parser.add_argument("--dlq", action="store_true", help="Show Dead Letter Queue contents")

    args = parser.parse_args()

    # Initialize adapter
    adapter = Phase3Adapter()

    if args.config:
        print("Configuration:")
        print(json.dumps(adapter.config.__dict__, indent=2))
        return

    if args.health:
        print("Health Status:")
        print(json.dumps(adapter.get_health(), indent=2))
        return

    if args.dlq:
        print("Dead Letter Queue:")
        print(json.dumps(adapter.get_dlq_contents(), indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    "Phase3Adapter",
    "create_adapter",
]
