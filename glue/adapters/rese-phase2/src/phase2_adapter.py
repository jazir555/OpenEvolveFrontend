"""
RESE Phase II Adapter

Adapter for Isomorphic Mapping (Phase II of RESE).
Provides RESTful API interface, request validation, and response transformation.

Following CLAUDE.md principles:
- Law of Configuration Explicitness: Env vars required
- Law of Idempotency: UPSERT logic for mappings
- Circuit Breaker: Detect mapping failures
- Structured Logging: JSON with correlation_id
- Timeout: All operations bounded (default 20000ms)

Author: RESE Team
Created: 2026-02-04
"""

import os
import sys
import json
import uuid
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "schemas"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from phase2_executor import (
        IsomorphicMappingExecutor,
        Phase2Logger,
        create_executor,
    )
    from rese_schemas import (
        Phase2Config,
        IsomorphicMappingResult,
    )
except ImportError:
    from glue.adapters.rese-phase2.src.phase2_executor import (
        IsomorphicMappingExecutor,
        Phase2Logger,
        create_executor,
    )
    from glue.schemas.rese_schemas import (
        Phase2Config,
        IsomorphicMappingResult,
    )


# ============================================================================
# DEAD LETTER QUEUE (DLQ)
# ============================================================================

class DeadLetterQueue:
    """Dead Letter Queue for failed mapping requests."""

    def __init__(self, logger: Optional[Phase2Logger] = None):
        self.logger = logger or Phase2Logger()
        self.failed_requests: List[Dict[str, Any]] = []
        self.max_size = int(os.getenv("PHASE2_DLQ_MAX_SIZE", "1000"))

    def add(self, request: Dict[str, Any], error: str, error_type: str):
        """Add failed request to DLQ."""
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
# PHASE II ADAPTER
# ============================================================================

class Phase2Adapter:
    """
    Adapter for RESE Phase II: Isomorphic Mapping.

    Provides:
    - Request validation
    - Phase II execution
    - Response transformation
    - Error handling with DLQ
    """

    def __init__(self):
        # Validate configuration at startup (Law of Configuration Explicitness)
        self.config = self._validate_config()
        self.logger = Phase2Logger()
        self.dlq = DeadLetterQueue(self.logger)

        # Initialize executor
        self.executor = create_executor(self.config)

        self.logger.info(
            "Phase II adapter initialized",
            config=self.config.to_dict()
        )

    def _validate_config(self) -> Phase2Config:
        """
        Validate configuration from environment.

        Crashes immediately if required vars are missing (Law of Configuration Explicitness).
        """
        try:
            config = Phase2Config.from_env()
            return config
        except Exception as e:
            print(f"FATAL: Configuration validation failed: {e}")
            print("Required environment variables:")
            print("  - PHASE2_MAX_TARGET_DOMAINS")
            print("  - PHASE2_IMECH_THRESHOLD")
            print("  - PHASE2_PATTERN_THRESHOLD")
            print("  - PHASE2_TIMEOUT_MS")
            print("  - PHASE2_MAX_MAPPINGS")
            print("  - PHASE2_ENABLE_CONSTRAINT_INVERSION")
            print("  - PHASE2_SEARCH_DEPTH")
            sys.exit(1)

    def execute_phase2(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Phase II: Isomorphic Mapping.

        Args:
            request: Phase II request with keys:
                - source_domain (str, required)
                - problem_description (str, required)
                - target_domains (list of str, optional)
                - constraints (list of str, optional)
                - context (dict, optional)
                - correlation_id (str, optional)

        Returns:
            Phase II result in canonical format

        Raises:
            ValueError: If request validation fails
        """
        correlation_id = request.get("correlation_id") or str(uuid.uuid4())
        self.logger = Phase2Logger(correlation_id)

        # Validate request
        validation_errors = self._validate_request(request)
        if validation_errors:
            error_msg = f"Request validation failed: {validation_errors}"
            self.logger.error(error_msg, errors=validation_errors)
            self.dlq.add(request, error_msg, "logic")
            raise ValueError(error_msg)

        source_domain = request["source_domain"]
        problem_description = request["problem_description"]
        target_domains = request.get("target_domains")
        constraints = request.get("constraints")
        context = request.get("context")

        self.logger.info(
            "Phase II execution request received",
            source_domain=source_domain,
            problem_length=len(problem_description),
            target_count=len(target_domains) if target_domains else 0
        )

        try:
            # Execute Phase II
            result = self.executor.execute_phase2(
                source_domain=source_domain,
                problem_description=problem_description,
                target_domains=target_domains,
                constraints=constraints,
                context=context
            )

            # Transform to canonical format
            canonical_result = self._to_canonical_format(result)

            self.logger.info(
                "Phase II execution complete",
                result_id=canonical_result["result_id"],
                mapping_count=canonical_result["mapping_count"],
                best_imech=canonical_result.get("best_imech_score", 0.0)
            )

            return canonical_result

        except Exception as e:
            error_msg = f"Phase II execution failed: {str(e)}"
            self.logger.error(error_msg, error=str(e))

            # Add to DLQ based on error type
            error_type = self._classify_error(e)
            self.dlq.add(request, error_msg, error_type)

            # Re-raise for client handling
            raise

    def _validate_request(self, request: Dict[str, Any]) -> List[str]:
        """Validate Phase II request."""
        errors = []

        if "source_domain" not in request:
            errors.append("Missing required field: source_domain")
        elif not isinstance(request["source_domain"], str):
            errors.append("source_domain must be a string")

        if "problem_description" not in request:
            errors.append("Missing required field: problem_description")
        elif not isinstance(request["problem_description"], str):
            errors.append("problem_description must be a string")

        if "target_domains" in request and not isinstance(request["target_domains"], list):
            errors.append("target_domains must be a list")

        if "constraints" in request and not isinstance(request["constraints"], list):
            errors.append("constraints must be a list")

        if "context" in request and not isinstance(request["context"], dict):
            errors.append("context must be a dictionary")

        return errors

    def _to_canonical_format(self, result: IsomorphicMappingResult) -> Dict[str, Any]:
        """
        Transform Phase II result to canonical format.

        Canonical format ensures consistent API responses.
        """
        return {
            "result_id": result.result_id,
            "source_domain": result.source_domain,
            "target_domains": result.target_domains,
            "mappings": [
                {
                    "mapping_id": m.mapping_id,
                    "source_domain": m.source_domain,
                    "target_domain": m.target_domain,
                    "isomorphism_type": m.isomorphism_type.value if isinstance(m.isomorphism_type, type(IsomorphismType.STRUCTURAL)) else m.isomorphism_type,
                    "i_mech_score": m.i_mech_score,
                    "fdg_overlap": m.fdg_overlap,
                    "confidence": m.confidence,
                    "validated": m.validated,
                }
                for m in result.mappings_found
            ],
            "best_mapping": {
                "mapping_id": result.best_mapping.mapping_id,
                "source_domain": result.best_mapping.source_domain,
                "target_domain": result.best_mapping.target_domain,
                "i_mech_score": result.best_mapping.i_mech_score,
                "confidence": result.best_mapping.confidence,
            } if result.best_mapping else None,
            "cross_domain_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "name": p.name,
                    "type": p.type.value if isinstance(p.type, type(PatternType.STRUCTURAL)) else p.type,
                    "domains": p.domains,
                    "confidence": p.confidence,
                }
                for p in result.cross_domain_patterns
            ],
            "inverted_constraints": [
                {
                    "constraint_id": c.constraint_id,
                    "original": c.original_constraint,
                    "inverted": c.inverted_constraint,
                    "inversion_type": c.inversion_type,
                    "reduction_factor": c.search_space_reduction,
                }
                for c in result.inverted_constraints
            ],
            "summary": {
                "mapping_count": len(result.mappings_found),
                "pattern_count": len(result.cross_domain_patterns),
                "inverted_count": len(result.inverted_constraints),
                "best_imech_score": result.best_mapping.i_mech_score if result.best_mapping else 0.0,
                "overall_confidence": result.confidence,
            },
            "execution_time_ms": result.execution_time_ms,
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
            "status": "healthy",
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
    """CLI interface for Phase II adapter."""
    import argparse

    parser = argparse.ArgumentParser(description="RESE Phase II: Isomorphic Mapping Adapter")
    parser.add_argument("--source", type=str, help="Source domain")
    parser.add_argument("--problem", type=str, help="Problem description")
    parser.add_argument("--targets", type=str, nargs="+", help="Target domains")
    parser.add_argument("--config", action="store_true", help="Show configuration and exit")
    parser.add_argument("--health", action="store_true", help="Show health status and exit")
    parser.add_argument("--dlq", action="store_true", help="Show Dead Letter Queue contents")

    args = parser.parse_args()

    # Initialize adapter
    adapter = Phase2Adapter()

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

    if args.source and args.problem:
        print("Executing Phase II...")
        request = {
            "source_domain": args.source,
            "problem_description": args.problem,
        }
        if args.targets:
            request["target_domains"] = args.targets

        result = adapter.execute_phase2(request)
        print("\nResult:")
        print(json.dumps(result, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
