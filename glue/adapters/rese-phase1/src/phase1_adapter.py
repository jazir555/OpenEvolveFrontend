#!/usr/bin/env python3
"""
RESE Phase I Adapter

This adapter provides the interface between the Phase I executor and the
Symbolic Constraint Engine (SCE).

Following CLAUDE.md principles:
- Law of the "Air Gap": No imports from core-projects
- Law of Runtime Truth: Verify SCE integration via probes
- Law of Idempotency: All operations safe to run 100x
- Circuit Breaker Pattern: Detect SCE failures
- Structured Logging: JSON with correlation_id
"""

import os
import sys
import json
import subprocess
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import logging

# Add local src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from phase1_executor import (
    Phase1Config,
    EpistemicAuditExecutor,
    EpistemicAuditResult,
    TacitAssumption,
    ContradictionDetection,
    FalsificationResult,
    StructuredLogger,
    CircuitBreaker,
)


class SCEAdapter:
    """Adapter for Symbolic Constraint Engine (SCE)

    Integrates with the TypeScript SCE implementation at glue/lib/rese-sce.ts

    Law of the "Air Gap": This adapter translates between Python and TypeScript,
    maintaining isolation between the two systems.
    """

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.sce_path = os.path.join(
            os.path.dirname(__file__),
            '../../lib/rese-sce.ts'
        )

        # Circuit breaker for SCE calls
        self.circuit_breaker = CircuitBreaker(
            threshold=int(os.getenv('SCE_CIRCUIT_BREAKER_THRESHOLD', '5')),
            timeout_ms=int(os.getenv('SCE_CIRCUIT_BREAKER_TIMEOUT_MS', '60000')),
            logger=logger.logger,
        )

        self.logger.info("SCEAdapter initialized", {
            'sce_path': self.sce_path,
        })

    def detect_contradictions(
        self,
        assumptions: List[TacitAssumption],
        correlation_id: str,
    ) -> List[ContradictionDetection]:
        """Detect contradictions using SCE

        Law of Runtime Truth: Execute SCE via node subprocess
        Law of Timeout Enforcement: All operations have timeouts

        Args:
            assumptions: Tacit assumptions to check
            correlation_id: Correlation ID for tracing

        Returns:
            List of detected contradictions
        """
        self.logger.info("Detecting contradictions via SCE", {
            'correlation_id': correlation_id,
            'assumption_count': len(assumptions),
        })

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise RuntimeError("SCE circuit breaker is OPEN")

        try:
            # Prepare input for SCE
            sce_input = {
                'operation': 'detectContradictions',
                'assumptions': [a.to_dict() for a in assumptions],
                'correlationId': correlation_id,
            }

            # Execute SCE via Node.js
            result = self._execute_sce(sce_input, correlation_id)

            # Transform result to canonical format
            contradictions = [
                ContradictionDetection.from_dict(c) for c in result.get('contradictions', [])
            ]

            self.logger.info("Contradiction detection completed", {
                'correlation_id': correlation_id,
                'contradictions_found': len(contradictions),
            })

            self.circuit_breaker.record_success()
            return contradictions

        except Exception as e:
            self.logger.error("SCE contradiction detection failed", e, {
                'correlation_id': correlation_id,
            })
            self.circuit_breaker.record_failure()
            raise

    def _execute_sce(self, input_data: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
        """Execute SCE operation

        Law of Timeout Enforcement: Enforced timeout
        """
        timeout_ms = int(os.getenv('SCE_TIMEOUT_MS', '10000'))
        timeout_sec = timeout_ms / 1000.0

        # Create temporary input file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name

        try:
            # Execute Node.js script
            cmd = [
                'node',
                '-e',
                f'''
const SCE = require('{self.sce_path}');
const fs = require('fs');

const input = JSON.parse(fs.readFileSync('{input_file}', 'utf8'));
const engine = new SCE.SymbolicConstraintEngine();

(async () => {{
    try {{
        if (input.operation === 'detectContradictions') {{
            const constraints = input.assumptions.map(a => ({{
                constraint_id: a.id,
                type: 'soft',
                category: 'tacit_assumption',
                description: a.description,
                dependencies: [],
                created_at: new Date()
            }}));

            const result = await engine.detectContradictions(input.correlationId);
            console.log(JSON.stringify(result));
        }}
    }} catch (error) {{
        console.error(JSON.stringify({{ error: error.message }}));
        process.exit(1);
    }}
}})();
                '''
            ]

            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            execution_time_ms = int((time.time() - start_time) * 1000)

            if result.returncode != 0:
                raise RuntimeError(f"SCE execution failed: {result.stderr}")

            # Parse output
            output = json.loads(result.stdout)

            self.logger.debug("SCE execution completed", {
                'correlation_id': correlation_id,
                'execution_time_ms': execution_time_ms,
            })

            return output

        finally:
            # Cleanup temp file
            try:
                os.unlink(input_file)
            except:
                pass


class Phase1Adapter:
    """Main adapter for Phase I operations

    Integrates:
    - EpistemicAuditExecutor (Python)
    - SCEAdapter (TypeScript SCE integration)
    - Canonical schema transformation
    """

    def __init__(self, config: Phase1Config = None):
        """Initialize Phase I adapter

        Args:
            config: Configuration object (loaded from env if None)
        """
        self.config = config or Phase1Config.from_env()
        self.logger = StructuredLogger('Phase1Adapter')

        # Initialize executor
        self.executor = EpistemicAuditExecutor(config=self.config)

        # Initialize SCE adapter
        self.sce_adapter = SCEAdapter(logger=self.logger)

        self.logger.info("Phase1Adapter initialized", {
            'enable_sce_integration': True,
            'enable_tacit_mining': self.config.ENABLE_TACIT_MINING,
            'enable_red_team': self.config.ENABLE_RED_TEAM_PROTOCOL,
        })

    def perform_audit(
        self,
        problem_description: str,
        failure_patterns: List[Dict[str, Any]],
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform Phase I audit and return canonical result

        Law of the "Air Gap": Transform to canonical format
        Law of Idempotency: Safe to run 100x

        Args:
            problem_description: Description of the problem to audit
            failure_patterns: Patterns of failure for tacit assumption mining
            correlation_id: Distributed tracing correlation ID

        Returns:
            Canonical EpistemicAuditResult as dict
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        self.logger.info("Starting Phase I audit via adapter", {
            'correlation_id': correlation_id,
            'problem_description': problem_description,
        })

        # Perform audit
        result = self.executor.perform_audit(
            problem_description=problem_description,
            failure_patterns=failure_patterns,
            correlation_id=correlation_id,
        )

        # Transform to canonical format (already in canonical format from executor)
        canonical_result = result.to_dict()

        # Add integration metadata
        canonical_result['metadata']['adapter_version'] = '1.0.0'
        canonical_result['metadata']['sce_integration'] = 'enabled'

        self.logger.info("Phase I audit completed via adapter", {
            'correlation_id': correlation_id,
            'audit_id': result.audit_id,
        })

        return canonical_result

    def health_check(self) -> Dict[str, Any]:
        """Health check for adapter

        Returns adapter status and component health
        """
        health = {
            'status': 'healthy',
            'adapter': 'phase1',
            'version': '1.0.0',
            'components': {
                'executor': 'healthy',
                'sce_adapter': self.sce_adapter.circuit_breaker.get_stats(),
            },
            'config': {
                'max_assumptions': self.config.MAX_ASSUMPTIONS,
                'max_constraints': self.config.MAX_CONSTRAINTS,
                'enable_tacit_mining': self.config.ENABLE_TACIT_MINING,
                'enable_red_team': self.config.ENABLE_RED_TEAM_PROTOCOL,
            },
            'stats': self.executor.get_stats(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        # Determine overall status
        if self.sce_adapter.circuit_breaker.get_stats()['state'] == 'open':
            health['status'] = 'degraded'

        return health


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for adapter CLI"""
    import argparse
    import uuid

    parser = argparse.ArgumentParser(description='RESE Phase I Adapter')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Audit command
    audit_parser = subparsers.add_parser('audit', help='Perform Phase I audit')
    audit_parser.add_argument('--problem', required=True, help='Problem description')
    audit_parser.add_argument('--patterns', required=True, help='Failure patterns (JSON)')
    audit_parser.add_argument('--correlation-id', help='Correlation ID')

    # Health check command
    subparsers.add_parser('health', help='Health check')

    args = parser.parse_args()

    # Create adapter
    adapter = Phase1Adapter()

    if args.command == 'audit':
        # Parse failure patterns
        failure_patterns = json.loads(args.patterns)

        # Perform audit
        result = adapter.perform_audit(
            problem_description=args.problem,
            failure_patterns=failure_patterns,
            correlation_id=args.correlation_id,
        )

        # Output result as JSON
        print(json.dumps(result, indent=2))

    elif args.command == 'health':
        # Health check
        health = adapter.health_check()
        print(json.dumps(health, indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
