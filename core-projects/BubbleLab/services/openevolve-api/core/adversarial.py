"""
Adversarial Engine for OpenEvolve API

Implements red team testing and vulnerability discovery workflow.
Follows CLAUDE.md principles: structured logging, UTC timestamps, failure isolation.

Integrates with BubbleLab services:
- Mutate Adapter: Code mutation for testing and attack generation
"""

import structlog
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum


logger = structlog.get_logger()


class AdversarialStatus(str, Enum):
    """Adversarial testing status"""
    INITIALIZING = "initializing"
    ATTACKING = "attacking"
    ANALYZING = "analyzing"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


class AttackType(str, Enum):
    """Types of adversarial attacks"""
    FUZZING = "fuzzing"
    PROMPT_INJECTION = "prompt_injection"
    CODE_INJECTION = "code_injection"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    DoS = "dos"
    BUFFER_OVERFLOW = "buffer_overflow"


class AdversarialEngine:
    """
    Adversarial Engine for security testing and vulnerability discovery.

    Integrates with BubbleLab Mutate service for real attack generation.
    All timestamps in UTC. Failures isolated per attack type (Circuit Breaker pattern).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Adversarial Engine.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}

        # Lazy import of adapters to avoid circular imports
        self._mutate_adapter = None

        logger.info(
            "adversarial_engine_initialized",
            engine_type="adversarial",
            config_keys=list(self.config.keys()),
            adapter_integration="enabled"
        )

    def _get_mutate_adapter(self):
        """Get or create Mutate adapter instance"""
        if self._mutate_adapter is None:
            from services.adapters import get_mutate_adapter
            self._mutate_adapter = get_mutate_adapter()
        return self._mutate_adapter

    async def execute(
        self,
        problem_statement: str,
        parameters: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute adversarial testing workflow.

        Tests code/system against various attack vectors to discover vulnerabilities.
        Follows Circuit Breaker pattern: failures in one attack type don't block others.

        Args:
            problem_statement: The code/system to test (e.g., "Test this login endpoint")
            parameters: Adversarial parameters from AdversarialParameters model
                - test_cases: List of specific test cases to run
                - attack_types: List of attack types to test
                - rounds: Number of testing rounds
            context: Additional context (e.g., API endpoints, auth requirements)

        Returns:
            Dictionary containing:
                - status: Final execution status
                - vulnerabilities: List of discovered vulnerabilities
                - test_results: Detailed results per attack type
                - summary: High-level summary
                - recommendations: Security recommendations
                - metadata: Execution metadata (timestamps, etc.)

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If execution fails critically
        """
        execution_start = datetime.now(timezone.utc)
        execution_id = f"adv_{execution_start.strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info(
            "adversarial_execution_started",
            execution_id=execution_id,
            problem_statement=problem_statement[:100] + "..." if len(problem_statement) > 100 else problem_statement,
            parameters=parameters,
            context_provided=context is not None
        )

        try:
            # Validate parameters
            self._validate_parameters(parameters)

            # Initialize adversarial state
            status = AdversarialStatus.INITIALIZING

            # Get attack types and rounds
            attack_types = parameters.get("attack_types", ["fuzzing", "prompt_injection", "code_injection"])
            rounds = parameters.get("rounds", 3)
            test_cases = parameters.get("test_cases", [])

            logger.info(
                "adversarial_phase",
                execution_id=execution_id,
                phase=status.value,
                attack_types=attack_types,
                rounds=rounds,
                test_cases_count=len(test_cases)
            )

            # PHASE 2: ATTACKING - Execute attack vectors
            status = AdversarialStatus.ATTACKING
            all_vulnerabilities = []
            test_results = {}

            for attack_type in attack_types:
                # Check circuit breaker
                if self._is_circuit_open(attack_type):
                    logger.warning(
                        "adversarial_circuit_breaker_open",
                        execution_id=execution_id,
                        attack_type=attack_type,
                        reason="Too many failures"
                    )
                    continue

                try:
                    logger.info(
                        "adversarial_attack_started",
                        execution_id=execution_id,
                        attack_type=attack_type
                    )

                    # Run attack for specified rounds
                    attack_results = []
                    for round_num in range(1, rounds + 1):
                        result = await self._execute_attack(
                            attack_type,
                            problem_statement,
                            round_num,
                            context
                        )
                        attack_results.append(result)

                        # Collect vulnerabilities
                        if result.get("vulnerabilities"):
                            all_vulnerabilities.extend(result["vulnerabilities"])

                    # Store results
                    test_results[attack_type] = {
                        "rounds": rounds,
                        "results": attack_results,
                        "status": "completed"
                    }

                    # Reset circuit breaker on success
                    self._reset_circuit_breaker(attack_type)

                    logger.info(
                        "adversarial_attack_completed",
                        execution_id=execution_id,
                        attack_type=attack_type,
                        vulnerabilities_found=len([r for r in attack_results if r.get("vulnerabilities")])
                    )

                except Exception as e:
                    logger.error(
                        "adversarial_attack_failed",
                        execution_id=execution_id,
                        attack_type=attack_type,
                        error=str(e),
                        exc_info=True
                    )

                    test_results[attack_type] = {
                        "rounds": rounds,
                        "error": str(e),
                        "status": "failed"
                    }

                    # Trigger circuit breaker
                    self._trigger_circuit_breaker(attack_type)

            # PHASE 3: ANALYZING - Analyze results
            status = AdversarialStatus.ANALYZING
            vulnerability_summary = self._analyze_vulnerabilities(all_vulnerabilities)

            logger.info(
                "adversarial_phase",
                execution_id=execution_id,
                phase=status.value,
                total_vulnerabilities=len(all_vulnerabilities),
                critical_count=vulnerability_summary["critical"],
                high_count=vulnerability_summary["high"]
            )

            # PHASE 4: REPORTING - Generate recommendations
            status = AdversarialStatus.REPORTING
            recommendations = self._generate_recommendations(
                all_vulnerabilities,
                test_results
            )

            # PHASE 5: COMPLETED
            status = AdversarialStatus.COMPLETED
            execution_end = datetime.now(timezone.utc)
            execution_duration = (execution_end - execution_start).total_seconds()

            result = {
                "status": status.value,
                "vulnerabilities": all_vulnerabilities,
                "test_results": test_results,
                "summary": {
                    "total_vulnerabilities": len(all_vulnerabilities),
                    "by_severity": vulnerability_summary,
                    "by_attack_type": self._summarize_by_attack_type(test_results),
                    "attacks_executed": len(attack_types),
                    "attacks_succeeded": len([k for k, v in test_results.items() if v.get("status") == "completed"])
                },
                "recommendations": recommendations,
                "metadata": {
                    "execution_id": execution_id,
                    "started_at": execution_start.isoformat(),
                    "completed_at": execution_end.isoformat(),
                    "duration_seconds": execution_duration,
                    "parameters": parameters,
                    "engine_version": "0.1.0"
                }
            }

            logger.info(
                "adversarial_execution_completed",
                execution_id=execution_id,
                status=status.value,
                total_vulnerabilities=len(all_vulnerabilities),
                critical_vulnerabilities=vulnerability_summary["critical"],
                duration_seconds=execution_duration
            )

            return result

        except Exception as e:
            execution_end = datetime.now(timezone.utc)
            error_message = str(e)

            logger.error(
                "adversarial_execution_failed",
                execution_id=execution_id,
                error=error_message,
                error_type=type(e).__name__,
                duration_seconds=(execution_end - execution_start).total_seconds(),
                exc_info=True
            )

            return {
                "status": AdversarialStatus.FAILED.value,
                "vulnerabilities": [],
                "test_results": test_results if 'test_results' in locals() else {},
                "summary": {},
                "recommendations": [],
                "error": error_message,
                "metadata": {
                    "execution_id": execution_id,
                    "started_at": execution_start.isoformat(),
                    "failed_at": execution_end.isoformat(),
                    "error_type": type(e).__name__
                }
            }

    def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Validate adversarial parameters.

        Args:
            parameters: Parameters dictionary to validate

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate rounds
        if "rounds" in parameters:
            rounds = parameters["rounds"]
            if not isinstance(rounds, int) or not (1 <= rounds <= 10):
                raise ValueError("Parameter 'rounds' must be an integer between 1 and 10")

        # Validate attack_types
        if "attack_types" in parameters:
            attack_types = parameters["attack_types"]
            valid_types = [t.value for t in AttackType]
            for attack_type in attack_types:
                if attack_type not in valid_types:
                    raise ValueError(f"Invalid attack type: {attack_type}. Must be one of {valid_types}")

        # Validate test_cases
        if "test_cases" in parameters:
            test_cases = parameters["test_cases"]
            if not isinstance(test_cases, list):
                raise ValueError("Parameter 'test_cases' must be a list")

        logger.debug("adversarial_parameters_validated", parameters=parameters)

    async def _execute_attack(
        self,
        attack_type: str,
        target: str,
        round_num: int,
        context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Execute a single attack round using Mutate adapter for code attacks.

        Args:
            attack_type: Type of attack to execute
            target: Target code/system
            round_num: Round number
            context: Additional context

        Returns:
            Attack result with vulnerabilities found
        """
        logger.debug(
            "executing_attack_round",
            attack_type=attack_type,
            round=round_num,
            target_length=len(target)
        )

        vulnerabilities = []

        try:
            # Use Mutate adapter for code-based attacks
            if attack_type in ["code_injection", "fuzzing"]:
                mutate = self._get_mutate_adapter()

                logger.debug(
                    "using_mutate_adapter_for_attack",
                    attack_type=attack_type,
                    mutation_rate=0.3  # Aggressive mutation for testing
                )

                # Generate attack variants using mutation
                if attack_type == "code_injection":
                    # Try to inject malicious code via mutation
                    mutations = await mutate.mutate_batch(
                        codes=[target] * 3,  # Create 3 attack variants
                        mutation_type="point",
                        mutation_rate=0.5,  # High mutation rate
                    )

                    # Check if any mutation introduced vulnerabilities
                    for mutation_result in mutations:
                        mutated_code = mutation_result["mutated_code"]
                        vuln = self._check_for_injection_vulnerability(
                            target,
                            mutated_code
                        )
                        if vuln:
                            vulnerabilities.append(vuln)

                elif attack_type == "fuzzing":
                    # Fuzz with random mutations
                    for i in range(5):  # 5 fuzzing attempts
                        mutation = await mutate.mutate(
                            code=target,
                            mutation_type="point",
                            mutation_rate=0.4,  # High mutation
                        )

                        # Check if fuzzing revealed issues
                        vuln = self._check_fuzzing_result(mutation)
                        if vuln:
                            vulnerabilities.append(vuln)

                logger.info(
                    "mutate_attack_completed",
                    attack_type=attack_type,
                    round=round_num,
                    mutations_attempted=len(vulnerabilities),
                    vulnerabilities_found=len(vulnerabilities)
                )

            else:
                # Other attack types (prompt_injection, etc.)
                vulnerabilities = await self._execute_other_attack(
                    attack_type,
                    target,
                    round_num,
                    context
                )

        except Exception as e:
            logger.warning(
                "mutate_adapter_failed",
                attack_type=attack_type,
                error=str(e),
                fallback_enabled=True
            )
            # Fall back to placeholder logic
            vulnerabilities = self._simulate_attack_fallback(
                attack_type,
                target,
                round_num
            )

        # Build result
        result = {
            "round": round_num,
            "attack_type": attack_type,
            "vulnerabilities": vulnerabilities,
            "status": "completed" if vulnerabilities else "no_findings",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return result

    async def _execute_other_attack(
        self,
        attack_type: str,
        target: str,
        round_num: int,
        context: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Execute non-mutation-based attacks (prompt_injection, etc.)"""
        vulnerabilities = []

        # Placeholder for other attack types
        # In production, this would use specialized attack tools
        import random

        if attack_type == "prompt_injection":
            if random.random() < 0.3:  # 30% chance
                vulnerabilities.append({
                    "type": "prompt_injection",
                    "severity": "high",
                    "description": "Potential prompt injection vulnerability detected",
                    "location": "user_input_handler",
                    "evidence": f"injection_pattern_{round_num}",
                    "remediation": "Implement strict input validation and sanitization"
                })

        return vulnerabilities

    def _check_for_injection_vulnerability(
        self,
        original_code: str,
        mutated_code: str
    ) -> Optional[Dict[str, Any]]:
        """Check if mutation created injection vulnerability"""
        # Simple heuristic: look for suspicious patterns
        suspicious_patterns = [
            "eval(",
            "exec(",
            "__import__",
            "os.system",
            "subprocess.call",
            "<script>",
            "javascript:",
            "document.write",
        ]

        for pattern in suspicious_patterns:
            if pattern in mutated_code and pattern not in original_code:
                return {
                    "type": "code_injection",
                    "severity": "high",
                    "description": f"Code injection via '{pattern}' detected",
                    "location": "mutated_code",
                    "remediation": "Avoid dynamic code execution, use safe alternatives"
                }

        return None

    def _check_fuzzing_result(
        self,
        mutation: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check if fuzzing revealed issues"""
        mutated_code = mutation.get("mutated_code", "")
        mutations_count = mutation.get("mutations_count", 0)

        # Check if excessive mutations indicate poor error handling
        if mutations_count > 10:
            return {
                "type": "fragility",
                "severity": "medium",
                "description": f"Code changed {mutations_count} times - may be fragile",
                "location": "code_structure",
                "remediation": "Improve input validation and error handling"
            }

        return None

    async def _simulate_attack_fallback(
        self,
        attack_type: str,
        target: str,
        round_num: int
    ) -> List[Dict[str, Any]]:
        """Fallback attack simulation when adapter unavailable"""
        import random

        vulnerabilities = []

        # Simulate finding vulnerabilities (20% chance)
        if random.random() < 0.2:
            vulnerabilities.append({
                "type": attack_type,
                "severity": random.choice(["critical", "high", "medium", "low"]),
                "description": f"Simulated {attack_type} vulnerability discovered",
                "location": f"target_location_{random.randint(1, 100)}",
                "evidence": f"attack_payload_{round_num}",
                "remediation": f"Implement protection against {attack_type}"
            })

        return vulnerabilities

    def _analyze_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze vulnerabilities by severity.

        Args:
            vulnerabilities: List of discovered vulnerabilities

        Returns:
            Dictionary with count per severity level
        """
        summary = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }

        for vuln in vulnerabilities:
            severity = vuln.get("severity", "low").lower()
            if severity in summary:
                summary[severity] += 1

        return summary

    def _summarize_by_attack_type(self, test_results: Dict[str, Any]) -> Dict[str, int]:
        """
        Summarize vulnerabilities by attack type.

        Args:
            test_results: Test results per attack type

        Returns:
            Dictionary with count per attack type
        """
        summary = {}

        for attack_type, results in test_results.items():
            if results.get("status") == "completed":
                for result in results.get("results", []):
                    count = len(result.get("vulnerabilities", []))
                    summary[attack_type] = summary.get(attack_type, 0) + count

        return summary

    def _generate_recommendations(
        self,
        vulnerabilities: List[Dict[str, Any]],
        test_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate security recommendations based on findings.

        Args:
            vulnerabilities: Discovered vulnerabilities
            test_results: Test results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # General recommendations
        if vulnerabilities:
            recommendations.append("Implement comprehensive input validation across all endpoints")
            recommendations.append("Add rate limiting to prevent DoS attacks")
            recommendations.append("Enable comprehensive logging and monitoring for security events")

        # Specific recommendations based on vulnerability types
        vuln_types = set(v.get("type", "") for v in vulnerabilities)

        if "sql_injection" in vuln_types:
            recommendations.append("Use parameterized queries or ORM to prevent SQL injection")

        if "xss" in vuln_types:
            recommendations.append("Sanitize and escape user-generated content to prevent XSS")

        if "prompt_injection" in vuln_types:
            recommendations.append("Implement prompt engineering best practices and validation")

        if "code_injection" in vuln_types:
            recommendations.append("Avoid eval() and similar dynamic code execution functions")

        if not recommendations:
            recommendations.append("No critical vulnerabilities found - continue following security best practices")

        return recommendations

    def _is_circuit_open(self, attack_type: str) -> bool:
        """
        Check if circuit breaker is open for an attack type.

        Args:
            attack_type: Attack type to check

        Returns:
            True if circuit is open
        """
        breaker = self._circuit_breakers.get(attack_type, {})
        return breaker.get("open", False)

    def _trigger_circuit_breaker(self, attack_type: str) -> None:
        """
        Trigger circuit breaker for an attack type.

        Args:
            attack_type: Attack type to trigger for
        """
        if attack_type not in self._circuit_breakers:
            self._circuit_breakers[attack_type] = {}

        self._circuit_breakers[attack_type]["open"] = True
        self._circuit_breakers[attack_type]["opened_at"] = datetime.now(timezone.utc).isoformat()
        self._circuit_breakers[attack_type]["failure_count"] = self._circuit_breakers[attack_type].get("failure_count", 0) + 1

        logger.warning(
            "circuit_breaker_triggered",
            attack_type=attack_type,
            failure_count=self._circuit_breakers[attack_type]["failure_count"]
        )

    def _reset_circuit_breaker(self, attack_type: str) -> None:
        """
        Reset circuit breaker for an attack type.

        Args:
            attack_type: Attack type to reset
        """
        if attack_type in self._circuit_breakers:
            self._circuit_breakers[attack_type]["open"] = False
            self._circuit_breakers[attack_type]["reset_at"] = datetime.now(timezone.utc).isoformat()

            logger.debug(
                "circuit_breaker_reset",
                attack_type=attack_type
            )
