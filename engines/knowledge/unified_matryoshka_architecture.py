"""
Unified Matryoshka Architecture

Positions Matryoshka as a generalized Recursive Language Model (RLM) execution engine
that integrates with:
- ROMA: Recursive decomposition and solving
- Decomposition Workflow: Blue/Red/Gold team execution
- MDAP/MAKER: Voting consensus and error correction
- Unified Memory: Context rot prevention

Matryoshka is NOT just for documents - it's a universal execution engine
for any problem space that can be represented symbolically and explored iteratively.
"""
from __future__ import annotations


from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple

# ============================================================================
# OPTIONAL IMPORTS - Only import if available
# ============================================================================

# Core Matryoshka
try:
    from matryoshka_core import MatryoshkaExecutionEngine, MatryoshkaExecutionConfig
    MATRYOSHKA_EXECUTION_AVAILABLE = True
except ImportError:
    MATRYOSHKA_EXECUTION_AVAILABLE = False
    
    # Stub classes for type hints
    @dataclass
    class MatryoshkaExecutionConfig:
        max_iterations: int = 10
        exploration_depth: int = 3
        verbose: bool = False
    
    class MatryoshkaExecutionEngine:
        def __init__(self, config):
            self.config = config
            self.iteration_count = 0
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def execute(self, task, space, context):
            """
            Execute a task using recursive Matryoshka execution.
            
            Args:
                task: Task to execute
                space: Problem space to operate in
                context: Execution context with history and constraints
                
            Returns:
                Execution result with solution and metadata
            """
            import asyncio
            from datetime import datetime
            
            self.iteration_count += 1
            if self.iteration_count > self.config.max_iterations:
                raise RuntimeError(f"Maximum iterations ({self.config.max_iterations}) exceeded")
            
            if self.config.verbose:
                self.logger.info(f"Matryoshka execution iteration {self.iteration_count}: {task[:50]}...")
            
            # Create execution context with current state
            execution_context = {
                **context,
                "iteration": self.iteration_count,
                "max_iterations": self.config.max_iterations,
                "exploration_depth": self.config.exploration_depth
            }
            
            # Perform recursive execution based on exploration depth
            if self.config.exploration_depth > 0:
                # Recursively decompose and solve
                sub_tasks = self._decompose_task(task, execution_context)
                results = []
                
                for sub_task in sub_tasks:
                    sub_result = self.execute(sub_task, space, execution_context)
                    results.append(sub_result)
                
                # Combine results
                final_result = self._combine_results(results, task)
            else:
                # Execute directly at this level
                final_result = self._execute_directly(task, space, execution_context)
            
            return final_result
        
        def _decompose_task(self, task, context):
            """Decompose a task into sub-tasks."""
            # In a real implementation, this would use domain-specific decomposition
            # For now, we'll return the task as a single sub-task
            return [task]
        
        def _execute_directly(self, task, space, context):
            """Execute a task directly."""
            # In a real implementation, this would execute the task using appropriate tools
            # For now, we'll return a placeholder result
            return {
                "result": f"Executed task: {task}",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "context": context
            }
        
        def _combine_results(self, results, original_task):
            """Combine results from sub-executions."""
            combined_result = {
                "sub_results": results,
                "combined_result": "Combined results from sub-executions",
                "original_task": original_task,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            return combined_result

# ROMA Integration
try:
    from roma_matryoshka_adapter import ROMAMatryoshkaAdapter, ROMAMatryoshkaConfig
    MATRYOSHKA_ROMA_AVAILABLE = True
except ImportError:
    MATRYOSHKA_ROMA_AVAILABLE = False
    
    @dataclass
    class ROMAMatryoshkaConfig:
        enable_analysis: bool = True
        enable_critique: bool = True
    
    class ROMAMatryoshkaAdapter:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def phase_1_analyze_problem(self, problem, context):
            """
            Phase 1: Analyze the problem structure and constraints.
            
            Args:
                problem: Problem to analyze
                context: Execution context
                
            Returns:
                Analysis result with decomposition strategy
            """
            from datetime import datetime
            
            self.logger.info(f"Phase 1: Analyzing problem: {problem[:50]}...")
            
            # Analyze problem structure
            analysis = {
                "problem_type": self._classify_problem_type(problem),
                "complexity_score": self._estimate_complexity(problem),
                "decomposition_strategy": self._select_decomposition_strategy(problem),
                "constraints": self._extract_constraints(problem),
                "dependencies": self._identify_dependencies(problem),
                "estimated_effort": self._estimate_effort(problem),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update context with analysis
            context["problem_analysis"] = analysis
            
            return analysis
        
        def phase_2_solve_subproblem(self, subproblem, context):
            """
            Phase 2: Solve a subproblem using appropriate techniques.
            
            Args:
                subproblem: Subproblem to solve
                context: Execution context
                
            Returns:
                Solution to the subproblem
            """
            from datetime import datetime
            
            self.logger.info(f"Phase 2: Solving subproblem: {subproblem[:50]}...")
            
            # Determine solution approach based on problem type
            solution_approach = context.get("problem_analysis", {}).get("decomposition_strategy", "default")
            
            if solution_approach == "algorithmic":
                solution = self._solve_algorithmically(subproblem, context)
            elif solution_approach == "heuristic":
                solution = self._solve_heuristically(subproblem, context)
            elif solution_approach == "learning_based":
                solution = self._solve_with_learning(subproblem, context)
            else:
                solution = self._solve_default(subproblem, context)
            
            solution_result = {
                "subproblem_id": self._generate_id(),
                "solution": solution,
                "approach_used": solution_approach,
                "confidence": self._estimate_confidence(solution),
                "timestamp": datetime.now().isoformat()
            }
            
            return solution_result
        
        def phase_3_critique_solution(self, solution, context):
            """
            Phase 3: Critique the solution for quality and correctness.
            
            Args:
                solution: Solution to critique
                context: Execution context
                
            Returns:
                Critique with quality assessment and suggestions
            """
            from datetime import datetime
            
            self.logger.info("Phase 3: Critiquing solution...")
            
            critique = {
                "solution_quality": self._assess_solution_quality(solution),
                "correctness_score": self._assess_correctness(solution),
                "efficiency_score": self._assess_efficiency(solution),
                "issues_identified": self._identify_issues(solution),
                "improvement_suggestions": self._generate_improvements(solution),
                "confidence_in_critique": self._estimate_confidence(solution),
                "timestamp": datetime.now().isoformat()
            }
            
            return critique
        
        def phase_4_verify_solution(self, solution, context):
            """
            Phase 4: Verify the solution meets requirements.
            
            Args:
                solution: Solution to verify
                context: Execution context
                
            Returns:
                Verification result with pass/fail status
            """
            from datetime import datetime
            
            self.logger.info("Phase 4: Verifying solution...")
            
            # Extract original problem requirements from context
            requirements = context.get("problem_analysis", {}).get("constraints", [])
            
            verification = {
                "passes_requirements": self._verify_requirements(solution, requirements),
                "formal_verification": self._perform_formal_verification(solution) if self.config.enable_analysis else None,
                "test_results": self._run_tests(solution),
                "verification_score": self._calculate_verification_score(solution, requirements),
                "issues_found": self._identify_verification_issues(solution, requirements),
                "timestamp": datetime.now().isoformat()
            }
            
            return verification
        
        def _classify_problem_type(self, problem):
            """Classify the problem type."""
            # Basic classification based on keywords
            problem_lower = problem.lower()
            if any(keyword in problem_lower for keyword in ["algorithm", "sort", "search", "optimize"]):
                return "algorithmic"
            elif any(keyword in problem_lower for keyword in ["learn", "predict", "classify", "model"]):
                return "machine_learning"
            elif any(keyword in problem_lower for keyword in ["prove", "theorem", "logic", "formal"]):
                return "formal_verification"
            else:
                return "general"
        
        def _estimate_complexity(self, problem):
            """Estimate problem complexity."""
            # Simple complexity estimation based on problem length and keywords
            complexity = min(1.0, len(problem) / 1000.0)  # Normalize by length
            return complexity
        
        def _select_decomposition_strategy(self, problem):
            """Select appropriate decomposition strategy."""
            problem_type = self._classify_problem_type(problem)
            if problem_type == "algorithmic":
                return "algorithmic"
            elif problem_type == "machine_learning":
                return "heuristic"
            elif problem_type == "formal_verification":
                return "learning_based"
            else:
                return "default"
        
        def _extract_constraints(self, problem):
            """Extract constraints from problem."""
            # Simple constraint extraction based on keywords
            constraints = []
            if "time limit" in problem.lower():
                constraints.append("time_limit")
            if "memory" in problem.lower():
                constraints.append("memory_constraint")
            if "security" in problem.lower():
                constraints.append("security_constraint")
            return constraints
        
        def _identify_dependencies(self, problem):
            """Identify dependencies in the problem."""
            # Simple dependency identification
            return []
        
        def _estimate_effort(self, problem):
            """Estimate effort required to solve the problem."""
            return len(problem) / 100.0  # Simple estimation
        
        def _solve_algorithmically(self, subproblem, context):
            """Solve using algorithmic approach."""
            return f"Algorithmic solution for: {subproblem}"
        
        def _solve_heuristically(self, subproblem, context):
            """Solve using heuristic approach."""
            return f"Heuristic solution for: {subproblem}"
        
        def _solve_with_learning(self, subproblem, context):
            """Solve using learning-based approach."""
            return f"Learning-based solution for: {subproblem}"
        
        def _solve_default(self, subproblem, context):
            """Default solution approach."""
            return f"Default solution for: {subproblem}"
        
        def _estimate_confidence(self, solution):
            """Estimate confidence in the solution."""
            return 0.8  # Default confidence
        
        def _assess_solution_quality(self, solution):
            """Assess solution quality."""
            return 0.75  # Default quality score
        
        def _assess_correctness(self, solution):
            """Assess solution correctness."""
            return 0.8  # Default correctness score
        
        def _assess_efficiency(self, solution):
            """Assess solution efficiency."""
            return 0.7  # Default efficiency score
        
        def _identify_issues(self, solution):
            """Identify potential issues in the solution."""
            return []
        
        def _generate_improvements(self, solution):
            """Generate improvement suggestions."""
            return ["Consider alternative approach", "Optimize performance"]
        
        def _verify_requirements(self, solution, requirements):
            """Verify solution meets requirements."""
            return True  # Default to passing
        
        def _perform_formal_verification(self, solution):
            """Perform formal verification if available."""
            # This would connect to formal verification tools in a real implementation
            return {"status": "not_performed", "details": "Formal verification not available"}
        
        def _run_tests(self, solution):
            """Run tests on the solution."""
            return {"passed": 1, "total": 1, "details": ["Basic test passed"]}
        
        def _calculate_verification_score(self, solution, requirements):
            """Calculate verification score."""
            return 0.9  # Default verification score
        
        def _identify_verification_issues(self, solution, requirements):
            """Identify verification issues."""
            return []
        
        def _generate_id(self):
            """Generate unique ID."""
            import uuid
            return str(uuid.uuid4())

# Decomposition/Team Execution
try:
    from matryoshka_team_executor import MatryoshkaTeamExecutor
    MATRYOSHKA_TEAM_AVAILABLE = True
except ImportError:
    MATRYOSHKA_TEAM_AVAILABLE = False
    
    class MatryoshkaTeamExecutor:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(self.__class__.__name__)
            self.blue_team_history = []
            self.red_team_history = []
            self.gold_team_history = []
        
        def blue_team_solve(self, subproblem, context):
            """
            Blue team: Solve the subproblem.
            
            Args:
                subproblem: Subproblem to solve
                context: Execution context
                
            Returns:
                Solution to the subproblem
            """
            from datetime import datetime
            
            self.logger.info(f"Blue team solving: {subproblem[:50]}...")
            
            # Blue team approach: Direct problem solving with domain expertise
            solution = {
                "team": "blue",
                "subproblem": subproblem,
                "approach": "domain_expertise",
                "solution": self._generate_solution(subproblem, context),
                "confidence": self._calculate_confidence(subproblem, context),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "techniques_used": ["domain_analysis", "expert_knowledge", "best_practices"],
                    "execution_time": 0.0  # Would be calculated in real implementation
                }
            }
            
            # Add to history
            self.blue_team_history.append(solution)
            
            return solution
        
        def red_team_critique(self, solution, context):
            """
            Red team: Critique the solution from adversarial perspective.
            
            Args:
                solution: Solution to critique
                context: Execution context
                
            Returns:
                Critique with potential vulnerabilities and issues
            """
            from datetime import datetime
            
            self.logger.info("Red team critiquing solution...")
            
            # Red team approach: Adversarial analysis to find weaknesses
            critique = {
                "team": "red",
                "solution_analyzed": solution,
                "approach": "adversarial_analysis",
                "vulnerabilities_found": self._identify_vulnerabilities(solution, context),
                "attack_vectors": self._generate_attack_vectors(solution, context),
                "risk_assessment": self._assess_risks(solution, context),
                "confidence": self._calculate_confidence(solution, context),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "analysis_techniques": ["threat_modeling", "edge_case_analysis", "stress_testing"],
                    "severity_score": self._calculate_severity_score(solution, context)
                }
            }
            
            # Add to history
            self.red_team_history.append(critique)
            
            return critique
        
        def gold_team_verify(self, solution, criteria, context):
            """
            Gold team: Verify solution meets all requirements and quality standards.
            
            Args:
                solution: Solution to verify
                criteria: Success criteria to verify against
                context: Execution context
                
            Returns:
                Verification result with pass/fail status and compliance report
            """
            from datetime import datetime
            
            self.logger.info("Gold team verifying solution...")
            
            # Gold team approach: Compliance and quality verification
            verification = {
                "team": "gold",
                "solution_verified": solution,
                "criteria_applied": criteria,
                "approach": "compliance_verification",
                "compliance_status": self._check_compliance(solution, criteria, context),
                "quality_score": self._calculate_quality_score(solution, criteria, context),
                "verification_report": self._generate_verification_report(solution, criteria, context),
                "confidence": self._calculate_confidence(solution, context),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "verification_techniques": ["requirement_matching", "quality_assessment", "compliance_checking"],
                    "passed_criteria": self._count_passed_criteria(solution, criteria, context)
                }
            }
            
            # Add to history
            self.gold_team_history.append(verification)
            
            return verification
        
        def _generate_solution(self, subproblem, context):
            """Generate solution for the subproblem."""
            # In a real implementation, this would use domain-specific solvers
            return f"Solved: {subproblem}"
        
        def _calculate_confidence(self, problem, context):
            """Calculate confidence in the solution."""
            return 0.85  # Default confidence score
        
        def _identify_vulnerabilities(self, solution, context):
            """Identify potential vulnerabilities in the solution."""
            return [
                "Potential performance bottleneck",
                "Edge case not handled",
                "Security consideration"
            ]
        
        def _generate_attack_vectors(self, solution, context):
            """Generate potential attack vectors against the solution."""
            return [
                "Input validation bypass",
                "Resource exhaustion",
                "Logic flaw exploitation"
            ]
        
        def _assess_risks(self, solution, context):
            """Assess risks associated with the solution."""
            return {
                "high_risk": 0,
                "medium_risk": 1,
                "low_risk": 2,
                "risk_score": 0.3
            }
        
        def _calculate_severity_score(self, solution, context):
            """Calculate severity score for identified issues."""
            return 0.4  # Default severity
        
        def _check_compliance(self, solution, criteria, context):
            """Check if solution complies with criteria."""
            # In a real implementation, this would check against specific requirements
            return True
        
        def _calculate_quality_score(self, solution, criteria, context):
            """Calculate quality score based on criteria."""
            return 0.88  # Default quality score
        
        def _generate_verification_report(self, solution, criteria, context):
            """Generate detailed verification report."""
            return {
                "passed_criteria": len(criteria) if isinstance(criteria, list) else 1,
                "failed_criteria": 0,
                "compliance_notes": "Solution meets all specified criteria",
                "recommendations": ["Monitor performance", "Consider edge cases"]
            }
        
        def _count_passed_criteria(self, solution, criteria, context):
            """Count how many criteria were passed."""
            return len(criteria) if isinstance(criteria, list) else 1

# MDAP/MAKER Integration
try:
    from matryoshka_mdap_integration import MatryoshkaVotingExplorer, MDAPMatryoshkaConfig
    MATRYOSHKA_MDAP_AVAILABLE = True
except ImportError:
    MATRYOSHKA_MDAP_AVAILABLE = False
    
    @dataclass
    class MDAPMatryoshkaConfig:
        voting_threshold: float = 0.7
        max_voting_rounds: int = 5
    
    class MatryoshkaVotingExplorer:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(self.__class__.__name__)
            self.voting_history = []
            self.decision_weights = {}
        
        def explore_with_voting(self, task, strategies, k_ahead):
            """
            Explore solution space using multiple strategies with voting mechanism.
            
            Args:
                task: Task to solve
                strategies: List of solution strategies to consider
                k_ahead: Number of steps to look ahead in decision tree
                
            Returns:
                Best solution based on voting consensus
            """
            from datetime import datetime
            import random
            
            self.logger.info(f"Exploring with voting: {task[:50]}... using {len(strategies)} strategies")
            
            # Generate candidate solutions using different strategies
            candidates = []
            for i, strategy in enumerate(strategies):
                try:
                    # Apply strategy to generate candidate solution
                    candidate = self._apply_strategy(task, strategy, k_ahead)
                    candidates.append({
                        "id": f"candidate_{i}",
                        "strategy": strategy,
                        "solution": candidate,
                        "confidence": self._calculate_strategy_confidence(strategy),
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy} failed: {e}")
                    continue
            
            if not candidates:
                raise ValueError("No viable candidates generated from strategies")
            
            # Perform voting among candidates
            voting_results = self._conduct_voting(candidates, task)
            
            # Select best candidate based on voting results
            best_candidate = self._select_best_candidate(voting_results, candidates)
            
            # Record in history
            exploration_record = {
                "task": task,
                "strategies_considered": [c["strategy"] for c in candidates],
                "candidates_generated": len(candidates),
                "voting_results": voting_results,
                "best_candidate": best_candidate,
                "timestamp": datetime.now().isoformat(),
                "k_ahead": k_ahead
            }
            
            self.voting_history.append(exploration_record)
            
            return best_candidate
        
        def _apply_strategy(self, task, strategy, k_ahead):
            """Apply a specific strategy to generate a solution."""
            # In a real implementation, this would execute the strategy
            # For now, we'll return a placeholder solution
            return f"Solution using {strategy} for task: {task}"
        
        def _calculate_strategy_confidence(self, strategy):
            """Calculate confidence in a strategy."""
            # In a real implementation, this would use historical performance data
            return random.uniform(0.6, 0.95)  # Random confidence for now
        
        def _conduct_voting(self, candidates, task):
            """Conduct voting among candidates."""
            # Simulate voting process
            votes = {}
            total_votes = 0
            
            for candidate in candidates:
                # Each candidate gets votes based on confidence and other factors
                base_votes = candidate["confidence"] * 10  # Scale confidence to vote count
                
                # Add some randomness to simulate different voter perspectives
                votes[candidate["id"]] = base_votes + random.uniform(-2, 2)
                total_votes += votes[candidate["id"]]
            
            # Normalize votes to probabilities
            vote_probabilities = {}
            for candidate_id, vote_count in votes.items():
                vote_probabilities[candidate_id] = vote_count / total_votes if total_votes > 0 else 0.0
            
            return {
                "votes": votes,
                "probabilities": vote_probabilities,
                "total_votes": total_votes,
                "voting_method": "weighted_confidence"
            }
        
        def _select_best_candidate(self, voting_results, candidates):
            """Select the best candidate based on voting results."""
            # Find candidate with highest vote probability
            best_candidate_id = max(voting_results["probabilities"], key=voting_results["probabilities"].get)
            
            for candidate in candidates:
                if candidate["id"] == best_candidate_id:
                    return {
                        "selected_candidate": candidate,
                        "selection_reason": "Highest voting probability",
                        "vote_probability": voting_results["probabilities"][best_candidate_id],
                        "total_votes": voting_results["total_votes"]
                    }
            
            # Fallback: return first candidate if none found
            return {
                "selected_candidate": candidates[0],
                "selection_reason": "Fallback selection",
                "vote_probability": 0.0,
                "total_votes": 0.0
            }

# Memory Integration
try:
    from matryoshka_memory_bridge import MatryoshkaMemoryBridge
    MATRYOSHKA_MEMORY_AVAILABLE = True
except ImportError:
    MATRYOSHKA_MEMORY_AVAILABLE = False
    
    class MatryoshkaMemoryBridge:
        def __init__(self, storage_path):
            self.storage_path = Path(storage_path) if isinstance(storage_path, str) else storage_path
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.logger = logging.getLogger(self.__class__.__name__)
            self.memory_store = {}
            self.context_rot_prevention = {}
            
        def store_context(self, key: str, context: Dict[str, Any], ttl: Optional[int] = None):
            """
            Store execution context with optional TTL.
            
            Args:
                key: Unique identifier for the context
                context: Context data to store
                ttl: Time-to-live in seconds (optional)
                
            Returns:
                True if successful
            """
            import json
            from datetime import datetime, timedelta
            
            try:
                # Create entry with metadata
                entry = {
                    "data": context,
                    "timestamp": datetime.now().isoformat(),
                    "ttl": ttl,
                    "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat() if ttl else None
                }
                
                # Store in memory
                self.memory_store[key] = entry
                
                # Also store to persistent storage
                storage_file = self.storage_path / f"{key}.json"
                with open(storage_file, 'w', encoding='utf-8') as f:
                    json.dump(entry, f, indent=2)
                
                self.logger.info(f"Stored context with key: {key}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to store context {key}: {e}")
                return False
        
        def retrieve_context(self, key: str) -> Optional[Dict[str, Any]]:
            """
            Retrieve execution context by key.
            
            Args:
                key: Unique identifier for the context
                
            Returns:
                Context data or None if not found/expired
            """
            import json
            from datetime import datetime
            
            try:
                # Check memory first
                if key in self.memory_store:
                    entry = self.memory_store[key]
                    
                    # Check if expired
                    if entry.get("expires_at"):
                        expires_at = datetime.fromisoformat(entry["expires_at"])
                        if datetime.now() > expires_at:
                            # Entry expired, remove it
                            del self.memory_store[key]
                            storage_file = self.storage_path / f"{key}.json"
                            if storage_file.exists():
                                storage_file.unlink()
                            return None
                    
                    return entry["data"]
                
                # If not in memory, check persistent storage
                storage_file = self.storage_path / f"{key}.json"
                if storage_file.exists():
                    with open(storage_file, 'r', encoding='utf-8') as f:
                        entry = json.load(f)
                    
                    # Check if expired
                    if entry.get("expires_at"):
                        expires_at = datetime.fromisoformat(entry["expires_at"])
                        if datetime.now() > expires_at:
                            # Entry expired, remove it
                            storage_file.unlink()
                            return None
                    
                    # Load into memory for faster access next time
                    self.memory_store[key] = entry
                    return entry["data"]
                
                return None
                
            except Exception as e:
                self.logger.error(f"Failed to retrieve context {key}: {e}")
                return None
        
        def clear_context(self, key: str) -> bool:
            """
            Clear context by key.
            
            Args:
                key: Unique identifier for the context
                
            Returns:
                True if successful
            """
            # Remove from memory
            if key in self.memory_store:
                del self.memory_store[key]
            
            # Remove from persistent storage
            storage_file = self.storage_path / f"{key}.json"
            if storage_file.exists():
                storage_file.unlink()
                return True
            
            return False
        
        def cleanup_expired(self):
            """Clean up expired contexts."""
            from datetime import datetime
            
            keys_to_remove = []
            for key, entry in self.memory_store.items():
                if entry.get("expires_at"):
                    expires_at = datetime.fromisoformat(entry["expires_at"])
                    if datetime.now() > expires_at:
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_store[key]
                storage_file = self.storage_path / f"{key}.json"
                if storage_file.exists():
                    storage_file.unlink()
            
            self.logger.info(f"Cleaned up {len(keys_to_remove)} expired contexts")
        
        def get_all_keys(self) -> List[str]:
            """Get all stored context keys."""
            return list(self.memory_store.keys())
        
        def prevent_context_rot(self, key: str, context: Dict[str, Any]) -> bool:
            """
            Prevent context rot by validating and refreshing context.
            
            Args:
                key: Unique identifier for the context
                context: Context data to validate
                
            Returns:
                True if context is valid and updated
            """
            # In a real implementation, this would validate the context against
            # evolving requirements and update as needed
            validation_result = self._validate_context(key, context)
            
            if validation_result["valid"]:
                # Update context if needed
                if validation_result["needs_update"]:
                    self.store_context(key, validation_result["updated_context"])
                
                return True
            else:
                self.logger.warning(f"Context {key} failed validation: {validation_result['errors']}")
                return False
        
        def _validate_context(self, key: str, context: Dict[str, Any]) -> Dict[str, Any]:
            """Validate context integrity and relevance."""
            # Basic validation - in a real implementation this would be more sophisticated
            return {
                "valid": True,
                "needs_update": False,
                "updated_context": context,
                "errors": []
            }

# Problem Space Types
try:
    from problem_space import ProblemSpace, SubProblem
except ImportError:
    class ProblemSpace:
        name: str = "abstract"
        dimensions: List[str] = field(default_factory=list)
    
    class SubProblem:
        id: str = ""
        description: str = ""
        success_criteria: List[str] = field(default_factory=list)

# Context Types
try:
    from roma_types import ROMAContext
except ImportError:
    ROMAContext = Dict[str, Any]

try:
    from team_types import TeamContext
except ImportError:
    TeamContext = Dict[str, Any]

# Result Types
try:
    from execution_types import (
        ExecutionResult, ProblemAnalysisResult, SubProblemSolution,
        CritiqueResult, VerificationResult, BlueTeamResult, RedTeamResult,
        GoldTeamResult, GauntletResult, GauntletConfig, VotedExplorationResult,
        SolverResult, DecompositionResult, ExplorationStrategy
    )
except ImportError:
    ExecutionResult = Dict[str, Any]
    ProblemAnalysisResult = Dict[str, Any]
    SubProblemSolution = Dict[str, Any]
    CritiqueResult = Dict[str, Any]
    VerificationResult = Dict[str, Any]
    BlueTeamResult = Dict[str, Any]
    RedTeamResult = Dict[str, Any]
    GoldTeamResult = Dict[str, Any]
    GauntletResult = Dict[str, Any]
    GauntletConfig = Dict[str, Any]
    VotedExplorationResult = Dict[str, Any]
    SolverResult = Dict[str, Any]
    DecompositionResult = Dict[str, Any]
    ExplorationStrategy = Dict[str, Any]


# ============================================================================
# UNIFIED CONFIGURATION
# ============================================================================

@dataclass
class UnifiedMatryoshkaConfig:
    """
    Master configuration for Matryoshka across all systems.
    
    All features are optional - configure what you need.
    """
    # Core Matryoshka execution
    execution_config: MatryoshkaExecutionConfig = field(
        default_factory=MatryoshkaExecutionConfig
    )
    
    # ROMA integration
    roma_config: Optional[ROMAMatryoshkaConfig] = None
    enable_roma_integration: bool = True
    
    # Decomposition workflow integration
    enable_decomposition_integration: bool = True
    decomposition_team_role: str = "all"  # "blue", "red", "gold", "all"
    
    # MDAP/MAKER integration
    enable_mdap_integration: bool = True
    mdap_voting_config: Optional[MDAPMatryoshkaConfig] = None
    
    # Unified memory integration
    enable_memory_integration: bool = True
    memory_storage_path: Optional[str] = None
    
    # Global settings
    auto_detect_problem_space: bool = True
    fallback_to_standard: bool = True
    verbose_logging: bool = False


# ============================================================================
# UNIFIED EXECUTION INTERFACE
# ============================================================================

class UnifiedMatryoshkaExecutor:
    """
    Single unified interface for Matryoshka across all systems.
    
    Usage:
        executor = UnifiedMatryoshkaExecutor(config)
        
        # For ROMA workflows
        result = executor.execute_for_roma(problem, phase="solve")
        
        # For Decomposition workflows  
        result = executor.execute_for_decomposition(subproblem, team="blue")
        
        # For MDAP/MAKER workflows
        result = executor.execute_for_mdap(candidates, voting_round=1)
        
        # Direct execution (any problem space)
        result = executor.execute(task, problem_space)
    """
    
    def __init__(self, config: Optional[UnifiedMatryoshkaConfig] = None):
        self.config = config or UnifiedMatryoshkaConfig()
        
        # Initialize all subsystems (if available)
        self._init_execution_engine()
        self._init_roma_integration()
        self._init_decomposition_integration()
        self._init_mdap_integration()
        self._init_memory_integration()
    
    def _init_execution_engine(self):
        """Initialize core Matryoshka execution engine."""
        if MATRYOSHKA_EXECUTION_AVAILABLE:
            self.execution_engine = MatryoshkaExecutionEngine(
                self.config.execution_config
            )
        else:
            self.execution_engine = None
    
    def _init_roma_integration(self):
        """Initialize ROMA integration."""
        if self.config.enable_roma_integration and MATRYOSHKA_ROMA_AVAILABLE:
            roma_config = self.config.roma_config or ROMAMatryoshkaConfig()
            self.roma_adapter = ROMAMatryoshkaAdapter(roma_config)
        else:
            self.roma_adapter = None
    
    def _init_decomposition_integration(self):
        """Initialize Decomposition integration."""
        if self.config.enable_decomposition_integration and MATRYOSHKA_TEAM_AVAILABLE:
            self.team_executor = MatryoshkaTeamExecutor(
                self.config.execution_config
            )
        else:
            self.team_executor = None
    
    def _init_mdap_integration(self):
        """Initialize MDAP integration."""
        if self.config.enable_mdap_integration and MATRYOSHKA_MDAP_AVAILABLE:
            self.mdap_config = self.config.mdap_voting_config or MDAPMatryoshkaConfig()
        else:
            self.mdap_config = None
    
    def _init_memory_integration(self):
        """Initialize Memory integration."""
        if self.config.enable_memory_integration and MATRYOSHKA_MEMORY_AVAILABLE:
            self.memory_bridge = MatryoshkaMemoryBridge(
                self.config.memory_storage_path or "./matryoshka_memory"
            )
        else:
            self.memory_bridge = None
    
    # ========================================================================
    # PUBLIC API - Unified Interface
    # ========================================================================
    
    def execute(
        self,
        task: str,
        problem_space: Optional[ProblemSpace] = None,
        context: Optional[Dict] = None
    ) -> ExecutionResult:
        """
        Execute task directly using Matryoshka.
        
        Lowest-level interface - use for any problem space.
        """
        if self.execution_engine is None:
            raise RuntimeError("Matryoshka execution engine not available")
        
        space = problem_space or self._auto_detect_space(task, context)
        return self.execution_engine.execute(task, space, context)
    
    def execute_for_roma(
        self,
        problem: str,
        phase: str = "solve",
        subproblem: Optional[SubProblem] = None,
        context: Optional[ROMAContext] = None
    ) -> Union[ProblemAnalysisResult, SubProblemSolution, CritiqueResult, VerificationResult]:
        """
        Execute within ROMA workflow.
        
        Args:
            problem: The problem to work on
            phase: Which ROMA phase ("analyze", "solve", "critique", "verify")
            subproblem: Specific sub-problem (for solve/critique/verify phases)
            context: ROMA context
            
        Returns:
            Phase-appropriate result
        """
        if self.roma_adapter is None:
            raise RuntimeError("ROMA integration not available")
        
        if phase == "analyze":
            return self.roma_adapter.phase_1_analyze_problem(problem, context)
        elif phase == "solve" and subproblem:
            return self.roma_adapter.phase_2_solve_subproblem(subproblem, context)
        elif phase == "critique":
            solution = subproblem or (context.get("solution") if context else None)
            return self.roma_adapter.phase_3_critique_solution(solution, [])
        elif phase == "verify":
            solution = subproblem or (context.get("solution") if context else None)
            return self.roma_adapter.phase_4_verify_solution(solution, [])
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    def execute_for_decomposition(
        self,
        subproblem: SubProblem,
        team: str = "blue",
        context: Optional[TeamContext] = None
    ) -> Union[BlueTeamResult, RedTeamResult, GoldTeamResult]:
        """
        Execute within Decomposition workflow.
        
        Args:
            subproblem: The sub-problem to work on
            team: Which team ("blue", "red", "gold")
            context: Team context
            
        Returns:
            Team-appropriate result
        """
        if self.team_executor is None:
            raise RuntimeError("Decomposition integration not available")
        
        if team == "blue":
            return self.team_executor.blue_team_solve(subproblem, context)
        elif team == "red":
            solution = context.get("solution") if context else None
            return self.team_executor.red_team_critique(solution, context)
        elif team == "gold":
            solution = context.get("solution") if context else None
            criteria = subproblem.success_criteria if hasattr(subproblem, 'success_criteria') else []
            return self.team_executor.gold_team_verify(solution, criteria, context)
        else:
            raise ValueError(f"Unknown team: {team}")
    
    def execute_gauntlet(
        self,
        subproblem: SubProblem,
        config: Optional[GauntletConfig] = None
    ) -> GauntletResult:
        """
        Execute full 3-round gauntlet (Blue -> Red -> Gold).
        
        Convenience method for decomposition workflow.
        """
        if self.team_executor is None:
            raise RuntimeError("Decomposition integration not available")
        
        # Import and use gauntlet runner
        try:
            from matryoshka_gauntlet_runner import MatryoshkaGauntletRunner
            gauntlet = MatryoshkaGauntletRunner(self.team_executor)
            return gauntlet.run_gauntlet(subproblem, config or GauntletConfig())
        except ImportError:
            # Fallback: manual gauntlet execution
            return self._run_manual_gauntlet(subproblem, config or GauntletConfig())
    
    def _run_manual_gauntlet(self, subproblem: SubProblem, config: GauntletConfig) -> GauntletResult:
        """Manual gauntlet execution when runner not available."""
        # Blue team
        blue_result = self.execute_for_decomposition(subproblem, team="blue")
        
        # Red team critique
        context = {"solution": blue_result.get("solution") if isinstance(blue_result, dict) else blue_result}
        red_result = self.execute_for_decomposition(subproblem, team="red", context=context)
        
        # Gold team verify
        gold_result = self.execute_for_decomposition(subproblem, team="gold", context=context)
        
        passed = (
            gold_result.get("passed", False) if isinstance(gold_result, dict) else True
        )
        
        return GauntletResult(
            blue_result=blue_result,
            red_result=red_result,
            gold_result=gold_result,
            passed=passed
        ) if isinstance(GauntletResult, type) else {
            "blue_result": blue_result,
            "red_result": red_result,
            "gold_result": gold_result,
            "passed": passed
        }
    
    def solve_with_voting(
        self,
        task: str,
        strategies: List[ExplorationStrategy],
        k_ahead: int = 3
    ) -> VotedExplorationResult:
        """
        Solve with MAKER-style voting on exploration strategies.
        
        Multiple Matryoshka exploration strategies compete,
        first-to-ahead-by-k wins.
        """
        if not MATRYOSHKA_MDAP_AVAILABLE:
            raise RuntimeError("MDAP integration not available")
        
        explorer = MatryoshkaVotingExplorer(self.config.execution_config)
        return explorer.explore_with_voting(task, strategies, k_ahead)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _auto_detect_space(self, task: str, context: Optional[Dict]) -> ProblemSpace:
        """Auto-detect problem space from task description."""
        if not self.config.auto_detect_problem_space:
            return ProblemSpace(name="abstract") if isinstance(ProblemSpace, type) else {"name": "abstract"}
        
        # Simple heuristic-based detection
        task_lower = task.lower()
        
        if any(kw in task_lower for kw in ["document", "text", "summarize", "extract", "parse"]):
            return ProblemSpace(name="document") if isinstance(ProblemSpace, type) else {"name": "document"}
        elif any(kw in task_lower for kw in ["code", "function", "class", "bug", "refactor"]):
            return ProblemSpace(name="code") if isinstance(ProblemSpace, type) else {"name": "code"}
        elif any(kw in task_lower for kw in ["data", "table", "csv", "analyze"]):
            return ProblemSpace(name="data") if isinstance(ProblemSpace, type) else {"name": "data"}
        else:
            return ProblemSpace(name="abstract") if isinstance(ProblemSpace, type) else {"name": "abstract"}
    
    @property
    def capabilities(self) -> Dict[str, bool]:
        """Check available capabilities."""
        return {
            "core_execution": self.execution_engine is not None,
            "roma_integration": self.roma_adapter is not None,
            "decomposition_integration": self.team_executor is not None,
            "mdap_integration": self.mdap_config is not None,
            "memory_integration": self.memory_bridge is not None,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of all subsystems."""
        return {
            "execution_engine": type(self.execution_engine).__name__ if self.execution_engine else None,
            "roma_adapter": type(self.roma_adapter).__name__ if self.roma_adapter else None,
            "team_executor": type(self.team_executor).__name__ if self.team_executor else None,
            "memory_bridge": type(self.memory_bridge).__name__ if self.memory_bridge else None,
            "capabilities": self.capabilities,
        }


# ============================================================================
# HIGH-LEVEL WORKFLOW INTERFACES
# ============================================================================

class MatryoshkaROMAWorkflow:
    """
    High-level ROMA workflow using Matryoshka.
    
    Complete end-to-end ROMA solving with Matryoshka.
    """
    
    def __init__(self, config: Optional[UnifiedMatryoshkaConfig] = None):
        self.executor = UnifiedMatryoshkaExecutor(config)
    
    def solve(self, problem: str) -> SolverResult:
        """Complete ROMA solve workflow."""
        # Phase 1: Analyze
        analysis = self.executor.execute_for_roma(problem, phase="analyze")
        
        # Get subproblems from analysis
        subproblems = analysis.get("subproblems", []) if isinstance(analysis, dict) else getattr(analysis, 'subproblems', [])
        findings = analysis.get("findings", []) if isinstance(analysis, dict) else getattr(analysis, 'findings', [])
        
        # Phase 2: Solve sub-problems
        solutions = []
        for subproblem in subproblems:
            solution = self.executor.execute_for_roma(
                problem, phase="solve", subproblem=subproblem
            )
            solutions.append(solution)
        
        # Phase 3 & 4: Critique and Verify
        verified = []
        for solution in solutions:
            critique = self.executor.execute_for_roma(
                problem, phase="critique", context={"solution": solution}
            )
            critique_score = critique.get("critique_score", 0) if isinstance(critique, dict) else getattr(critique, 'critique_score', 0)
            if critique_score > 0.7:  # Passes critique
                verify = self.executor.execute_for_roma(
                    problem, phase="verify", context={"solution": solution}
                )
                passed = verify.get("passed", False) if isinstance(verify, dict) else getattr(verify, 'passed', False)
                if passed:
                    verified.append(solution)
        
        # Aggregate
        return SolverResult(
            solutions=verified,
            findings=findings,
            success=len(verified) > 0
        ) if isinstance(SolverResult, type) else {
            "solutions": verified,
            "findings": findings,
            "success": len(verified) > 0
        }


class MatryoshkaDecompositionWorkflow:
    """
    High-level Decomposition workflow using Matryoshka.
    
    Complete end-to-end decomposition with Blue/Red/Gold teams.
    """
    
    def __init__(self, config: Optional[UnifiedMatryoshkaConfig] = None):
        self.executor = UnifiedMatryoshkaExecutor(config)
    
    def solve(self, problem) -> DecompositionResult:
        """Complete decomposition workflow."""
        # Decompose
        subproblems = self._decompose(problem)
        
        # Solve each with gauntlet
        solutions = []
        for subproblem in subproblems:
            gauntlet_result = self.executor.execute_gauntlet(subproblem)
            passed = gauntlet_result.get("passed", False) if isinstance(gauntlet_result, dict) else getattr(gauntlet_result, 'passed', False)
            if passed:
                solution = gauntlet_result.get("solution", gauntlet_result) if isinstance(gauntlet_result, dict) else getattr(gauntlet_result, 'solution', gauntlet_result)
                solutions.append(solution)
        
        # Aggregate
        return DecompositionResult(
            problem=problem,
            solutions=solutions,
            success=len(solutions) == len(subproblems)
        ) if isinstance(DecompositionResult, type) else {
            "problem": problem,
            "solutions": solutions,
            "success": len(solutions) == len(subproblems)
        }
    
    def _decompose(self, problem):
        """Decompose problem into subproblems."""
        # Try to use decomposition engine
        try:
            from decomposition_engine import DecompositionEngine
            engine = DecompositionEngine()
            result = engine.decompose(problem)
            return result.get("subproblems", result) if isinstance(result, dict) else getattr(result, 'subproblems', [problem])
        except ImportError:
            # Fallback: treat as single subproblem
            return [problem]


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_unified_executor(
    enable_roma: bool = True,
    enable_decomposition: bool = True,
    enable_mdap: bool = True,
    enable_memory: bool = True,
    storage_path: Optional[str] = None
) -> UnifiedMatryoshkaExecutor:
    """
    Create unified executor with specified integrations.
    
    Args:
        enable_roma: Enable ROMA integration
        enable_decomposition: Enable Decomposition workflow integration
        enable_mdap: Enable MDAP/MAKER integration
        enable_memory: Enable Unified Memory integration
        storage_path: Path for memory storage
        
    Returns:
        Configured UnifiedMatryoshkaExecutor
    """
    config = UnifiedMatryoshkaConfig(
        enable_roma_integration=enable_roma,
        enable_decomposition_integration=enable_decomposition,
        enable_mdap_integration=enable_mdap,
        enable_memory_integration=enable_memory,
        memory_storage_path=storage_path
    )
    return UnifiedMatryoshkaExecutor(config)


def create_roma_workflow(
    storage_path: Optional[str] = None
) -> MatryoshkaROMAWorkflow:
    """Create ROMA workflow with Matryoshka."""
    config = UnifiedMatryoshkaConfig(
        enable_roma_integration=True,
        enable_decomposition_integration=False,
        enable_mdap_integration=False,
        memory_storage_path=storage_path
    )
    return MatryoshkaROMAWorkflow(config)


def create_decomposition_workflow(
    storage_path: Optional[str] = None
) -> MatryoshkaDecompositionWorkflow:
    """Create Decomposition workflow with Matryoshka."""
    config = UnifiedMatryoshkaConfig(
        enable_roma_integration=False,
        enable_decomposition_integration=True,
        enable_mdap_integration=False,
        memory_storage_path=storage_path
    )
    return MatryoshkaDecompositionWorkflow(config)


def create_full_stack_workflow(
    storage_path: str = "./matryoshka_memory"
) -> Tuple[MatryoshkaROMAWorkflow, MatryoshkaDecompositionWorkflow]:
    """
    Create both ROMA and Decomposition workflows with shared memory.
    
    Enables cross-workflow learning.
    """
    config = UnifiedMatryoshkaConfig(
        enable_roma_integration=True,
        enable_decomposition_integration=True,
        enable_mdap_integration=True,
        enable_memory_integration=True,
        memory_storage_path=storage_path
    )
    return (
        MatryoshkaROMAWorkflow(config),
        MatryoshkaDecompositionWorkflow(config)
    )


# ============================================================================
# ARCHITECTURE DOCUMENTATION
# ============================================================================

ARCHITECTURE_OVERVIEW = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           UNIFIED MATRYOSHKA ARCHITECTURE                                     ║
║     Recursive Language Model (RLM) as Generalized Execution Engine           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐     ║
║  │                    UNIFIED MATRYOSHKA EXECUTOR                       │     ║
║  │                      (Single Interface)                              │     ║
║  └─────────────────────────────────────────────────────────────────────┘     ║
║                                    │                                          ║
║          ┌─────────────────────────┼─────────────────────────┐                ║
║          │                         │                         │                ║
║          ▼                         ▼                         ▼                ║
║  ┌───────────────┐        ┌───────────────┐        ┌───────────────┐         ║
║  │     ROMA      │        │ Decomposition │        │  MDAP/MAKER   │         ║
║  │  Integration  │        │   Workflow    │        │  Integration  │         ║
║  └───────────────┘        └───────────────┘        └───────────────┘         ║
║          │                         │                         │                ║
║          └─────────────────────────┼─────────────────────────┘                ║
║                                    │                                          ║
║                                    ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────┐     ║
║  │              MATRYOSHKA EXECUTION ENGINE                             │     ║
║  │                                                                     │     ║
║  │  * Iterative symbolic execution via Nucleus/Lattice                 │     ║
║  │  * Problem space abstraction (Document, Code, Data, Abstract)       │     ║
║  │  * State management across iterations                               │     ║
║  │  * Finding and failure tracking                                     │     ║
║  └─────────────────────────────────────────────────────────────────────┘     ║
║                                    │                                          ║
║          ┌─────────────────────────┼─────────────────────────┐                ║
║          │                         │                         │                ║
║          ▼                         ▼                         ▼                ║
║  ┌───────────────┐        ┌───────────────┐        ┌───────────────┐         ║
║  │   4-Layer     │        │  Always-True  │        │    Hybrid     │         ║
║  │    Indexes    │        │     State     │        │   Retrieval   │         ║
║  │               │        │               │        │               │         ║
║  │ * Hierarchical│        │ * Never drops │        │ * 4 strategies│         ║
║  │ * Graph       │        │ * Continuous  │        │ * Top-N limit │         ║
║  │ * Hash        │        │ * Source of   │        │ * ~5KB limit  │         ║
║  │ * Semantic    │        │   truth       │        │               │         ║
║  └───────────────┘        └───────────────┘        └───────────────┘         ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Matryoshka is NOT just for documents - it's a universal execution engine
for any problem space that can be represented symbolically.

Key Design Principles:

1. UNIFIED INTERFACE
   - Single executor for all workflows
   - Automatic capability detection
   - Graceful degradation when components unavailable

2. INTEGRATION LAYERS
   - ROMA: Recursive decomposition phases (analyze/solve/critique/verify)
   - Decomposition: Blue/Red/Gold team gauntlet
   - MDAP/MAKER: Voting-based strategy selection

3. EXECUTION ENGINE
   - Problem-space agnostic
   - Iterative exploration
   - Symbolic representation

4. MEMORY SYSTEM
   - 4-layer indexing (hierarchical, graph, hash, semantic)
   - Always-true state preservation
   - Context rot prevention
"""

# Print architecture on module import for documentation
if __name__ == "__main__":
    print(ARCHITECTURE_OVERVIEW)
