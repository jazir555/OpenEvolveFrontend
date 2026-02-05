"""
CrewAI Integration for Adaptive MDAP.

Properly integrates with the existing CrewAI infrastructure in the project,
using the actual CrewAI Agent, Task, Crew, and Process classes.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from adaptive_mdap.core.types import SolveStrategy
from adaptive_mdap.utils.logger import get_logger

logger = get_logger("integrations.crewai")

# Import from existing project CrewAI integration
try:
    from crewai import Agent, Task, Crew, Process
    from crewai_mdap_maker_engine import MAKERAgentFactory, MAKERConfig
    from crewai_mdap_integrator import MDAPConfig, RedFlagRules
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = None
    Task = None
    Crew = None
    Process = None
    MAKERAgentFactory = None
    MAKERConfig = None


@dataclass
class AdaptiveCrewConfig:
    """Configuration for adaptive CrewAI crews."""
    strategy: SolveStrategy
    n_agents: int
    k_ahead: int
    max_retries: int
    llm_config: Optional[Dict[str, Any]] = None


class CrewAIIntegration:
    """
    Integration with CrewAI using the project's existing CrewAI infrastructure.
    
    This creates actual CrewAI Crews with Agents for adaptive allocation:
    - Complexity assessment agents
    - Strategy selection crews
    - Execution monitoring through CrewAI tasks
    """
    
    def __init__(self, crewai_client: Optional[Any] = None):
        """
        Initialize CrewAI integration.
        
        Args:
            crewai_client: Optional CrewAI client from existing integration
        """
        self.client = crewai_client
        self._crews: Dict[str, Crew] = {}
        self._metrics: List[Dict[str, Any]] = []
        
        if not CREWAI_AVAILABLE:
            logger.warning("CrewAI not available, integration disabled")
        else:
            logger.info("Initialized CrewAIIntegration with actual CrewAI classes")
    
    def create_complexity_assessment_crew(
        self,
        subproblem_id: str,
        description: str,
        domain: str,
    ) -> Optional[Crew]:
        """
        Create a CrewAI crew to assess problem complexity.
        
        Args:
            subproblem_id: ID of the sub-problem
            description: Problem description
            domain: Problem domain
            
        Returns:
            CrewAI Crew for complexity assessment
        """
        if not CREWAI_AVAILABLE or not MAKERAgentFactory:
            return None
        
        # Create agents for complexity assessment
        assessor = Agent(
            role="Complexity Assessor",
            goal="Assess the complexity of a given problem",
            backstory="Expert at analyzing problem difficulty based on description, domain, and structure",
            verbose=True,
        )
        
        reviewer = Agent(
            role="Complexity Reviewer",
            goal="Review and validate complexity assessments",
            backstory="Expert at validating complexity scores and ensuring consistency",
            verbose=True,
        )
        
        # Create tasks
        assess_task = Task(
            description=f"Assess complexity of problem: {description}\nDomain: {domain}",
            expected_output="Complexity score (0.0-1.0) with reasoning",
            agent=assessor,
        )
        
        review_task = Task(
            description="Review the complexity assessment for accuracy",
            expected_output="Validated complexity score",
            agent=reviewer,
            context=[assess_task],
        )
        
        # Create crew
        crew = Crew(
            agents=[assessor, reviewer],
            tasks=[assess_task, review_task],
            process=Process.sequential,
            verbose=True,
        )
        
        self._crews[f"complexity_{subproblem_id}"] = crew
        
        logger.debug(f"Created complexity assessment crew for {subproblem_id}")
        return crew
    
    def create_strategy_selection_crew(
        self,
        subproblem_id: str,
        complexity_score: float,
        thresholds: List[float],
    ) -> Optional[Crew]:
        """
        Create a CrewAI crew to select the appropriate strategy.
        
        Args:
            subproblem_id: ID of the sub-problem
            complexity_score: Computed complexity score
            thresholds: Allocation thresholds
            
        Returns:
            CrewAI Crew for strategy selection
        """
        if not CREWAI_AVAILABLE:
            return None
        
        # Create strategy selector agent
        selector = Agent(
            role="Strategy Selector",
            goal="Select the optimal solving strategy based on complexity",
            backstory=f"Expert at matching problem complexity to resources. Thresholds: {thresholds}",
            verbose=True,
        )
        
        # Create task
        select_task = Task(
            description=(
                f"Select strategy for problem with complexity {complexity_score:.2f}\n"
                f"Thresholds: LOW < {thresholds[0]:.2f} < MEDIUM < {thresholds[1]:.2f} < HIGH\n"
                f"Options: DIRECT (1 agent), MDAP_LIGHT (3 agents), MAKER_FULL (5+ agents)"
            ),
            expected_output="Selected strategy with justification",
            agent=selector,
        )
        
        # Create crew
        crew = Crew(
            agents=[selector],
            tasks=[select_task],
            process=Process.sequential,
            verbose=True,
        )
        
        self._crews[f"strategy_{subproblem_id}"] = crew
        
        return crew
    
    def create_execution_crew(
        self,
        subproblem_id: str,
        strategy: SolveStrategy,
        config: AdaptiveCrewConfig,
    ) -> Optional[Crew]:
        """
        Create a CrewAI crew for executing the selected strategy.
        
        Args:
            subproblem_id: ID of the sub-problem
            strategy: Selected solving strategy
            config: Crew configuration
            
        Returns:
            CrewAI Crew for execution
        """
        if not CREWAI_AVAILABLE or not MAKERAgentFactory:
            return None
        
        agents = []
        
        # Create agents based on strategy
        if strategy == SolveStrategy.DIRECT:
            # Single solver agent
            solver = Agent(
                role="Direct Solver",
                goal="Solve the problem directly",
                backstory="Expert problem solver",
                verbose=True,
            )
            agents.append(solver)
            
        elif strategy == SolveStrategy.MDAP_LIGHT:
            # 3 agents with light voting
            for i in range(3):
                agent = Agent(
                    role=f"MDAP_Voter_{i+1}",
                    goal="Vote on the best solution",
                    backstory="Expert at evaluating solutions",
                    verbose=True,
                )
                agents.append(agent)
                
        elif strategy == SolveStrategy.MAKER_FULL:
            # 5+ agents with full voting
            for i in range(config.n_agents):
                agent = MAKERAgentFactory.create_voter_agent(name=f"MAKER_Voter_{i+1}")
                if agent:
                    agents.append(agent)
        
        # Create execution task
        task = Task(
            description=f"Execute strategy {strategy.value} for problem {subproblem_id}",
            expected_output="Solution to the problem",
            agent=agents[0] if agents else None,
        )
        
        # Create crew
        crew = Crew(
            agents=agents,
            tasks=[task],
            process=Process.hierarchical if len(agents) > 1 else Process.sequential,
            verbose=True,
        )
        
        self._crews[f"execution_{subproblem_id}"] = crew
        
        logger.info(f"Created execution crew for {subproblem_id} with {len(agents)} agents")
        return crew
    
    def log_allocation_decision(
        self,
        subproblem_id: str,
        complexity_score: float,
        strategy: str,
        n_agents: int,
    ) -> None:
        """
        Log an allocation decision via CrewAI task.
        
        Args:
            subproblem_id: ID of the sub-problem
            complexity_score: Complexity score
            strategy: Allocated strategy
            n_agents: Number of agents
        """
        logger.info(
            f"Allocation for {subproblem_id}: strategy={strategy}, "
            f"complexity={complexity_score:.3f}, agents={n_agents}"
        )
        
        # Record metric
        self._metrics.append({
            "type": "allocation",
            "subproblem_id": subproblem_id,
            "complexity_score": complexity_score,
            "strategy": strategy,
            "n_agents": n_agents,
            "timestamp": time.time(),
        })
    
    def log_execution_outcome(
        self,
        subproblem_id: str,
        strategy: str,
        success: bool,
        duration_ms: float,
        cost: float,
    ) -> None:
        """
        Log an execution outcome.
        
        Args:
            subproblem_id: ID of the sub-problem
            strategy: Strategy that was used
            success: Whether execution succeeded
            duration_ms: Execution duration in milliseconds
            cost: Actual cost incurred
        """
        logger.info(
            f"Outcome for {subproblem_id}: success={success}, "
            f"strategy={strategy}, duration={duration_ms:.2f}ms, cost={cost:.2f}"
        )
        
        self._metrics.append({
            "type": "outcome",
            "subproblem_id": subproblem_id,
            "strategy": strategy,
            "success": success,
            "duration_ms": duration_ms,
            "cost": cost,
            "timestamp": time.time(),
        })
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get logged metrics."""
        return self._metrics.copy()
    
    def get_crew(self, crew_id: str) -> Optional[Crew]:
        """Get a specific crew by ID."""
        return self._crews.get(crew_id)
