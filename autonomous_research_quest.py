"""
Autonomous Research-Quest Orchestrator

Links Research-Quest SOP generation with the RESE (Recursive Execution) engine
and CrewAI Agentic loops for 100% autonomous discovery cycles.

Features:
- Stage-aware SOP generation using MAKER-v2 logic
- Agentic loop execution via AIHierarchicalCrew
- RESE-style reliability (Circuit Breaker, Exponential Backoff)
- Real-time progress streaming readiness
"""

import asyncio
import logging
import uuid
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field

# Internal Imports
from sop_generator_research_quest import ResearchQuestSOPGenerator, RESEARCH_QUEST_STAGES
from crewai_research_core import AIHierarchicalCrew, HierarchicalTask, CrewLevel
from openevolve.kernel.schema import WorkflowState, SubProblem, SubProblemStatus
from glue.orchestration.rese_pipeline import CircuitBreaker, PipelineLogger, retry_with_backoff

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autonomous_research")

@dataclass
class StageExecutionResult:
    """Result of an autonomous stage execution."""
    stage_id: int
    success: bool
    sop_markdown: str
    execution_output: Dict[str, Any]
    metrics: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class AutonomousResearchQuestOrchestrator:
    """
    Orchestrates autonomous research cycles by linking SOP generation to agentic execution.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.sop_generator = ResearchQuestSOPGenerator()
        self.crew = AIHierarchicalCrew(manager_llm_config={"model": model, "temperature": 0.2})
        self.logger = PipelineLogger()
        self.circuit_breaker = CircuitBreaker(threshold=3, logger=self.logger)
        
        # Register core research workers
        self._initialize_workers()

    def _initialize_workers(self):
        """Initialize the default research crew."""
        self.crew.register_worker(
            agent_id="research_architect",
            name="Dr. Archi",
            role="Research Architect",
            expertise=["Problem Decomposition", "Methodology Design", "System Architecture"],
            max_capacity=3
        )
        self.crew.register_worker(
            agent_id="data_scientist",
            name="Analyst Anna",
            role="Data Scientist",
            expertise=["Data Analysis", "Statistical Verification", "Pattern Recognition"],
            max_capacity=5
        )
        self.crew.register_worker(
            agent_id="domain_specialist",
            name="Expert Ed",
            role="Domain Specialist",
            expertise=["Scientific Research", "Literature Review", "Hypothesis Testing"],
            max_capacity=5
        )

    async def execute_research_stage(
        self,
        research_question: str,
        stage_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> StageExecutionResult:
        """
        Executes a single Research-Quest stage autonomously.
        
        1. Generate SOP for the stage
        2. Convert SOP to HierarchicalTask
        3. Execute via AI Crew with RESE reliability
        """
        correlation_id = f"res_quest_{stage_id}_{uuid.uuid4().hex[:8]}"
        self.logger.info(f"Starting Autonomous Stage {stage_id} for: {research_question}", correlation_id=correlation_id)

        try:
            # Step 1: Generate SOP
            # Requirement: once approved in UI, system generates SOP
            sop = await self.sop_generator.generate_stage_sop(
                research_question=research_question,
                stage_id=stage_id,
                context=json.dumps(context) if context else None
            )
            sop_md = sop.to_markdown()
            self.logger.info(f"Generated SOP for Stage {stage_id}", correlation_id=correlation_id)

            # Step 2: Create Hierarchical Task
            task = HierarchicalTask(
                task_id=f"task_{correlation_id}",
                title=f"Research-Quest Stage {stage_id}: {sop.title}",
                description=f"Follow the SOP to complete the research stage.

SOP Details:
{sop.description}

Tasks:
" + 
                            "
".join([f"- {p.action}" for p in sop.protocols]),
                level=CrewLevel.MANAGER,
                context={
                    "stage_id": stage_id,
                    "research_question": research_question,
                    "sop": sop.to_dict(),
                    "additional_context": context
                }
            )

            # Step 3: Execute with RESE-style reliability
            async def _run_task():
                return await self.crew.execute_with_delegation(task, context=context)

            # Wrap in circuit breaker and retry logic
            start_time = time.time()
            
            # Note: retry_with_backoff is synchronous in rese_pipeline.py, 
            # we adapt it or use a simple async retry here for the POC
            max_retries = 2
            execution_data = None
            last_err = None
            
            for attempt in range(max_retries + 1):
                try:
                    execution_data = await _run_task()
                    break
                except Exception as e:
                    last_err = e
                    self.logger.warning(f"Attempt {attempt+1} failed: {e}", correlation_id=correlation_id)
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
            
            if not execution_data:
                raise last_err or RuntimeError("Execution failed after retries")

            duration = time.time() - start_time
            self.logger.info(f"Stage {stage_id} completed successfully in {duration:.2f}s", correlation_id=correlation_id)

            return StageExecutionResult(
                stage_id=stage_id,
                success=True,
                sop_markdown=sop_md,
                execution_output=execution_data,
                metrics={
                    "duration_seconds": duration,
                    "correlation_id": correlation_id,
                    "worker_count": len(execution_data.get("worker_results", []))
                }
            )

        except Exception as e:
            self.logger.error(f"Stage {stage_id} failed: {e}", correlation_id=correlation_id)
            return StageExecutionResult(
                stage_id=stage_id,
                success=False,
                sop_markdown="",
                execution_output={"error": str(e)},
                metrics={"correlation_id": correlation_id}
            )

    async def run_autonomous_quest(
        self,
        research_question: str,
        start_stage: int = 1,
        end_stage: int = 8
    ) -> Dict[int, StageExecutionResult]:
        """Runs multiple research stages in sequence."""
        results = {}
        current_context = {"research_history": []}

        for stage_id in range(start_stage, end_stage + 1):
            result = await self.execute_research_stage(research_question, stage_id, context=current_context)
            results[stage_id] = result
            
            if not result.success:
                logger.error(f"Autonomous quest halted at stage {stage_id}")
                break
            
            # Propagate knowledge to next stage
            current_context["research_history"].append({
                "stage_id": stage_id,
                "findings": result.execution_output.get("final_result", {}).get("summary", "No summary")
            })
            
        return results

# =============================================================================
# INTEGRATION TEST / DEMO
# =============================================================================

async def demo_autonomous_research():
    print("
" + "="*80)
    print("DEMO: AUTONOMOUS RESEARCH-QUEST PIPELINE")
    print("="*80)
    
    orchestrator = AutonomousResearchQuestOrchestrator()
    question = "Investigate the feasibility of room-temperature superconducting graphene-piezoelectric composites."
    
    print(f"Research Question: {question}
")
    
    # Execute Stage 1: Initialization
    print(">>> Executing Stage 1: Initialization...")
    result = await orchestrator.execute_research_stage(question, 1)
    
    if result.success:
        print(f"
[SUCCESS] Stage 1 Completed.")
        print(f"SOP Length: {len(result.sop_markdown)} characters")
        print(f"Findings Summary: {result.execution_output.get('final_result', {}).get('summary', 'N/A')}")
        print(f"Duration: {result.metrics['duration_seconds']:.2f}s")
    else:
        print(f"
[FAILED] Stage 1 Error: {result.execution_output.get('error')}")

if __name__ == "__main__":
    asyncio.run(demo_autonomous_research())
