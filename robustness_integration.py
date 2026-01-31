"""
Robustness Integration - The "Iron Dome" Layer

Integrates all 5 robustness components into the OpenEvolve system:
1. Execution Sandbox - Secure code execution
2. Vision-Language Monitor - UI verification and visual feedback
3. Live Web Interface - Web research and knowledge ingestion
4. System 1 Router - Latency optimization
5. Chronicle Memory - Temporal episodic memory

This creates a comprehensive safety and intelligence layer for the system.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Import the robustness components
from execution_sandbox import (
    ExecutionSandbox, SandboxConfig, SandboxProvider,
    SecurityPolicy, execute_securely
)
from vision_language_monitor import (
    VisionLanguageMonitor, VLMConfig, VLMProvider,
    AnalysisType, VisualAnalysis
)
from live_web_interface import (
    ResearchAgent, ResearchQuery, BrowserConfig,
    BrowserEngine, quick_research
)
from system1_router import (
    System1Router, RouterConfig, ComplexityLevel,
    ModelTier, RouteDecision
)
from chronicle_memory import (
    ChronicleMemory, ChronicleEvent, EventType,
    Outcome, create_chronicle
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RobustnessConfig:
    """Configuration for the robustness layer"""
    # Sandbox config
    sandbox_provider: SandboxProvider = SandboxProvider.DOCKER
    sandbox_timeout: int = 30
    
    # VLM config
    vlm_provider: VLMProvider = VLMProvider.OPENAI
    vlm_model: str = "gpt-4o"
    
    # Web research config
    browser_engine: BrowserEngine = BrowserEngine.PLAYWRIGHT
    enable_multion: bool = False
    
    # Router config
    router_config: RouterConfig = field(default_factory=RouterConfig)
    
    # Chronicle config
    chronicle_storage_path: str = "./chronicle_store"
    
    # Feature toggles
    enable_sandbox: bool = True
    enable_vlm: bool = True
    enable_web_research: bool = True
    enable_router: bool = True
    enable_chronicle: bool = True


class RobustnessCoordinator:
    """
    Main coordinator for all robustness components.
    
    Provides a unified interface for:
    - Secure code execution
    - Visual verification
    - Web research
    - Intelligent routing
    - Memory management
    """
    
    def __init__(self, config: RobustnessConfig = None):
        self.config = config or RobustnessConfig()
        
        # Initialize components
        self.sandbox: Optional[ExecutionSandbox] = None
        self.vlm: Optional[VisionLanguageMonitor] = None
        self.web_research: Optional[ResearchAgent] = None
        self.router: Optional[System1Router] = None
        self.chronicle: Optional[ChronicleMemory] = None
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all enabled components"""
        if self._initialized:
            return
        
        logger.info("Initializing Robustness Layer...")
        
        # 1. Execution Sandbox
        if self.config.enable_sandbox:
            sandbox_config = SandboxConfig(
                provider=self.config.sandbox_provider,
                timeout_seconds=self.config.sandbox_timeout
            )
            self.sandbox = ExecutionSandbox(sandbox_config)
            await self.sandbox.start()
            logger.info("✓ Execution Sandbox initialized")
        
        # 2. Vision-Language Monitor
        if self.config.enable_vlm:
            vlm_config = VLMConfig(
                provider=self.config.vlm_provider,
                model=self.config.vlm_model
            )
            self.vlm = VisionLanguageMonitor(vlm_config)
            await self.vlm.initialize()
            logger.info("✓ Vision-Language Monitor initialized")
        
        # 3. Live Web Interface
        if self.config.enable_web_research:
            browser_config = BrowserConfig(
                engine=self.config.browser_engine
            )
            self.web_research = ResearchAgent(
                browser_config,
                enable_multion=self.config.enable_multion
            )
            await self.web_research.initialize()
            logger.info("✓ Live Web Interface initialized")
        
        # 4. System 1 Router
        if self.config.enable_router:
            self.router = System1Router(self.config.router_config)
            logger.info("✓ System 1 Router initialized")
        
        # 5. Chronicle Memory
        if self.config.enable_chronicle:
            self.chronicle = await create_chronicle(
                storage_path=self.config.chronicle_storage_path
            )
            logger.info("✓ Chronicle Memory initialized")
        
        self._initialized = True
        logger.info("Robustness Layer fully initialized")
    
    async def close(self):
        """Cleanup all components"""
        if self.sandbox:
            await self.sandbox.stop()
        if self.vlm:
            await self.vlm.close()
        if self.web_research:
            await self.web_research.close()
        
        self._initialized = False
        logger.info("Robustness Layer shutdown complete")
    
    # =================================================================
    # Secure Execution Interface
    # =================================================================
    
    async def execute_code_securely(
        self,
        code: str,
        language: str = "python",
        timeout: int = None,
        agent_id: str = None
    ) -> Dict[str, Any]:
        """
        Execute code in the secure sandbox
        
        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout
            agent_id: ID of the agent executing the code
            
        Returns:
            Execution result with output and metadata
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not enabled")
        
        # Record in chronicle
        if self.chronicle and agent_id:
            self.chronicle.set_agent(agent_id)
            await self.chronicle.start_action(
                "code_execution",
                {"language": language, "code_length": len(code)},
                f"Executing {language} code in sandbox"
            )
        
        try:
            result = await self.sandbox.execute(code, language, timeout)
            
            # Record completion
            if self.chronicle and agent_id:
                outcome = Outcome.SUCCESS if result.status.value == "success" else Outcome.FAILURE
                await self.chronicle.complete_action(
                    outcome=outcome,
                    result={"exit_code": result.exit_code},
                    duration_ms=result.execution_time_ms
                )
            
            return {
                "success": result.status.value == "success",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "execution_time_ms": result.execution_time_ms,
                "execution_id": result.execution_id,
                "sandbox_id": result.sandbox_id
            }
            
        except Exception as e:
            if self.chronicle and agent_id:
                await self.chronicle.complete_action(
                    outcome=Outcome.ERROR,
                    result={"error": str(e)}
                )
            raise
    
    # =================================================================
    # Visual Verification Interface
    # =================================================================
    
    async def verify_ui_fix(
        self,
        url: str,
        description: str,
        acceptance_criteria: List[str],
        agent_id: str = None
    ) -> Dict[str, Any]:
        """
        Verify a UI fix visually using VLM
        
        Example: Blue Team says "I fixed the node rendering"
        → VLM takes screenshot and confirms visually
        
        Args:
            url: URL of the application
            description: Description of the fix
            acceptance_criteria: List of criteria to verify
            agent_id: ID of the agent requesting verification
            
        Returns:
            Visual analysis results
        """
        if not self.vlm:
            raise RuntimeError("VLM not enabled")
        
        # Record in chronicle
        if self.chronicle and agent_id:
            self.chronicle.set_agent(agent_id)
            await self.chronicle.start_action(
                "ui_verification",
                {"url": url, "description": description},
                f"Verifying UI fix: {description}"
            )
        
        try:
            analysis = await self.vlm.verify_ui_fix(
                url, description, acceptance_criteria
            )
            
            # Parse result for success/failure
            success = "failed" not in analysis.summary.lower()
            
            # Record completion
            if self.chronicle and agent_id:
                outcome = Outcome.SUCCESS if success else Outcome.FAILURE
                await self.chronicle.complete_action(
                    outcome=outcome,
                    result={"verified": success, "analysis_id": analysis.analysis_id}
                )
            
            return {
                "verified": success,
                "summary": analysis.summary,
                "issues": analysis.issues,
                "analysis_id": analysis.analysis_id,
                "confidence": analysis.confidence
            }
            
        except Exception as e:
            if self.chronicle and agent_id:
                await self.chronicle.complete_action(
                    outcome=Outcome.ERROR,
                    result={"error": str(e)}
                )
            raise
    
    async def monitor_canvas(
        self,
        url: str = "http://localhost:8501",
        expected_nodes: List[Dict] = None,
        agent_id: str = None
    ) -> Dict[str, Any]:
        """
        Monitor Bubblelab canvas rendering
        
        Args:
            url: URL of the Bubblelab instance
            expected_nodes: Expected nodes to verify
            agent_id: ID of the monitoring agent
            
        Returns:
            Canvas analysis results
        """
        if not self.vlm:
            raise RuntimeError("VLM not enabled")
        
        analysis = await self.vlm.monitor_bubblelab_canvas(url, expected_nodes)
        
        return {
            "analysis_id": analysis.analysis_id,
            "summary": analysis.summary,
            "issues": analysis.issues,
            "elements_detected": len(analysis.elements),
            "timestamp": analysis.timestamp.isoformat()
        }
    
    # =================================================================
    # Web Research Interface
    # =================================================================
    
    async def research_error_solution(
        self,
        error_message: str,
        context: str = "",
        agent_id: str = None
    ) -> Dict[str, Any]:
        """
        Research solution for an error from the web
        
        When Blue Team hits a Z3 error they haven't seen,
        they should research the actual solution instead of hallucinating.
        
        Args:
            error_message: The error message
            context: Additional context
            agent_id: ID of the researching agent
            
        Returns:
            Research results with potential solutions
        """
        if not self.web_research:
            raise RuntimeError("Web research not enabled")
        
        # Record in chronicle
        if self.chronicle and agent_id:
            self.chronicle.set_agent(agent_id)
            await self.chronicle.start_action(
                "error_research",
                {"error": error_message[:100]},
                f"Researching solution for error: {error_message[:50]}..."
            )
        
        try:
            result = await self.web_research.fetch_error_solution(
                error_message, context
            )
            
            if result:
                # Record completion
                if self.chronicle and agent_id:
                    await self.chronicle.complete_action(
                        outcome=Outcome.SUCCESS,
                        result={
                            "sources": [p.url for p in result.pages],
                            "findings_count": len(result.key_findings)
                        }
                    )
                
                return {
                    "found": True,
                    "query": result.query,
                    "summary": result.summary,
                    "sources": [p.url for p in result.pages],
                    "key_findings": result.key_findings,
                    "execution_time": result.execution_time_seconds
                }
            else:
                if self.chronicle and agent_id:
                    await self.chronicle.complete_action(
                        outcome=Outcome.FAILURE,
                        result={"error": "No results found"}
                    )
                
                return {"found": False, "reason": "No research results"}
                
        except Exception as e:
            if self.chronicle and agent_id:
                await self.chronicle.complete_action(
                    outcome=Outcome.ERROR,
                    result={"error": str(e)}
                )
            raise
    
    async def ingest_documentation(
        self,
        urls: List[str],
        agent_id: str = None
    ) -> Dict[str, Any]:
        """
        Ingest documentation into knowledge base
        
        Args:
            urls: Documentation URLs to ingest
            agent_id: ID of the ingesting agent
            
        Returns:
            Ingestion results
        """
        if not self.web_research:
            raise RuntimeError("Web research not enabled")
        
        query = ResearchQuery(
            query="Documentation ingestion",
            target_sources=urls,
            depth="standard"
        )
        
        result = await self.web_research.research(query)
        
        return {
            "pages_ingested": len(result.pages),
            "sources": [p.url for p in result.pages],
            "summary": result.summary
        }
    
    # =================================================================
    # Routing Interface
    # =================================================================
    
    async def route_request(
        self,
        request: str,
        context: Dict[str, Any] = None,
        handlers: Dict[ModelTier, Callable] = None
    ) -> Dict[str, Any]:
        """
        Route a request to appropriate processing tier
        
        Args:
            request: The user request
            context: Additional context
            handlers: Optional handlers for each tier
            
        Returns:
            Routing decision and response
        """
        if not self.router:
            raise RuntimeError("Router not enabled")
        
        # Register handlers if provided
        if handlers:
            for tier, handler in handlers.items():
                self.router.register_handler(tier, handler)
        
        # Route and execute
        response, result = await self.router.execute_routed(request, context)
        
        return {
            "response": response,
            "routing": {
                "complexity": result.decision.complexity.value,
                "model_tier": result.decision.model_tier.value,
                "selected_model": result.decision.selected_model,
                "estimated_latency_ms": result.decision.estimated_latency_ms,
                "estimated_cost": result.decision.estimated_cost,
                "actual_latency_ms": result.actual_latency_ms,
                "reasoning": result.decision.reasoning
            }
        }
    
    def get_complexity(self, request: str) -> str:
        """Quick complexity classification"""
        if not self.router:
            return "unknown"
        
        decision = asyncio.run(self.router.route(request))
        return decision.complexity.value
    
    # =================================================================
    # Memory Interface
    # =================================================================
    
    async def check_for_loops(
        self,
        action: str,
        parameters: Dict[str, Any],
        agent_id: str = None
    ) -> Dict[str, Any]:
        """
        Check if an action would create a loop
        
        Prevents the Self-Healing loop from running indefinitely.
        
        Args:
            action: Action name
            parameters: Action parameters
            agent_id: ID of the checking agent
            
        Returns:
            Loop detection results
        """
        if not self.chronicle:
            return {"should_prevent": False}
        
        if agent_id:
            self.chronicle.set_agent(agent_id)
        
        should_prevent, warning = await self.chronicle.check_for_loops(
            action, parameters
        )
        
        return {
            "should_prevent": should_prevent,
            "warning": warning,
            "recommendation": "pivot_strategy" if should_prevent else "proceed"
        }
    
    async def record_attempt(
        self,
        action: str,
        parameters: Dict[str, Any] = None,
        agent_id: str = None
    ) -> str:
        """
        Record the start of an attempt
        
        Args:
            action: Action name
            parameters: Action parameters
            agent_id: ID of the acting agent
            
        Returns:
            Event ID
        """
        if not self.chronicle:
            return ""
        
        if agent_id:
            self.chronicle.set_agent(agent_id)
        
        event = await self.chronicle.start_action(action, parameters)
        return event.event_id
    
    async def complete_attempt(
        self,
        success: bool,
        result: Any = None,
        lesson: str = None
    ):
        """
        Complete the current attempt
        
        Args:
            success: Whether the attempt succeeded
            result: Result of the attempt
            lesson: Lesson learned
        """
        if not self.chronicle:
            return
        
        outcome = Outcome.SUCCESS if success else Outcome.FAILURE
        await self.chronicle.complete_action(
            outcome=outcome,
            result=result,
            lesson=lesson
        )
    
    async def get_experience(
        self,
        action_type: str = None,
        timeframe_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get experience summary"""
        if not self.chronicle:
            return {}
        
        return await self.chronicle.get_experience_summary(
            action_type, timeframe_minutes
        )
    
    # =================================================================
    # Blue Team Integration
    # =================================================================
    
    async def blue_team_execute_fix(
        self,
        code: str,
        fix_description: str,
        agent_id: str = "blue-team"
    ) -> Dict[str, Any]:
        """
        Blue Team: Execute a fix securely with full verification
        
        This is the complete pipeline:
        1. Record attempt in chronicle
        2. Check for loops
        3. Execute code in sandbox
        4. Verify results (visual if applicable)
        5. Record outcome
        
        Args:
            code: Fix code to execute
            fix_description: Description of the fix
            agent_id: Blue Team agent ID
            
        Returns:
            Complete execution results
        """
        results = {
            "fix_description": fix_description,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # 1. Check for loops
        loop_check = await self.check_for_loops(
            "code_fix", {"description": fix_description}, agent_id
        )
        results["loop_check"] = loop_check
        
        if loop_check["should_prevent"]:
            results["status"] = "prevented"
            results["reason"] = loop_check["warning"]
            return results
        
        # 2. Execute in sandbox
        try:
            execution = await self.execute_code_securely(
                code, agent_id=agent_id
            )
            results["execution"] = execution
            
            if execution["success"]:
                results["status"] = "success"
            else:
                results["status"] = "execution_failed"
                
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            await self.complete_attempt(False, {"error": str(e)})
            return results
        
        # 3. Complete the attempt
        await self.complete_attempt(
            success=results["status"] == "success",
            result={"exit_code": execution.get("exit_code")},
            lesson=f"Fix '{fix_description}' executed with status {results['status']}"
        )
        
        return results
    
    async def blue_team_research_fix(
        self,
        error_message: str,
        agent_id: str = "blue-team"
    ) -> Dict[str, Any]:
        """
        Blue Team: Research a fix for an error
        
        When Blue Team encounters an unfamiliar error, they should
        research the solution instead of hallucinating.
        
        Args:
            error_message: The error to research
            agent_id: Blue Team agent ID
            
        Returns:
            Research results with potential solutions
        """
        # Check for loops
        loop_check = await self.check_for_loops(
            "error_research", {"error": error_message[:50]}, agent_id
        )
        
        if loop_check["should_prevent"]:
            return {
                "status": "prevented",
                "warning": loop_check["warning"]
            }
        
        # Record attempt
        await self.record_attempt("error_research", {"error": error_message[:100]}, agent_id)
        
        # Research
        research = await self.research_error_solution(error_message, agent_id=agent_id)
        
        return {
            "status": "researched",
            "results": research
        }
    
    # =================================================================
    # Statistics and Monitoring
    # =================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all components"""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        if self.router:
            stats["components"]["router"] = self.router.get_stats()
        
        if self.chronicle:
            stats["components"]["chronicle"] = asyncio.run(
                self.chronicle.get_stats()
            )
        
        if self.sandbox:
            stats["components"]["sandbox"] = {
                "executions": len(self.sandbox.get_execution_history())
            }
        
        return stats


# Singleton instance for global access
_robustness_coordinator: Optional[RobustnessCoordinator] = None


async def get_robustness_layer(
    config: RobustnessConfig = None
) -> RobustnessCoordinator:
    """Get or create the global robustness layer"""
    global _robustness_coordinator
    
    if _robustness_coordinator is None:
        _robustness_coordinator = RobustnessCoordinator(config)
        await _robustness_coordinator.initialize()
    
    return _robustness_coordinator


# Convenience functions
async def execute_secure(code: str, language: str = "python") -> Dict[str, Any]:
    """Quick secure execution"""
    layer = await get_robustness_layer()
    return await layer.execute_code_securely(code, language)


async def verify_fix(url: str, description: str, criteria: List[str]) -> Dict[str, Any]:
    """Quick UI fix verification"""
    layer = await get_robustness_layer()
    return await layer.verify_ui_fix(url, description, criteria)


async def research_error(error: str) -> Dict[str, Any]:
    """Quick error research"""
    layer = await get_robustness_layer()
    return await layer.research_error_solution(error)


# Example usage
if __name__ == "__main__":
    async def demo():
        print("=" * 70)
        print("ROBUSTNESS INTEGRATION LAYER DEMO - The 'Iron Dome'")
        print("=" * 70)
        
        # Initialize
        config = RobustnessConfig(
            enable_sandbox=True,
            enable_vlm=True,
            enable_web_research=False,  # Skip for demo (requires browser)
            enable_router=True,
            enable_chronicle=True
        )
        
        layer = await get_robustness_layer(config)
        
        print("\n✓ Robustness Layer initialized")
        print(f"  Components: {list(layer.get_stats()['components'].keys())}")
        
        # Demo 1: Secure Code Execution
        print("\n" + "=" * 70)
        print("DEMO 1: Secure Code Execution (Sandbox)")
        print("=" * 70)
        
        code = """
import sys
print("Running in secure sandbox!")
print(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
x = sum(range(100))
print(f"Result: {x}")
"""
        
        result = await layer.execute_code_securely(code, agent_id="demo-agent")
        print(f"\nExecution: {'✓ SUCCESS' if result['success'] else '✗ FAILED'}")
        print(f"Output: {result['stdout'][:200]}")
        print(f"Time: {result['execution_time_ms']:.0f}ms")
        
        # Demo 2: Loop Detection
        print("\n" + "=" * 70)
        print("DEMO 2: Loop Detection (Chronicle Memory)")
        print("=" * 70)
        
        # Simulate multiple attempts
        for i in range(3):
            check = await layer.check_for_loops(
                "strategy_A",
                {"approach": "quick_fix"},
                agent_id="demo-agent"
            )
            
            if check["should_prevent"]:
                print(f"\nAttempt {i+1}: ⚠️ LOOP DETECTED!")
                print(f"  Warning: {check['warning']}")
                break
            else:
                print(f"\nAttempt {i+1}: ✓ Proceeding")
                # Record the attempt
                await layer.record_attempt("strategy_A", {"approach": "quick_fix"}, "demo-agent")
                await layer.complete_attempt(success=False)
        
        # Demo 3: Routing
        print("\n" + "=" * 70)
        print("DEMO 3: System 1 Router (Latency Optimization)")
        print("=" * 70)
        
        test_requests = [
            "What time is it?",
            "Fix this typo",
            "Optimize this Z3 solver configuration",
            "Generate a formally verified sorting algorithm"
        ]
        
        for request in test_requests:
            decision = await layer.router.route(request)
            print(f"\nRequest: {request[:40]}...")
            print(f"  Complexity: {decision.complexity.value}")
            print(f"  Tier: {decision.model_tier.value}")
            print(f"  Model: {decision.selected_model}")
            print(f"  Est. Latency: {decision.estimated_latency_ms:.0f}ms")
        
        # Stats
        print("\n" + "=" * 70)
        print("Statistics:")
        print("=" * 70)
        print(json.dumps(layer.get_stats(), indent=2))
        
        # Cleanup
        await layer.close()
        print("\n✓ Demo complete")
    
    asyncio.run(demo())
