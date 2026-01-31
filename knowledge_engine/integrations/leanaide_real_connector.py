"""
Real LeanAIDE Client Connector - Production Ready

Provides actual integration with LeanAIDE server:
- Full task execution
- Proof state management
- Tactic application
- Error handling and recovery
- Connection pooling
- Health checking

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import logging
import aiohttp
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

# Import LeanAIDE client if available
try:
    from leanaide_client import LeanAideClient, LeanAideConfig, TaskType, LeanAideResult
    LEANAIDE_LIB_AVAILABLE = True
except ImportError:
    LEANAIDE_LIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LeanAideClient library not available, using direct HTTP")


class LeanTaskType(Enum):
    """LeanAIDE task types."""
    TRANSLATE_THM = "translate_thm"
    TRANSLATE_DEF = "translate_def"
    PROVE = "prove"
    VERIFY = "verify"
    ELABORATE = "elaborate"
    MATH_QUERY = "math_query"


@dataclass
class LeanAideRealConfig:
    """Configuration for LeanAIDE connection."""
    host: str = "localhost"
    port: int = 7654
    base_url: Optional[str] = None
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_pool_size: int = 10
    keepalive_timeout: float = 30.0
    
    def __post_init__(self):
        if self.base_url is None:
            self.base_url = f"http://{self.host}:{self.port}"


@dataclass
class ProofState:
    """Current proof state from Lean."""
    goals: List[Dict[str, Any]] = field(default_factory=list)
    current_goal: Optional[str] = None
    hypotheses: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        return len(self.goals) == 0 and self.error is None


@dataclass
class TacticResult:
    """Result of tactic application."""
    success: bool
    new_state: Optional[ProofState] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    proof_step: Optional[str] = None


class LeanAideRealConnector:
    """
    Real LeanAIDE connector with full functionality.
    
    Provides:
    - Direct HTTP API communication
    - Connection pooling
    - Automatic retries
    - Health checking
    - Proof state tracking
    - Error recovery
    """
    
    def __init__(self, config: Optional[LeanAideRealConfig] = None):
        self.config = config or LeanAideRealConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._closed = False
        
        # Statistics
        self.stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
            "avg_response_time_ms": 0.0
        }
        
        # Circuit breaker state
        self.circuit_state = "closed"  # closed, open, half-open
        self.failure_count = 0
        self.failure_threshold = 5
        self.recovery_timeout = 30.0
        self.last_failure_time = None
        
        logger.info(f"LeanAideRealConnector initialized: {self.config.base_url}")
    
    async def initialize(self):
        """Initialize HTTP session."""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.config.connection_pool_size,
                keepalive_timeout=self.config.keepalive_timeout
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            logger.info("HTTP session initialized")
    
    async def close(self):
        """Close connection."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("HTTP session closed")
        self._closed = True
    
    async def health_check(self) -> bool:
        """Check if LeanAIDE server is healthy."""
        if self._closed:
            return False
        
        try:
            async with self.session.get(
                f"{self.config.base_url}/",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    async def execute_task(
        self,
        task_type: LeanTaskType,
        content: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a task on LeanAIDE server.
        
        Args:
            task_type: Type of task
            content: Task content
            timeout: Optional timeout override
            
        Returns:
            Task result
        """
        if self._closed:
            raise RuntimeError("Connector is closed")
        
        # Check circuit breaker
        if not await self._check_circuit():
            return {
                "success": False,
                "error": "Circuit breaker is open"
            }
        
        await self.initialize()
        
        url = f"{self.config.base_url}/api/task"
        payload = {
            "task": task_type.value,
            "content": content
        }
        
        start_time = datetime.now(timezone.utc)
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                self.stats["requests"] += 1
                
                async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout or self.config.timeout)
                ) as response:
                    response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update stats
                        self.stats["successes"] += 1
                        self._update_avg_time(response_time)
                        
                        # Reset circuit breaker on success
                        self._record_success()
                        
                        return result
                    
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"HTTP {response.status}: {error_text}")
                
            except Exception as e:
                last_error = e
                self.stats["retries"] += 1
                self._record_failure()
                
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        # All retries failed
        self.stats["failures"] += 1
        logger.error(f"All retries failed: {last_error}")
        
        return {
            "success": False,
            "error": str(last_error),
            "retries_exhausted": True
        }
    
    async def prove_theorem(
        self,
        theorem_statement: str,
        auto_tactics: Optional[List[str]] = None,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Prove a theorem using LeanAIDE.
        
        Args:
            theorem_statement: Theorem to prove
            auto_tactics: Optional list of tactics to try
            max_steps: Maximum proof steps
            
        Returns:
            Proof result with tactics and proof script
        """
        try:
            # Try to prove using PROVE task
            result = await self.execute_task(
                LeanTaskType.PROVE,
                theorem_statement
            )
            
            if result.get("success"):
                return {
                    "success": True,
                    "theorem": theorem_statement,
                    "proof": result.get("proof"),
                    "tactics": result.get("tactics", []),
                    "execution_time_ms": result.get("execution_time_ms", 0)
                }
            else:
                return {
                    "success": False,
                    "theorem": theorem_statement,
                    "error": result.get("error", "Proof failed"),
                    "execution_time_ms": result.get("execution_time_ms", 0)
                }
                
        except Exception as e:
            logger.error(f"Theorem proving failed: {e}")
            return {
                "success": False,
                "theorem": theorem_statement,
                "error": str(e)
            }
    
    async def apply_tactic(
        self,
        goal_state: str,
        tactic: str
    ) -> TacticResult:
        """
        Apply tactic to goal state.
        
        Args:
            goal_state: Current goal state
            tactic: Tactic to apply
            
        Returns:
            Tactic result with new state
        """
        try:
            # Use ELABORATE task to apply tactic
            content = json.dumps({
                "goal": goal_state,
                "tactic": tactic
            })
            
            result = await self.execute_task(
                LeanTaskType.ELABORATE,
                content
            )
            
            if result.get("success"):
                new_state = ProofState(
                    goals=result.get("goals", []),
                    current_goal=result.get("current_goal"),
                    hypotheses=result.get("hypotheses", [])
                )
                
                return TacticResult(
                    success=True,
                    new_state=new_state,
                    execution_time_ms=result.get("execution_time_ms", 0),
                    proof_step=result.get("proof_step")
                )
            else:
                return TacticResult(
                    success=False,
                    error=result.get("error"),
                    execution_time_ms=result.get("execution_time_ms", 0)
                )
                
        except Exception as e:
            return TacticResult(
                success=False,
                error=str(e)
            )
    
    async def translate_to_lean(
        self,
        natural_language: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate natural language to Lean.
        
        Args:
            natural_language: Natural language statement
            context: Optional context
            
        Returns:
            Translation result
        """
        content = natural_language
        if context:
            content = f"Context: {context}\nStatement: {natural_language}"
        
        return await self.execute_task(
            LeanTaskType.TRANSLATE_THM,
            content
        )
    
    async def _check_circuit(self) -> bool:
        """Check circuit breaker state."""
        if self.circuit_state == "closed":
            return True
        
        if self.circuit_state == "open":
            if self.last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
                if elapsed > self.recovery_timeout:
                    self.circuit_state = "half-open"
                    logger.info("Circuit breaker: half-open")
                    return True
            return False
        
        # half-open
        return True
    
    def _record_success(self):
        """Record successful request."""
        if self.circuit_state == "half-open":
            self.circuit_state = "closed"
            logger.info("Circuit breaker: closed")
        
        self.failure_count = 0
    
    def _record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.circuit_state = "open"
            logger.warning(f"Circuit breaker: open (failures: {self.failure_count})")
    
    def _update_avg_time(self, response_time_ms: float):
        """Update average response time."""
        n = self.stats["requests"]
        old_avg = self.stats["avg_response_time_ms"]
        self.stats["avg_response_time_ms"] = (old_avg * (n - 1) + response_time_ms) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self.stats,
            "circuit_state": self.circuit_state,
            "failure_count": self.failure_count,
            "success_rate": (
                self.stats["successes"] / max(self.stats["requests"], 1)
            )
        }


# Global connector
_leanaide_connector: Optional[LeanAideRealConnector] = None


async def get_leanaide_connector() -> LeanAideRealConnector:
    """Get global LeanAIDE connector."""
    global _leanaide_connector
    if _leanaide_connector is None:
        _leanaide_connector = LeanAideRealConnector()
        await _leanaide_connector.initialize()
    return _leanaide_connector


# Example usage
async def example_connector():
    """Example: Using LeanAIDE connector."""
    print("LeanAIDE Real Connector Example")
    print("=" * 60)
    
    connector = await get_leanaide_connector()
    
    # Health check
    healthy = await connector.health_check()
    print(f"\nServer health: {'OK' if healthy else 'FAIL'}")
    
    if healthy:
        # Translate to Lean
        result = await connector.translate_to_lean(
            "For all natural numbers n, n plus 0 equals n"
        )
        print(f"\nTranslation result:")
        print(f"  Success: {result.get('success')}")
        print(f"  Result: {result.get('result', 'N/A')[:100]}")
    
    # Statistics
    stats = connector.get_statistics()
    print(f"\nStatistics:")
    print(f"  Requests: {stats['requests']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Avg response time: {stats['avg_response_time_ms']:.1f} ms")


if __name__ == "__main__":
    asyncio.run(example_connector())
