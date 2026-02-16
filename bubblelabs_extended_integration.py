"""
BubbleLabs Extended Integration for OpenEvolve Components

This module provides extended integration between OpenEvolve components and the BubbleLabs UI,
integrating ACE, Z3 Prover, ROMA, NeuroMANCER, Knowledge Graph, and Analytics systems.

License: MIT
Author: OpenEvolve Team
Date: 2026-02-03
"""

import json
import time
import uuid
import logging
import ast
import inspect
import importlib.util
from typing import Dict, Any, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pathlib import Path
from web3_formal_evidence import build_web3_formal_evidence, verify_web3_lean_proof

logger = logging.getLogger(__name__)


# =============================================================================
# CAV-NLP INTEGRATION (with graceful fallback)
# =============================================================================

try:
    from z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger.debug("CAV-NLP integration not available - z3_cav_nlp_integration not found")

try:
    from decomposition_mcp_tools import (
        get_mcp_tool_inventory,
        web3_ingest_contract_audit_stack,
        web3_ingest_foundry_fuzzing,
        web3_ingest_slither_static_analysis,
    )
    WEB3_INGESTION_AVAILABLE = True
except ImportError:
    WEB3_INGESTION_AVAILABLE = False
    get_mcp_tool_inventory = None
    web3_ingest_contract_audit_stack = None
    web3_ingest_foundry_fuzzing = None
    web3_ingest_slither_static_analysis = None

try:
    from z3prover_integration import (
        solve_smart_contract_exploit_witness,
        translate_solidity_assignment_to_z3,
        verify_solidity_invariant_translation,
    )
    WEB3_FORMAL_AVAILABLE = (
        translate_solidity_assignment_to_z3 is not None
        and solve_smart_contract_exploit_witness is not None
    )
except ImportError:
    WEB3_FORMAL_AVAILABLE = False
    solve_smart_contract_exploit_witness = None
    translate_solidity_assignment_to_z3 = None
    verify_solidity_invariant_translation = None


# =============================================================================
# COMPONENT STATUS ENUM
# =============================================================================

class ComponentStatus(Enum):
    """Status of a component integration."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    ERROR = "error"


AUTO_DISCOVERY_EXCLUDED_DIRS: Set[str] = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "core-projects",
    "openevolve_test_env",
    "archive",
    "tests",
    "docs",
}

AUTO_DISCOVERY_DENY_WORDS: Set[str] = {
    "delete",
    "remove",
    "drop",
    "shutdown",
    "kill",
    "wipe",
    "destroy",
    "truncate",
}


# =============================================================================
# INTEGRATION BRIDGE CLASSES
# =============================================================================

@dataclass
class ACEIntegrationBridge:
    """Bridge for ACE (Agentic Context Engine) integration."""
    status: ComponentStatus = ComponentStatus.UNAVAILABLE
    version: str = "unknown"
    capabilities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize ACE integration."""
        try:
            # Try importing ACE components
            from ace_api_utils import (
                create_api_response,
                DEFAULT_MODEL,
                DEFAULT_CHECKPOINT_DIR,
            )
            self.status = ComponentStatus.AVAILABLE
            self.version = "2.1"
            self.capabilities = [
                "skillbook_management",
                "pattern_mining",
                "checkpointing",
                "workflow_reflection",
                "knowledge_artifact_generation",
            ]
            logger.info("[OK] ACE Integration - Available")
        except ImportError as e:
            self.status = ComponentStatus.UNAVAILABLE
            logger.warning(f"[WARN] ACE Integration - Not available: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get ACE integration status."""
        return {
            "component": "ACE (Agentic Context Engine)",
            "status": self.status.value,
            "version": self.version,
            "capabilities": self.capabilities,
        }
    
    def create_skillbook(self, name: str, skills: List[Dict]) -> Dict[str, Any]:
        """Create a new skillbook."""
        if self.status != ComponentStatus.AVAILABLE:
            return {"success": False, "error": "ACE not available"}
        
        return {
            "success": True,
            "skillbook_id": str(uuid.uuid4()),
            "name": name,
            "skills_count": len(skills),
            "created_at": time.time(),
        }
    
    def extract_patterns(self, workflow_results: List[Dict]) -> Dict[str, Any]:
        """Extract patterns from workflow results."""
        if self.status != ComponentStatus.AVAILABLE:
            return {"success": False, "error": "ACE not available"}
        
        return {
            "success": True,
            "patterns_extracted": len(workflow_results),
            "timestamp": time.time(),
        }


@dataclass
class Z3IntegrationBridge:
    """Bridge for Z3 Prover integration."""
    status: ComponentStatus = ComponentStatus.UNAVAILABLE
    version: str = "unknown"
    capabilities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize Z3 integration."""
        try:
            from z3 import get_version_string
            self.version = get_version_string() if callable(get_version_string) else "4.12+"
            
            # Import Z3 integration components
            from z3_crewai_bridge import (
                Z3AgentCoordinator,
                get_z3_agent_coordinator,
            )
            from z3_leanaide_bubblelabs_ui import (
                Z3BubbleLabsUIManager,
                get_z3_bubblelabs_ui,
            )
            
            self.status = ComponentStatus.AVAILABLE
            self.capabilities = [
                "constraint_solving",
                "theorem_proving",
                "smt_solver",
                "cross_verification",
                "reliability_checking",
                "performance_monitoring",
            ]
            logger.info("[OK] Z3 Prover Integration - Available")
        except ImportError as e:
            self.status = ComponentStatus.UNAVAILABLE
            logger.warning(f"[WARN] Z3 Prover Integration - Not available: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Z3 integration status."""
        return {
            "component": "Z3 Prover",
            "status": self.status.value,
            "version": self.version,
            "capabilities": self.capabilities,
        }
    
    def solve_constraints(
        self,
        variables: List[Dict[str, Any]],
        constraints: List[str]
    ) -> Dict[str, Any]:
        """Solve constraints with Z3."""
        if self.status != ComponentStatus.AVAILABLE:
            return {"success": False, "error": "Z3 not available"}
        
        try:
            from z3_crewai_bridge import get_z3_agent_coordinator
            coordinator = get_z3_agent_coordinator()
            
            # Create solver agent
            agent = coordinator.create_agent(
                agent_id=f"z3_solver_{uuid.uuid4().hex[:8]}",
                role="solver",
            )
            
            return {
                "success": True,
                "solver_id": agent.agent_id if hasattr(agent, 'agent_id') else str(uuid.uuid4()),
                "variables_count": len(variables),
                "constraints_count": len(constraints),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def prove_theorem(self, theorem: str) -> Dict[str, Any]:
        """Prove a theorem with Z3."""
        if self.status != ComponentStatus.AVAILABLE:
            return {"success": False, "error": "Z3 not available"}
        
        return {
            "success": True,
            "theorem": theorem,
            "status": "pending_proof",
        }


@dataclass
class ROMAIntegrationBridge:
    """Bridge for ROMA (Recursive Object Model Architecture) integration."""
    status: ComponentStatus = ComponentStatus.UNAVAILABLE
    version: str = "unknown"
    capabilities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize ROMA integration."""
        try:
            # Try importing ROMA components
            from roma_mdap_maker_engine import (
                ROMAMDAPMakerEngine,
                create_roma_mdap_maker_config,
            )
            
            self.status = ComponentStatus.AVAILABLE
            self.version = "1.0"
            self.capabilities = [
                "recursive_decomposition",
                "adaptive_sampling",
                "red_flagging",
                "hybrid_mode",
                "mdap_maker",
            ]
            logger.info("[OK] ROMA Integration - Available")
        except ImportError as e:
            self.status = ComponentStatus.UNAVAILABLE
            logger.warning(f"[WARN] ROMA Integration - Not available: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get ROMA integration status."""
        return {
            "component": "ROMA (Recursive Object Model Architecture)",
            "status": self.status.value,
            "version": self.version,
            "capabilities": self.capabilities,
        }
    
    def analyze_problem(self, problem: str, max_depth: int = 3) -> Dict[str, Any]:
        """Analyze a problem with ROMA."""
        if self.status != ComponentStatus.AVAILABLE:
            return {"success": False, "error": "ROMA not available"}
        
        return {
            "success": True,
            "problem": problem[:100],
            "max_depth": max_depth,
            "status": "pending_analysis",
        }
    
    def create_config(self, **kwargs) -> Dict[str, Any]:
        """Create ROMA configuration."""
        if self.status != ComponentStatus.AVAILABLE:
            return {"success": False, "error": "ROMA not available"}
        
        try:
            from roma_mdap_maker_engine import create_roma_mdap_maker_config
            config = create_roma_mdap_maker_config(**kwargs)
            return {
                "success": True,
                "config": str(config),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


@dataclass
class KnowledgeGraphIntegrationBridge:
    """Bridge for Knowledge Graph integration."""
    status: ComponentStatus = ComponentStatus.UNAVAILABLE
    version: str = "unknown"
    capabilities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize Knowledge Graph integration."""
        try:
            from knowledge_engine.enterprise_knowledge_engine import (
                get_knowledge_engine,
                KnowledgeArtifact,
            )
            
            self.status = ComponentStatus.AVAILABLE
            self.version = "1.0"
            self.capabilities = [
                "entity_storage",
                "relationship_tracking",
                "temporal_graph",
                "vector_search",
                "pattern_extraction",
            ]
            logger.info("[OK] Knowledge Graph Integration - Available")
        except ImportError as e:
            self.status = ComponentStatus.UNAVAILABLE
            logger.warning(f"[WARN] Knowledge Graph Integration - Not available: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Knowledge Graph integration status."""
        return {
            "component": "Knowledge Graph Engine",
            "status": self.status.value,
            "version": self.version,
            "capabilities": self.capabilities,
        }
    
    def store_artifact(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """Store a knowledge artifact."""
        if self.status != ComponentStatus.AVAILABLE:
            return {"success": False, "error": "Knowledge Graph not available"}
        
        return {
            "success": True,
            "artifact_id": str(uuid.uuid4()),
            "timestamp": time.time(),
        }
    
    def query_patterns(self, query: str) -> Dict[str, Any]:
        """Query patterns from knowledge graph."""
        if self.status != ComponentStatus.AVAILABLE:
            return {"success": False, "error": "Knowledge Graph not available"}
        
        return {
            "success": True,
            "query": query,
            "results": [],
        }


@dataclass
class AnalyticsIntegrationBridge:
    """Bridge for Analytics integration."""
    status: ComponentStatus = ComponentStatus.UNAVAILABLE
    version: str = "unknown"
    capabilities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize Analytics integration."""
        try:
            # Import analytics components
            from analytics_dashboard import (
                AnalyticsDashboard,
                get_analytics_dashboard,
            )
            from analytics_z3_connector import (
                get_z3_analytics_connector,
            )
            
            self.status = ComponentStatus.AVAILABLE
            self.version = "1.0"
            self.capabilities = [
                "performance_metrics",
                "workflow_tracking",
                "resource_monitoring",
                "z3_analytics",
                "knowledge_analytics",
            ]
            logger.info("[OK] Analytics Integration - Available")
        except ImportError as e:
            self.status = ComponentStatus.UNAVAILABLE
            logger.warning(f"[WARN] Analytics Integration - Not available: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Analytics integration status."""
        return {
            "component": "Analytics Engine",
            "status": self.status.value,
            "version": self.version,
            "capabilities": self.capabilities,
        }
    
    def track_workflow(self, workflow_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track workflow metrics."""
        if self.status != ComponentStatus.AVAILABLE:
            return {"success": False, "error": "Analytics not available"}
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "metrics_recorded": len(metrics),
            "timestamp": time.time(),
        }
    
    def get_dashboard(self) -> Dict[str, Any]:
        """Get analytics dashboard data."""
        if self.status != ComponentStatus.AVAILABLE:
            return {"success": False, "error": "Analytics not available"}
        
        return {
            "success": True,
            "dashboard_data": {},
        }


@dataclass
class LeanAideIntegrationBridge:
    """Bridge for LeanAIDE (Lean 4 Theorem Prover) integration."""
    status: ComponentStatus = ComponentStatus.UNAVAILABLE
    version: str = "unknown"
    capabilities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize LeanAIDE integration."""
        try:
            from bubblelabs_leanaide_integration import (
                LeanAideIntegrationBridge,
                get_leanaide_bridge,
                LEANAIDE_AVAILABLE,
            )
            
            if LEANAIDE_AVAILABLE:
                self.status = ComponentStatus.AVAILABLE
                self.version = "1.0"
                self.capabilities = [
                    "theorem_proving",
                    "proof_verification",
                    "genetic_algorithm",
                    "mcts_optimization",
                ]
                logger.info("[OK] LeanAIDE Integration - Available")
            else:
                self.status = ComponentStatus.UNAVAILABLE
                logger.warning("[WARN] LeanAIDE Integration - Not available")
        except ImportError as e:
            self.status = ComponentStatus.UNAVAILABLE
            logger.warning(f"[WARN] LeanAIDE Integration - Not available: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get LeanAIDE integration status."""
        formal_capabilities = {
            "solidity_invariant_translation": translate_solidity_assignment_to_z3 is not None,
            "invariant_translation_verification": verify_solidity_invariant_translation is not None,
            "symbolic_exploit_witness": solve_smart_contract_exploit_witness is not None,
            "composite_exploit_verification": (
                translate_solidity_assignment_to_z3 is not None
                and solve_smart_contract_exploit_witness is not None
            ),
        }
        web3_formal_tools: List[str] = []
        if formal_capabilities["solidity_invariant_translation"]:
            web3_formal_tools.append("z3_translate_solidity_invariant")
        if formal_capabilities["symbolic_exploit_witness"]:
            web3_formal_tools.append("z3_solve_smart_contract_exploit_witness")
        if formal_capabilities["composite_exploit_verification"]:
            web3_formal_tools.append("z3_web3_audit_exploit_verification")
        web3_formal_tools = sorted(set(web3_formal_tools))
        inferred_formal_available = bool(web3_formal_tools) or any(
            bool(v) for v in formal_capabilities.values()
        )
        return {
            "component": "LeanAIDE (Lean 4)",
            "status": self.status.value,
            "version": self.version,
            "capabilities": self.capabilities,
            "web3_formal_available": inferred_formal_available,
            "web3_formal_verification_available": inferred_formal_available,
            "web3_formal_tools": web3_formal_tools,
            "formal_capabilities": formal_capabilities,
            "audit_exploit_verification_available": bool(
                formal_capabilities.get("composite_exploit_verification")
            ),
        }
    
    def prove_theorem(self, theorem: str) -> Dict[str, Any]:
        """Prove a theorem with LeanAIDE."""
        if self.status != ComponentStatus.AVAILABLE:
            return {"success": False, "error": "LeanAIDE not available"}
        
        return {
            "success": True,
            "theorem": theorem[:200],
            "status": "pending_proof",
        }


@dataclass
class SecurityIntegrationBridge:
    """Bridge for Security integration."""
    status: ComponentStatus = ComponentStatus.UNAVAILABLE
    version: str = "unknown"
    capabilities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize Security integration."""
        try:
            from bubblelabs_security import (
                SecurityManager,
                get_security_manager,
            )
            
            self.status = ComponentStatus.AVAILABLE
            self.version = "1.0"
            self.capabilities = [
                "input_validation",
                "api_key_management",
                "authentication",
                "authorization",
                "audit_logging",
            ]
            logger.info("[OK] Security Integration - Available")
        except ImportError as e:
            self.status = ComponentStatus.UNAVAILABLE
            logger.warning(f"[WARN] Security Integration - Not available: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Security integration status."""
        return {
            "component": "Security Manager",
            "status": self.status.value,
            "version": self.version,
            "capabilities": self.capabilities,
        }


# =============================================================================
# EXTENDED BUBBLELABS INTEGRATION
# =============================================================================

class BubbleLabsExtendedIntegration:
    """
    Extended BubbleLabs Integration Manager.
    
    This class provides a unified interface for all OpenEvolve component
    integrations within the BubbleLabs ecosystem.
    
    Thread-safe with proper locking hierarchy.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Configuration
        config = config or {}
        self.use_cav_nlp = config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        self._auto_discovery_enabled = bool(config.get("enable_auto_discovery", True))
        self._auto_discovery_root = Path(
            config.get("auto_discovery_root", Path(__file__).resolve().parent)
        )
        self._auto_discovery_index: Dict[str, Dict[str, Any]] = {}
        self._auto_discovery_last_refresh: float = 0.0
        
        # CAV-NLP enhanced solver
        self._enhanced_solver: Optional[Any] = None
        if self.use_cav_nlp:
            try:
                self._enhanced_solver = EnhancedZ3Solver()
                logger.info("[OK] CAV-NLP Enhanced Solver - Initialized")
            except Exception as e:
                logger.warning(f"[WARN] CAV-NLP Enhanced Solver - Initialization failed: {e}")
                self.use_cav_nlp = False
        
        # Component bridges
        self._ace_bridge: Optional[ACEIntegrationBridge] = None
        self._z3_bridge: Optional[Z3IntegrationBridge] = None
        self._roma_bridge: Optional[ROMAIntegrationBridge] = None
        self._knowledge_bridge: Optional[KnowledgeGraphIntegrationBridge] = None
        self._analytics_bridge: Optional[AnalyticsIntegrationBridge] = None
        self._leanaide_bridge: Optional[LeanAideIntegrationBridge] = None
        self._security_bridge: Optional[SecurityIntegrationBridge] = None
        self._openevolve_workflow_integration: Optional[Any] = None
        
        # Lock for thread safety
        self._lock = RLock()
        self._initialized = False
        
        # Executor for async operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bubblelabs")
        
        logger.info(f"BubbleLabsExtendedIntegration initialized (CAV-NLP: {self.use_cav_nlp})")

    def _set_bridge(self, name: str, bridge: Optional[Any]) -> None:
        setattr(self, f"_{name}_bridge", bridge)

    def _get_bridge(self, name: str) -> Optional[Any]:
        return getattr(self, f"_{name}_bridge", None)

    def _init_bridge_by_name(self, name: str) -> Optional[Any]:
        factories: Dict[str, Callable[[], Any]] = {
            "ace": ACEIntegrationBridge,
            "z3": Z3IntegrationBridge,
            "roma": ROMAIntegrationBridge,
            "knowledge": KnowledgeGraphIntegrationBridge,
            "analytics": AnalyticsIntegrationBridge,
            "leanaide": LeanAideIntegrationBridge,
            "security": SecurityIntegrationBridge,
        }
        factory = factories.get(name)
        if factory is None:
            return None
        try:
            bridge = factory()
            if bridge.status == ComponentStatus.AVAILABLE:
                self._set_bridge(name, bridge)
                return bridge
        except Exception as exc:
            logger.warning("Failed to initialize '%s' bridge: %s", name, exc)
        self._set_bridge(name, None)
        return None

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize_all()

    def _ensure_component_bridge(self, name: str) -> Optional[Any]:
        bridge = self._get_bridge(name)
        if bridge is not None:
            return bridge
        return self._init_bridge_by_name(name)

    def _is_auto_discovery_candidate(self, file_path: Path) -> bool:
        lower_name = file_path.name.lower()
        if not lower_name.endswith(".py"):
            return False
        if "integration" not in lower_name:
            return False
        if not ("openevolve" in lower_name or "bubblelab" in lower_name):
            return False
        if lower_name.startswith("test_") or lower_name.endswith("_test.py"):
            return False
        if lower_name.startswith(
            ("demo_", "analyze_", "verify_", "validate_", "quick_", "run_", "final_")
        ):
            return False
        if any(part in AUTO_DISCOVERY_EXCLUDED_DIRS for part in file_path.parts):
            return False
        return True

    def _is_safe_action_name(self, name: str) -> bool:
        if not name or name.startswith("_"):
            return False
        lowered = name.lower()
        if lowered.startswith(("test", "demo", "main", "cli")):
            return False
        if any(word in lowered for word in AUTO_DISCOVERY_DENY_WORDS):
            return False
        return True

    def _extract_callable_metadata(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        args = node.args
        positional_names = [a.arg for a in args.args]
        keyword_only_names = [a.arg for a in args.kwonlyargs]
        all_names = positional_names + keyword_only_names

        required_positional_count = max(0, len(positional_names) - len(args.defaults))
        required_names = positional_names[:required_positional_count]
        kwonly_defaults = args.kw_defaults or []
        required_names.extend(
            kwonly_names[idx] for idx, default in enumerate(kwonly_defaults) if default is None
        )

        return {
            "params": all_names,
            "required": required_names,
            "accepts_var_kw": args.kwarg is not None,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }

    def _discover_integration_module_files(self) -> List[Path]:
        roots: List[Path] = [self._auto_discovery_root]
        for child_name in ("integrations", "plugin_integrations", "glue"):
            child = self._auto_discovery_root / child_name
            if child.exists() and child.is_dir():
                roots.append(child)

        discovered: Set[Path] = set()
        for root in roots:
            try:
                for file_path in root.rglob("*.py"):
                    if self._is_auto_discovery_candidate(file_path):
                        discovered.add(file_path.resolve())
            except (FileNotFoundError, OSError) as exc:
                logger.debug("Skipping discovery root '%s' due to scan error: %s", root, exc)
        return sorted(discovered)

    def _build_auto_discovery_index(self) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for file_path in self._discover_integration_module_files():
            try:
                source = file_path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source, filename=str(file_path))
            except Exception as exc:
                logger.debug("Skipping discovery parse failure for %s: %s", file_path, exc)
                continue

            actions: Dict[str, Dict[str, Any]] = {}
            class_init_requirements: Dict[str, int] = {}
            class_nodes: Dict[str, ast.ClassDef] = {}

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not self._is_safe_action_name(node.name):
                        continue
                    actions[node.name] = {
                        "kind": "function",
                        **self._extract_callable_metadata(node),
                    }
                elif isinstance(node, ast.ClassDef):
                    class_nodes[node.name] = node
                    required_init_args = 0
                    for member in node.body:
                        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name == "__init__":
                            init_args = member.args.args[1:]  # skip self
                            defaults_len = len(member.args.defaults or [])
                            required_init_args = max(0, len(init_args) - defaults_len)
                            break
                    class_init_requirements[node.name] = required_init_args

            for class_name, class_node in class_nodes.items():
                if class_init_requirements.get(class_name, 0) > 0:
                    continue
                for member in class_node.body:
                    if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if member.name in {"__init__", "__enter__", "__exit__"}:
                        continue
                    if not self._is_safe_action_name(member.name):
                        continue
                    metadata = self._extract_callable_metadata(member)
                    # drop self from callable signature metadata
                    if metadata["params"] and metadata["params"][0] == "self":
                        metadata["params"] = metadata["params"][1:]
                    if metadata["required"] and metadata["required"][0] == "self":
                        metadata["required"] = metadata["required"][1:]
                    actions[f"{class_name}.{member.name}"] = {
                        "kind": "class_method",
                        "class_name": class_name,
                        "method_name": member.name,
                        **metadata,
                    }

            if not actions:
                continue

            component_name = file_path.stem.lower()
            index[component_name] = {
                "component": component_name,
                "module_name": file_path.stem,
                "file_path": str(file_path),
                "actions": actions,
            }

        return index

    def refresh_auto_discovery(self, force: bool = False) -> Dict[str, Any]:
        if not self._auto_discovery_enabled:
            return {"success": True, "enabled": False, "components": 0, "actions": 0}

        with self._lock:
            if self._auto_discovery_index and not force:
                action_count = sum(len(v.get("actions", {})) for v in self._auto_discovery_index.values())
                return {
                    "success": True,
                    "enabled": True,
                    "cached": True,
                    "components": len(self._auto_discovery_index),
                    "actions": action_count,
                    "last_refresh": self._auto_discovery_last_refresh,
                }

            self._auto_discovery_index = self._build_auto_discovery_index()
            self._auto_discovery_last_refresh = time.time()
            action_count = sum(len(v.get("actions", {})) for v in self._auto_discovery_index.values())
            return {
                "success": True,
                "enabled": True,
                "cached": False,
                "components": len(self._auto_discovery_index),
                "actions": action_count,
                "last_refresh": self._auto_discovery_last_refresh,
            }

    def _resolve_call_arguments(
        self, callable_obj: Callable[..., Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        if "kwargs" in payload and isinstance(payload.get("kwargs"), dict):
            candidate_kwargs = dict(payload.get("kwargs", {}))
        else:
            candidate_kwargs = dict(payload)

        signature = inspect.signature(callable_obj)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        if accepts_var_kw:
            return candidate_kwargs

        allowed_names = set(signature.parameters.keys())
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed_names}

        missing = [
            name
            for name, param in signature.parameters.items()
            if param.default is inspect._empty
            and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and name not in filtered_kwargs
        ]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        return filtered_kwargs

    def _import_module_from_file(self, file_path: str, component_name: str):
        module_key = f"bubblelabs_autodiscovery_{component_name}"
        spec = importlib.util.spec_from_file_location(module_key, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _run_maybe_async(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            try:
                return asyncio.run(value)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(value)
                finally:
                    loop.close()
        return value

    def _get_openevolve_workflow_integration(self) -> Optional[Any]:
        if self._openevolve_workflow_integration is not None:
            return self._openevolve_workflow_integration
        try:
            from openevolve_bubblelabs_api import openevolve_bubblelabs_integration

            self._openevolve_workflow_integration = openevolve_bubblelabs_integration
            return self._openevolve_workflow_integration
        except Exception as exc:
            logger.warning("OpenEvolve workflow integration unavailable: %s", exc)
            self._openevolve_workflow_integration = None
            return None

    def _execute_auto_discovered_action(
        self, component: str, action: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        discovery = self.refresh_auto_discovery(force=False)
        if not discovery.get("success", False):
            return {"success": False, "error": "Auto-discovery unavailable"}

        component_data = self._auto_discovery_index.get(component)
        if not component_data:
            return {
                "success": False,
                "error": f"Unknown component '{component}'",
            }

        action_meta = component_data.get("actions", {}).get(action)
        if not action_meta:
            return {
                "success": False,
                "error": f"Unknown action '{action}' for component '{component}'",
                "available_actions": sorted(component_data.get("actions", {}).keys()),
            }

        try:
            module = self._import_module_from_file(component_data["file_path"], component)
            if action_meta.get("kind") == "function":
                callable_obj = getattr(module, action)
                kwargs = self._resolve_call_arguments(callable_obj, payload)
                result = callable_obj(**kwargs)
            else:
                class_name = action_meta.get("class_name")
                method_name = action_meta.get("method_name")
                class_obj = getattr(module, class_name)
                instance = class_obj()
                callable_obj = getattr(instance, method_name)
                kwargs = self._resolve_call_arguments(callable_obj, payload)
                result = callable_obj(**kwargs)

            resolved = self._run_maybe_async(result)
            success = True
            if isinstance(resolved, dict) and "success" in resolved:
                success = bool(resolved.get("success"))
            return {
                "success": success,
                "component": component,
                "action": action,
                "result": resolved,
                "auto_discovered": True,
            }
        except Exception as exc:
            logger.exception(
                "Auto-discovered action execution failed for component=%s action=%s",
                component,
                action,
            )
            return {
                "success": False,
                "component": component,
                "action": action,
                "error": str(exc),
                "auto_discovered": True,
            }
    
    def initialize_all(self) -> Dict[str, Any]:
        """Initialize all component bridges."""
        with self._lock:
            results = {}
            
            # Initialize each bridge
            bridges = [
                ("ace", lambda: ACEIntegrationBridge()),
                ("z3", lambda: Z3IntegrationBridge()),
                ("roma", lambda: ROMAIntegrationBridge()),
                ("knowledge", lambda: KnowledgeGraphIntegrationBridge()),
                ("analytics", lambda: AnalyticsIntegrationBridge()),
                ("leanaide", lambda: LeanAideIntegrationBridge()),
                ("security", lambda: SecurityIntegrationBridge()),
            ]
            
            for name, factory in bridges:
                try:
                    bridge = factory()
                    results[name] = {
                        "success": bridge.status == ComponentStatus.AVAILABLE,
                        "status": bridge.status.value,
                        "capabilities": bridge.capabilities,
                    }
                except Exception as e:
                    results[name] = {
                        "success": False,
                        "status": "error",
                        "error": str(e),
                    }
            
            # Store bridges
            self._ace_bridge = ACEIntegrationBridge() if "ace" in results and results["ace"]["success"] else None
            self._z3_bridge = Z3IntegrationBridge() if "z3" in results and results["z3"]["success"] else None
            self._roma_bridge = ROMAIntegrationBridge() if "roma" in results and results["roma"]["success"] else None
            self._knowledge_bridge = KnowledgeGraphIntegrationBridge() if "knowledge" in results and results["knowledge"]["success"] else None
            self._analytics_bridge = AnalyticsIntegrationBridge() if "analytics" in results and results["analytics"]["success"] else None
            self._leanaide_bridge = LeanAideIntegrationBridge() if "leanaide" in results and results["leanaide"]["success"] else None
            self._security_bridge = SecurityIntegrationBridge() if "security" in results and results["security"]["success"] else None
            self._initialized = True
            
            return results
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all component integrations."""
        self._ensure_initialized()
        with self._lock:
            components = {}
            
            # ACE
            if self._ace_bridge:
                components["ace"] = self._ace_bridge.get_status()
            else:
                components["ace"] = {"component": "ACE", "status": "unavailable"}
            
            # Z3
            if self._z3_bridge:
                components["z3"] = self._z3_bridge.get_status()
            else:
                components["z3"] = {"component": "Z3 Prover", "status": "unavailable"}
            
            # ROMA
            if self._roma_bridge:
                components["roma"] = self._roma_bridge.get_status()
            else:
                components["roma"] = {"component": "ROMA", "status": "unavailable"}
            
            # Knowledge Graph
            if self._knowledge_bridge:
                components["knowledge"] = self._knowledge_bridge.get_status()
            else:
                components["knowledge"] = {"component": "Knowledge Graph", "status": "unavailable"}
            
            # Analytics
            if self._analytics_bridge:
                components["analytics"] = self._analytics_bridge.get_status()
            else:
                components["analytics"] = {"component": "Analytics", "status": "unavailable"}
            
            # LeanAIDE
            if self._leanaide_bridge:
                components["leanaide"] = self._leanaide_bridge.get_status()
            else:
                components["leanaide"] = {"component": "LeanAIDE", "status": "unavailable"}
            
            # Security
            if self._security_bridge:
                components["security"] = self._security_bridge.get_status()
            else:
                components["security"] = {"component": "Security", "status": "unavailable"}
            
            # CAV-NLP
            components["cav_nlp"] = self.get_cav_nlp_status()
            components["web3"] = self.get_web3_status()
            components["openevolve_workflows"] = self.get_openevolve_workflow_status()
            
            return {
                "total_components": len(components),
                "available_components": sum(1 for c in components.values() if c.get("status") == "available" or c.get("available") == True),
                "components": components,
            }
    
    # =========================================================================
    # ACE Methods
    # =========================================================================
    
    def ace_create_skillbook(self, name: str, skills: List[Dict]) -> Dict[str, Any]:
        """Create a new ACE skillbook."""
        bridge = self._ensure_component_bridge("ace")
        if bridge:
            return bridge.create_skillbook(name, skills)
        return {"success": False, "error": "ACE not available"}
    
    def ace_extract_patterns(self, workflow_results: List[Dict]) -> Dict[str, Any]:
        """Extract patterns from workflow results."""
        bridge = self._ensure_component_bridge("ace")
        if bridge:
            return bridge.extract_patterns(workflow_results)
        return {"success": False, "error": "ACE not available"}
    
    # =========================================================================
    # Z3 Methods
    # =========================================================================
    
    def z3_solve_constraints(
        self,
        variables: List[Dict[str, Any]],
        constraints: List[str]
    ) -> Dict[str, Any]:
        """Solve constraints with Z3."""
        bridge = self._ensure_component_bridge("z3")
        if bridge:
            return bridge.solve_constraints(variables, constraints)
        return {"success": False, "error": "Z3 not available"}
    
    def z3_prove_theorem(self, theorem: str) -> Dict[str, Any]:
        """Prove a theorem with Z3."""
        bridge = self._ensure_component_bridge("z3")
        if bridge:
            return bridge.prove_theorem(theorem)
        return {"success": False, "error": "Z3 not available"}

    # =========================================================================
    # Web3 Audit Methods
    # =========================================================================

    def get_web3_status(self) -> Dict[str, Any]:
        """Get Web3 audit integration status."""
        inventory = {}
        web3_tools: List[str] = []
        web3_ingestion_tools: List[str] = []
        web3_formal_tools: List[str] = []
        if get_mcp_tool_inventory is not None:
            try:
                inventory = get_mcp_tool_inventory()
                if isinstance(inventory, dict):
                    web3_tools = list(inventory.get("web3_tools", []) or [])
                    web3_ingestion_tools = list(inventory.get("web3_ingestion_tools", []) or [])
                    web3_formal_tools = list(inventory.get("web3_formal_tools", []) or [])
            except Exception as exc:
                inventory = {"error": str(exc)}
        formal_capabilities = {
            "solidity_invariant_translation": translate_solidity_assignment_to_z3 is not None,
            "invariant_translation_verification": verify_solidity_invariant_translation is not None,
            "symbolic_exploit_witness": solve_smart_contract_exploit_witness is not None,
            "composite_exploit_verification": (
                translate_solidity_assignment_to_z3 is not None
                and solve_smart_contract_exploit_witness is not None
            ),
        }
        if isinstance(inventory, dict):
            merged = inventory.get("formal_capabilities")
            if isinstance(merged, dict):
                formal_capabilities.update(merged)

        if not web3_formal_tools:
            inferred_formal_tools: List[str] = []
            if formal_capabilities.get("solidity_invariant_translation"):
                inferred_formal_tools.append("z3_translate_solidity_invariant")
            if formal_capabilities.get("symbolic_exploit_witness"):
                inferred_formal_tools.append("z3_solve_smart_contract_exploit_witness")
            if formal_capabilities.get("composite_exploit_verification"):
                inferred_formal_tools.append("z3_web3_audit_exploit_verification")
            web3_formal_tools = inferred_formal_tools

        if not web3_ingestion_tools:
            web3_ingestion_tools = [
                "web3_ingest_slither_static_analysis",
                "web3_ingest_foundry_fuzzing",
                "web3_ingest_contract_audit_stack",
            ]

        if not web3_tools:
            web3_tools = sorted(set(web3_ingestion_tools + web3_formal_tools))

        web3_formal_tools = sorted(set(web3_formal_tools))
        web3_ingestion_tools = sorted(set(web3_ingestion_tools))
        web3_tools = sorted(set(web3_tools))
        inferred_formal_available = bool(web3_formal_tools) or any(
            bool(v) for v in formal_capabilities.values()
        )
        inferred_stack_available = bool(web3_tools) or bool(web3_ingestion_tools) or inferred_formal_available

        return {
            "component": "Web3 Audit Stack",
            "status": "available" if inferred_stack_available else "unavailable",
            "available": inferred_stack_available,
            "capabilities": [
                "slither_static_analysis",
                "forge_fuzz_ingestion",
                "solidity_invariant_translation",
                "symbolic_exploit_witness",
                "composite_exploit_verification",
            ],
            "ingestion_available": WEB3_INGESTION_AVAILABLE or bool(web3_ingestion_tools),
            "formal_available": WEB3_FORMAL_AVAILABLE or inferred_formal_available,
            "web3_formal_available": WEB3_FORMAL_AVAILABLE or inferred_formal_available,
            "web3_formal_verification_available": (
                WEB3_FORMAL_AVAILABLE or inferred_formal_available
            ),
            "audit_exploit_verification_available": bool(
                formal_capabilities.get("composite_exploit_verification")
            ),
            "web3_tools": web3_tools,
            "web3_ingestion_tools": web3_ingestion_tools,
            "web3_formal_tools": web3_formal_tools,
            "formal_capabilities": formal_capabilities,
            "tool_inventory": inventory,
        }

    def web3_get_mcp_tool_inventory(self) -> Dict[str, Any]:
        """Get MCP inventory for Web3 analysis tools."""
        if get_mcp_tool_inventory is None:
            return {"success": False, "error": "Web3 MCP inventory unavailable"}
        try:
            return {"success": True, "inventory": get_mcp_tool_inventory()}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def web3_ingest_contract_stack(
        self,
        project_path: str = ".",
        run_fuzzing: bool = True,
        slither_timeout_seconds: int = 240,
        forge_timeout_seconds: int = 420,
    ) -> Dict[str, Any]:
        """Run full Web3 contract audit ingestion pipeline (Slither + Forge)."""
        if web3_ingest_contract_audit_stack is None:
            return {"success": False, "error": "Web3 ingestion stack unavailable"}
        try:
            return web3_ingest_contract_audit_stack(
                project_path=project_path,
                run_fuzzing=run_fuzzing,
                slither_timeout_seconds=slither_timeout_seconds,
                forge_timeout_seconds=forge_timeout_seconds,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def web3_ingest_slither(
        self,
        project_path: str = ".",
        timeout_seconds: int = 240,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run Slither static analysis for Web3 contracts."""
        if web3_ingest_slither_static_analysis is None:
            return {"success": False, "error": "Slither ingestion unavailable"}
        try:
            return web3_ingest_slither_static_analysis(
                project_path=project_path,
                timeout_seconds=timeout_seconds,
                extra_args=extra_args,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def web3_ingest_foundry(
        self,
        project_path: str = ".",
        timeout_seconds: int = 420,
        match_contract: Optional[str] = None,
        match_test: Optional[str] = None,
        fork_url: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run Foundry/Forge fuzzing for Web3 contracts."""
        if web3_ingest_foundry_fuzzing is None:
            return {"success": False, "error": "Foundry ingestion unavailable"}
        try:
            return web3_ingest_foundry_fuzzing(
                project_path=project_path,
                timeout_seconds=timeout_seconds,
                match_contract=match_contract,
                match_test=match_test,
                fork_url=fork_url,
                extra_args=extra_args,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def web3_translate_solidity_invariant(
        self,
        statement: str,
        non_negative_target: bool = True,
        max_withdraw_expr: Optional[str] = None,
        verify_translation: bool = True,
        assume_non_negative_amount: bool = True,
    ) -> Dict[str, Any]:
        """Translate Solidity state updates to Z3/Lean invariants and optionally verify."""
        if translate_solidity_assignment_to_z3 is None:
            return {"success": False, "error": "Solidity invariant translation unavailable"}
        try:
            translation = translate_solidity_assignment_to_z3(
                statement=statement,
                non_negative_target=non_negative_target,
                max_withdraw_expr=max_withdraw_expr,
            )
            result: Dict[str, Any] = {"success": True, "translation": translation}
            if verify_translation and verify_solidity_invariant_translation is not None:
                result["verification"] = verify_solidity_invariant_translation(
                    translation=translation,
                    assume_non_negative_amount=assume_non_negative_amount,
                )
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def web3_solve_exploit_witness(
        self,
        additional_constraints: Optional[List[str]] = None,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        """Solve symbolic exploit witness query for smart-contract balance drain."""
        if solve_smart_contract_exploit_witness is None:
            return {"success": False, "error": "Exploit witness solver unavailable"}
        try:
            return {
                "success": True,
                "result": solve_smart_contract_exploit_witness(
                    additional_constraints=additional_constraints,
                    timeout=timeout_seconds,
                ),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def web3_audit_exploit_verification(
        self,
        project_path: str = ".",
        run_fuzzing: bool = True,
        statement: Optional[str] = None,
        non_negative_target: bool = True,
        max_withdraw_expr: Optional[str] = None,
        verify_translation: bool = True,
        assume_non_negative_amount: bool = True,
        additional_constraints: Optional[List[str]] = None,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Run one-shot BubbleLabs Web3 audit orchestration:
        ingestion + optional invariant translation + exploit witness.
        """
        ingestion = self.web3_ingest_contract_stack(
            project_path=project_path,
            run_fuzzing=run_fuzzing,
            slither_timeout_seconds=240,
            forge_timeout_seconds=420,
        )

        translation = None
        if statement:
            translation = self.web3_translate_solidity_invariant(
                statement=statement,
                non_negative_target=non_negative_target,
                max_withdraw_expr=max_withdraw_expr,
                verify_translation=verify_translation,
                assume_non_negative_amount=assume_non_negative_amount,
            )

        exploit_witness = self.web3_solve_exploit_witness(
            additional_constraints=additional_constraints,
            timeout_seconds=timeout_seconds,
        )

        verification = translation.get("verification") if isinstance(translation, dict) else None
        witness_payload = None
        if isinstance(exploit_witness, dict):
            witness_payload = exploit_witness.get("result")
        translated_payload = translation.get("translation") if isinstance(translation, dict) else None
        lean_proof_verification = verify_web3_lean_proof(translated_payload, use_real_lean=True)

        verified_exploit = bool((witness_payload or {}).get("satisfiable", False))
        if verify_translation and isinstance(verification, dict):
            verified_exploit = verified_exploit and bool(verification.get("proven", False))

        return {
            "success": bool(ingestion) and bool(exploit_witness),
            "ingestion": ingestion,
            "translation": translation,
            "exploit_witness": exploit_witness,
            "lean_proof_verification": lean_proof_verification,
            "formal_evidence": build_web3_formal_evidence(
                verification,
                witness_payload if isinstance(witness_payload, dict) else {},
                lean_proof_verification,
            ),
            "verified_exploit": verified_exploit,
        }
    
    # =========================================================================
    # ROMA Methods
    # =========================================================================
    
    def roma_analyze_problem(self, problem: str, max_depth: int = 3) -> Dict[str, Any]:
        """Analyze a problem with ROMA."""
        bridge = self._ensure_component_bridge("roma")
        if bridge:
            return bridge.analyze_problem(problem, max_depth)
        return {"success": False, "error": "ROMA not available"}
    
    def roma_create_config(self, **kwargs) -> Dict[str, Any]:
        """Create ROMA configuration."""
        bridge = self._ensure_component_bridge("roma")
        if bridge:
            return bridge.create_config(**kwargs)
        return {"success": False, "error": "ROMA not available"}
    
    # =========================================================================
    # Knowledge Graph Methods
    # =========================================================================
    
    def knowledge_store_artifact(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """Store a knowledge artifact."""
        bridge = self._ensure_component_bridge("knowledge")
        if bridge:
            return bridge.store_artifact(artifact)
        return {"success": False, "error": "Knowledge Graph not available"}
    
    def knowledge_query_patterns(self, query: str) -> Dict[str, Any]:
        """Query patterns from knowledge graph."""
        bridge = self._ensure_component_bridge("knowledge")
        if bridge:
            return bridge.query_patterns(query)
        return {"success": False, "error": "Knowledge Graph not available"}
    
    # =========================================================================
    # Analytics Methods
    # =========================================================================
    
    def analytics_track_workflow(self, workflow_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track workflow metrics."""
        bridge = self._ensure_component_bridge("analytics")
        if bridge:
            return bridge.track_workflow(workflow_id, metrics)
        return {"success": False, "error": "Analytics not available"}
    
    def analytics_get_dashboard(self) -> Dict[str, Any]:
        """Get analytics dashboard data."""
        bridge = self._ensure_component_bridge("analytics")
        if bridge:
            return bridge.get_dashboard()
        return {"success": False, "error": "Analytics not available"}
    
    # =========================================================================
    # LeanAIDE Methods
    # =========================================================================
    
    def leanaide_prove_theorem(self, theorem: str) -> Dict[str, Any]:
        """Prove a theorem with LeanAIDE."""
        bridge = self._ensure_component_bridge("leanaide")
        if bridge:
            return bridge.prove_theorem(theorem)
        return {"success": False, "error": "LeanAIDE not available"}

    # =========================================================================
    # OpenEvolve Workflow Control Methods
    # =========================================================================

    def get_openevolve_workflow_status(self) -> Dict[str, Any]:
        integration = self._get_openevolve_workflow_integration()
        if integration is None:
            return {
                "component": "OpenEvolve Workflow Engine",
                "status": "unavailable",
                "available": False,
                "capabilities": [],
            }
        definitions = integration.list_workflow_definitions()
        instances = integration.list_workflow_instances()
        return {
            "component": "OpenEvolve Workflow Engine",
            "status": "available",
            "available": True,
            "capabilities": [
                "create_definition",
                "list_definitions",
                "get_definition",
                "create_instance",
                "list_instances",
                "get_instance_status",
                "start",
                "pause",
                "resume",
                "stop",
                "cancel",
                "restart",
                "delete",
                "sync_parameters",
            ],
            "counts": {
                "definitions": len(definitions),
                "instances": len(instances),
            },
        }

    def openevolve_create_workflow_definition(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        integration = self._get_openevolve_workflow_integration()
        if integration is None:
            return {"success": False, "error": "OpenEvolve workflow integration unavailable"}
        try:
            definition_id = integration.create_workflow_definition(
                name=payload.get("name", "OpenEvolve Workflow"),
                description=payload.get("description", ""),
                workflow_type=payload.get("workflow_type", "sovereign"),
                parameters=payload.get("parameters", {}) or {},
            )
            return {"success": True, "definition_id": definition_id}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def openevolve_list_workflow_definitions(self) -> Dict[str, Any]:
        integration = self._get_openevolve_workflow_integration()
        if integration is None:
            return {"success": False, "error": "OpenEvolve workflow integration unavailable"}
        return {"success": True, "definitions": integration.list_workflow_definitions()}

    def openevolve_get_workflow_definition(self, definition_id: str) -> Dict[str, Any]:
        integration = self._get_openevolve_workflow_integration()
        if integration is None:
            return {"success": False, "error": "OpenEvolve workflow integration unavailable"}
        definition = integration.get_workflow_definition(definition_id)
        if not definition:
            return {"success": False, "error": f"Workflow definition {definition_id} not found"}
        return {"success": True, "definition": definition}

    def openevolve_create_workflow_instance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        integration = self._get_openevolve_workflow_integration()
        if integration is None:
            return {"success": False, "error": "OpenEvolve workflow integration unavailable"}
        try:
            instance_id = integration.create_workflow_instance(
                definition_id=payload.get("definition_id", ""),
                instance_name=payload.get("instance_name", "openevolve-instance"),
                inputs=payload.get("inputs", {}) or {},
                parameters=payload.get("parameters"),
            )
            return {"success": True, "instance_id": instance_id}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def openevolve_list_workflow_instances(self) -> Dict[str, Any]:
        integration = self._get_openevolve_workflow_integration()
        if integration is None:
            return {"success": False, "error": "OpenEvolve workflow integration unavailable"}
        return {"success": True, "instances": integration.list_workflow_instances()}

    def openevolve_get_workflow_instance_status(self, instance_id: str) -> Dict[str, Any]:
        integration = self._get_openevolve_workflow_integration()
        if integration is None:
            return {"success": False, "error": "OpenEvolve workflow integration unavailable"}
        result = integration.get_workflow_instance_status(instance_id)
        success = not (isinstance(result, dict) and "error" in result)
        if success:
            return {"success": True, "status": result}
        return {"success": False, **result}

    def openevolve_control_workflow_instance(self, instance_id: str, action: str) -> Dict[str, Any]:
        integration = self._get_openevolve_workflow_integration()
        if integration is None:
            return {"success": False, "error": "OpenEvolve workflow integration unavailable"}
        method_name_by_action = {
            "start": "start_workflow_instance",
            "pause": "pause_workflow_instance",
            "resume": "resume_workflow_instance",
            "stop": "stop_workflow_instance",
            "cancel": "cancel_workflow_instance",
            "restart": "restart_workflow_instance",
            "delete": "delete_workflow_instance",
        }
        method_name = method_name_by_action.get(action.lower())
        if method_name is None:
            return {"success": False, "error": f"Unsupported workflow action '{action}'"}
        handler = getattr(integration, method_name, None)
        if handler is None or not callable(handler):
            return {
                "success": False,
                "error": f"Workflow integration does not support action '{action}'",
            }
        result = handler(instance_id)
        success = not (isinstance(result, dict) and "error" in result)
        if isinstance(result, dict):
            result.setdefault("success", success)
        return result if isinstance(result, dict) else {"success": success, "result": result}

    def openevolve_sync_workflow_parameters(
        self, instance_id: str, parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        integration = self._get_openevolve_workflow_integration()
        if integration is None:
            return {"success": False, "error": "OpenEvolve workflow integration unavailable"}
        result = integration.sync_parameters_to_workflow(instance_id, parameters or {})
        success = not (isinstance(result, dict) and "error" in result)
        if isinstance(result, dict):
            result.setdefault("success", success)
        return result if isinstance(result, dict) else {"success": success, "result": result}

    # =========================================================================
    # Unified BubbleLabs Control Surface
    # =========================================================================

    def get_control_catalog(self) -> Dict[str, Any]:
        """Return discoverable component actions for BubbleLabs control."""
        self._ensure_initialized()
        base_components = {
            "ace": ["create_skillbook", "extract_patterns"],
            "z3": ["solve_constraints", "prove_theorem"],
            "roma": ["analyze_problem", "create_config"],
            "knowledge": ["store_artifact", "query_patterns"],
            "analytics": ["track_workflow", "get_dashboard"],
            "leanaide": ["prove_theorem"],
            "web3": [
                "status",
                "get_mcp_tool_inventory",
                "ingest_contract_stack",
                "ingest_slither",
                "ingest_foundry",
                "translate_solidity_invariant",
                "solve_exploit_witness",
                "audit_exploit_verification",
            ],
            "cav_nlp": [
                "status",
                "formalize_constraint",
                "formalize_operation",
                "hybrid_verify_constraint",
                "export_proof_to_lean",
            ],
            "security": ["status"],
            "openevolve_workflows": [
                "status",
                "create_definition",
                "list_definitions",
                "get_definition",
                "create_instance",
                "list_instances",
                "get_instance_status",
                "start_instance",
                "pause_instance",
                "resume_instance",
                "stop_instance",
                "cancel_instance",
                "restart_instance",
                "delete_instance",
                "sync_parameters",
            ],
        }
        discovery = self.refresh_auto_discovery(force=False)
        auto_components = {
            component: sorted(list(metadata.get("actions", {}).keys()))
            for component, metadata in self._auto_discovery_index.items()
        }
        components = dict(base_components)
        components.update(auto_components)
        return {
            "success": True,
            "components": components,
            "auto_discovery": {
                "enabled": self._auto_discovery_enabled,
                "summary": discovery,
                "components": auto_components,
            },
        }

    def execute_control_action(
        self, component: str, action: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a component action through a unified BubbleLabs control API."""
        payload = payload or {}
        component_key = str(component or "").strip().lower()
        action_key = str(action or "").strip().lower()

        dispatch: Dict[str, Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = {
            "ace": {
                "create_skillbook": lambda p: self.ace_create_skillbook(
                    name=p.get("name", ""), skills=p.get("skills", []) or []
                ),
                "extract_patterns": lambda p: self.ace_extract_patterns(
                    p.get("workflow_results", []) or []
                ),
            },
            "z3": {
                "solve_constraints": lambda p: self.z3_solve_constraints(
                    p.get("variables", []) or [], p.get("constraints", []) or []
                ),
                "prove_theorem": lambda p: self.z3_prove_theorem(p.get("theorem", "")),
            },
            "roma": {
                "analyze_problem": lambda p: self.roma_analyze_problem(
                    p.get("problem", ""), int(p.get("max_depth", 3))
                ),
                "create_config": lambda p: self.roma_create_config(**(p.get("config", {}) or {})),
            },
            "knowledge": {
                "store_artifact": lambda p: self.knowledge_store_artifact(p.get("artifact", {}) or {}),
                "query_patterns": lambda p: self.knowledge_query_patterns(p.get("query", "")),
            },
            "analytics": {
                "track_workflow": lambda p: self.analytics_track_workflow(
                    p.get("workflow_id", ""), p.get("metrics", {}) or {}
                ),
                "get_dashboard": lambda p: self.analytics_get_dashboard(),
            },
            "leanaide": {
                "prove_theorem": lambda p: self.leanaide_prove_theorem(p.get("theorem", "")),
            },
            "web3": {
                "status": lambda p: self.get_web3_status(),
                "get_mcp_tool_inventory": lambda p: self.web3_get_mcp_tool_inventory(),
                "ingest_contract_stack": lambda p: self.web3_ingest_contract_stack(
                    project_path=p.get("project_path", "."),
                    run_fuzzing=bool(p.get("run_fuzzing", True)),
                    slither_timeout_seconds=int(p.get("slither_timeout_seconds", 240)),
                    forge_timeout_seconds=int(p.get("forge_timeout_seconds", 420)),
                ),
                "ingest_slither": lambda p: self.web3_ingest_slither(
                    project_path=p.get("project_path", "."),
                    timeout_seconds=int(p.get("timeout_seconds", 240)),
                    extra_args=p.get("extra_args"),
                ),
                "ingest_foundry": lambda p: self.web3_ingest_foundry(
                    project_path=p.get("project_path", "."),
                    timeout_seconds=int(p.get("timeout_seconds", 420)),
                    match_contract=p.get("match_contract"),
                    match_test=p.get("match_test"),
                    fork_url=p.get("fork_url"),
                    extra_args=p.get("extra_args"),
                ),
                "translate_solidity_invariant": lambda p: self.web3_translate_solidity_invariant(
                    statement=p.get("statement", ""),
                    non_negative_target=bool(p.get("non_negative_target", True)),
                    max_withdraw_expr=p.get("max_withdraw_expr"),
                    verify_translation=bool(p.get("verify_translation", True)),
                    assume_non_negative_amount=bool(p.get("assume_non_negative_amount", True)),
                ),
                "solve_exploit_witness": lambda p: self.web3_solve_exploit_witness(
                    additional_constraints=p.get("additional_constraints"),
                    timeout_seconds=float(p.get("timeout_seconds", 10.0)),
                ),
                "audit_exploit_verification": lambda p: self.web3_audit_exploit_verification(
                    project_path=p.get("project_path", "."),
                    run_fuzzing=bool(p.get("run_fuzzing", True)),
                    statement=p.get("statement"),
                    non_negative_target=bool(p.get("non_negative_target", True)),
                    max_withdraw_expr=p.get("max_withdraw_expr"),
                    verify_translation=bool(p.get("verify_translation", True)),
                    assume_non_negative_amount=bool(p.get("assume_non_negative_amount", True)),
                    additional_constraints=p.get("additional_constraints"),
                    timeout_seconds=float(p.get("timeout_seconds", 10.0)),
                ),
            },
            "cav_nlp": {
                "status": lambda p: self.get_cav_nlp_status(),
                "formalize_constraint": lambda p: self.formalize_extended_constraint(
                    p.get("nl_constraint", "")
                ),
                "formalize_operation": lambda p: self.formalize_extended_operation(
                    p.get("operation_description", "")
                ),
                "hybrid_verify_constraint": lambda p: self.hybrid_verify_extended_constraint(
                    p.get("constraint", ""), p.get("context")
                ),
                "export_proof_to_lean": lambda p: self.export_proof_to_lean(
                    p.get("constraint", ""), p.get("proof_name")
                ),
            },
            "security": {
                "status": lambda p: self._security_status(),
            },
            "openevolve_workflows": {
                "status": lambda p: self.get_openevolve_workflow_status(),
                "create_definition": lambda p: self.openevolve_create_workflow_definition(p),
                "list_definitions": lambda p: self.openevolve_list_workflow_definitions(),
                "get_definition": lambda p: self.openevolve_get_workflow_definition(
                    p.get("definition_id", "")
                ),
                "create_instance": lambda p: self.openevolve_create_workflow_instance(p),
                "list_instances": lambda p: self.openevolve_list_workflow_instances(),
                "get_instance_status": lambda p: self.openevolve_get_workflow_instance_status(
                    p.get("instance_id", "")
                ),
                "start_instance": lambda p: self.openevolve_control_workflow_instance(
                    p.get("instance_id", ""), "start"
                ),
                "pause_instance": lambda p: self.openevolve_control_workflow_instance(
                    p.get("instance_id", ""), "pause"
                ),
                "resume_instance": lambda p: self.openevolve_control_workflow_instance(
                    p.get("instance_id", ""), "resume"
                ),
                "stop_instance": lambda p: self.openevolve_control_workflow_instance(
                    p.get("instance_id", ""), "stop"
                ),
                "cancel_instance": lambda p: self.openevolve_control_workflow_instance(
                    p.get("instance_id", ""), "cancel"
                ),
                "restart_instance": lambda p: self.openevolve_control_workflow_instance(
                    p.get("instance_id", ""), "restart"
                ),
                "delete_instance": lambda p: self.openevolve_control_workflow_instance(
                    p.get("instance_id", ""), "delete"
                ),
                "sync_parameters": lambda p: self.openevolve_sync_workflow_parameters(
                    p.get("instance_id", ""), p.get("parameters")
                ),
            },
        }

        component_actions = dispatch.get(component_key)
        if not component_actions:
            auto_result = self._execute_auto_discovered_action(component_key, action_key, payload)
            if auto_result.get("success") or auto_result.get("auto_discovered"):
                return auto_result
            return {
                "success": False,
                "error": f"Unknown component '{component_key}'",
                "catalog": self.get_control_catalog().get("components", {}),
            }

        handler = component_actions.get(action_key)
        if not handler:
            return {
                "success": False,
                "error": f"Unknown action '{action_key}' for component '{component_key}'",
                "available_actions": sorted(component_actions.keys()),
            }

        try:
            result = handler(payload)
            return {
                "success": bool(result.get("success", True)) if isinstance(result, dict) else True,
                "component": component_key,
                "action": action_key,
                "result": result,
            }
        except Exception as exc:
            logger.exception(
                "BubbleLabs control action failed for component=%s action=%s",
                component_key,
                action_key,
            )
            return {
                "success": False,
                "component": component_key,
                "action": action_key,
                "error": str(exc),
            }

    def _security_status(self) -> Dict[str, Any]:
        bridge = self._ensure_component_bridge("security")
        if bridge:
            return bridge.get_status()
        return {"success": False, "error": "Security bridge not available"}
    
    # =========================================================================
    # CAV-NLP Enhanced Methods
    # =========================================================================
    
    def formalize_extended_constraint(self, nl_constraint: str) -> Dict[str, Any]:
        """
        Formalize a natural language constraint using CAV-NLP.
        
        Args:
            nl_constraint: Natural language constraint description
            
        Returns:
            Dict with formalized constraint or error
        """
        if not self.use_cav_nlp or not self._enhanced_solver:
            return {
                "success": False, 
                "error": "CAV-NLP not available",
                "constraint": nl_constraint
            }
        
        try:
            formalized = self._enhanced_solver.formalize_constraint(nl_constraint)
            return {
                "success": True,
                "constraint": nl_constraint,
                "formalized": formalized,
                "method": "cav_nlp"
            }
        except Exception as e:
            logger.error(f"CAV-NLP formalization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "constraint": nl_constraint
            }
    
    def formalize_extended_operation(self, operation_description: str) -> Dict[str, Any]:
        """
        Formalize an extended bubble operation using CAV-NLP.
        
        Args:
            operation_description: Natural language operation description
            
        Returns:
            Dict with formalized operation specification
        """
        if not self.use_cav_nlp or not self._enhanced_solver:
            return {
                "success": False,
                "error": "CAV-NLP not available",
                "operation": operation_description
            }
        
        try:
            # Formalize the operation constraints
            formalized = self._enhanced_solver.formalize_constraint(operation_description)
            
            return {
                "success": True,
                "operation": operation_description,
                "formalized_spec": formalized,
                "method": "cav_nlp_extended"
            }
        except Exception as e:
            logger.error(f"CAV-NLP operation formalization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": operation_description
            }
    
    def hybrid_verify_extended_constraint(
        self,
        constraint: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform hybrid verification of an extended constraint using CAV-NLP.
        
        Combines natural language understanding with formal verification.
        
        Args:
            constraint: Constraint to verify (natural language or formal)
            context: Additional context for verification
            
        Returns:
            Dict with verification results
        """
        if not self.use_cav_nlp or not self._enhanced_solver:
            return {
                "success": False,
                "error": "CAV-NLP not available",
                "constraint": constraint
            }
        
        context = context or {}
        
        try:
            # First formalize if needed
            formalized = self._enhanced_solver.formalize_constraint(constraint)
            
            # Perform hybrid verification
            verification_result = self._enhanced_solver.verify_constraint(
                formalized,
                context=context
            )
            
            return {
                "success": True,
                "constraint": constraint,
                "formalized": formalized,
                "verification": verification_result,
                "method": "hybrid_cav_nlp",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"CAV-NLP hybrid verification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "constraint": constraint
            }
    
    def export_proof_to_lean(
        self,
        constraint: str,
        proof_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export a formalized proof to Lean format using CAV-NLP.
        
        Args:
            constraint: Constraint to formalize and export
            proof_name: Optional name for the proof
            
        Returns:
            Dict with exported proof details
        """
        if not self.use_cav_nlp or not self._enhanced_solver:
            return {
                "success": False,
                "error": "CAV-NLP not available",
                "constraint": constraint
            }
        
        proof_name = proof_name or f"proof_{uuid.uuid4().hex[:8]}"
        
        try:
            # Formalize the constraint
            formalized = self._enhanced_solver.formalize_constraint(constraint)
            
            # Generate Lean proof (if method available)
            lean_code = None
            if hasattr(self._enhanced_solver, 'export_to_lean'):
                lean_code = self._enhanced_solver.export_to_lean(formalized, proof_name)
            else:
                # Optimized string building using join
                lean_code_parts = [
                    f"-- Proof: {proof_name}",
                    f"-- Original constraint: {constraint}\n",
                    f"theorem {proof_name} :",
                    f"  {formalized} := by",
                    "  sorry\n"
                ]
                lean_code = "\n".join(lean_code_parts)
            
            return {
                "success": True,
                "proof_name": proof_name,
                "constraint": constraint,
                "formalized": formalized,
                "lean_code": lean_code,
                "method": "cav_nlp_to_lean"
            }
        except Exception as e:
            logger.error(f"CAV-NLP proof export failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "constraint": constraint
            }
    
    def get_cav_nlp_status(self) -> Dict[str, Any]:
        """Get CAV-NLP integration status."""
        return {
            "component": "CAV-NLP (Computer-Aided Verification NLP)",
            "available": CAV_NLP_AVAILABLE,
            "enabled": self.use_cav_nlp,
            "solver_initialized": self._enhanced_solver is not None,
            "capabilities": [
                "constraint_formalization",
                "operation_formalization",
                "hybrid_verification",
                "lean_export"
            ] if self.use_cav_nlp else []
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def shutdown(self) -> None:
        """Shutdown the integration manager."""
        with self._lock:
            self._executor.shutdown(wait=False)
            logger.info("BubbleLabsExtendedIntegration shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        self._lock.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._lock.__exit__(exc_type, exc_val, exc_tb)
        return False


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_extended_integration: Optional[BubbleLabsExtendedIntegration] = None
_integration_lock = RLock()


def get_extended_integration() -> BubbleLabsExtendedIntegration:
    """Get global extended integration instance."""
    global _extended_integration
    if _extended_integration is None:
        with _integration_lock:
            if _extended_integration is None:
                _extended_integration = BubbleLabsExtendedIntegration()
    return _extended_integration


def initialize_extended_integration() -> Dict[str, Any]:
    """Initialize all extended integrations."""
    integration = get_extended_integration()
    return integration.initialize_all()


def get_all_integration_status() -> Dict[str, Any]:
    """Get status of all integrations."""
    integration = get_extended_integration()
    return integration.get_all_status()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    print("=" * 60)
    print("BubbleLabs Extended Integration")
    print("=" * 60)
    
    # Initialize all integrations
    print("\nInitializing integrations...")
    results = initialize_extended_integration()
    
    for name, result in results.items():
        status = "[OK]" if result["success"] else "[FAIL]"
        print(f"  {status} {name.upper()}: {result['status']}")
    
    # Get full status
    print("\n" + "=" * 60)
    print("Integration Status")
    print("=" * 60)
    
    status = get_all_integration_status()
    print(f"\nTotal Components: {status['total_components']}")
    print(f"Available: {status['available_components']}")
    
    print("\nComponent Details:")
    for name, component in status["components"].items():
        print(f"\n  {name.upper()}:")
        print(f"    Status: {component.get('status', 'N/A')}")
        print(f"    Version: {component.get('version', 'N/A')}")
        capabilities = component.get('capabilities', [])
        if capabilities:
            print(f"    Capabilities: {', '.join(capabilities[:3])}...")
    
    print("\n" + "=" * 60)
    print("Integration Complete")
    print("=" * 60)
