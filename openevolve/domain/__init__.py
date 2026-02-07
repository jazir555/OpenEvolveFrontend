"""
OpenEvolve Domain module
Re-exports domain optimizers from core-projects for unified interface
"""
import sys
from pathlib import Path
import importlib.util

# Add core-projects to Python path if not already there
# From __init__.py: domain -> openevolve -> Frontend -> core-projects/openevolve
core_projects_path = Path(__file__).parent.parent.parent / "core-projects" / "openevolve"
if str(core_projects_path) not in sys.path:
    sys.path.insert(0, str(core_projects_path))

# Import and re-export all domain optimizers and utilities from core-projects
try:
    # Import the core-projects domain module directly by file path to avoid circular import
    domain_module_path = core_projects_path / "openevolve" / "domain" / "__init__.py"

    spec = importlib.util.spec_from_file_location("openevolve_core_domain", domain_module_path)
    if spec and spec.loader:
        domain_module = importlib.util.module_from_spec(spec)
        sys.modules['openevolve_core_domain'] = domain_module
        spec.loader.exec_module(domain_module)

        # Extract all the classes and functions we need
        FinanceOptimizer = domain_module.FinanceOptimizer
        TradingOptimizer = domain_module.TradingOptimizer
        ScienceOptimizer = domain_module.ScienceOptimizer
        EngineeringOptimizer = domain_module.EngineeringOptimizer
        PharmaOptimizer = domain_module.PharmaOptimizer
        WebDesignOptimizer = domain_module.WebDesignOptimizer
        DomainOptimizer = domain_module.DomainOptimizer
        detect_domain = domain_module.detect_domain
        get_optimizer = domain_module.get_optimizer
        optimize_by_domain = domain_module.optimize_by_domain
        optimize_multi_domain = domain_module.optimize_multi_domain
        EvolutionMode = domain_module.EvolutionMode
        DomainType = domain_module.DomainType

        __all__ = [
            'FinanceOptimizer',
            'TradingOptimizer',
            'ScienceOptimizer',
            'EngineeringOptimizer',
            'PharmaOptimizer',
            'WebDesignOptimizer',
            'DomainOptimizer',
            'detect_domain',
            'get_optimizer',
            'optimize_by_domain',
            'optimize_multi_domain',
            'EvolutionMode',
            'DomainType',
        ]
    else:
        raise ImportError("Could not load core-projects domain module")
except (ImportError, AttributeError) as e:
    # If core-projects not available, provide stubs
    import warnings
    warnings.warn(f"Core projects not available: {e}")

    class DomainOptimizer:
        """Domain optimizer base class (stub)."""
        def __init__(self, sub_domain: str = "general"):
            self.sub_domain = sub_domain

    class FinanceOptimizer(DomainOptimizer):
        """Finance domain optimizer (stub)."""
        pass

    class TradingOptimizer(DomainOptimizer):
        """Trading domain optimizer (stub)."""
        pass

    class ScienceOptimizer(DomainOptimizer):
        """Science domain optimizer (stub)."""
        pass

    class EngineeringOptimizer(DomainOptimizer):
        """Engineering domain optimizer (stub)."""
        pass

    class PharmaOptimizer(DomainOptimizer):
        """Pharma domain optimizer (stub)."""
        pass

    class WebDesignOptimizer(DomainOptimizer):
        """Web design domain optimizer (stub)."""
        pass

    def detect_domain(problem_description: str) -> str:
        """Detect domain from problem description (stub)."""
        return "general"

    def get_optimizer(domain: str, sub_domain: str = "general") -> DomainOptimizer:
        """Get optimizer for domain (stub)."""
        return DomainOptimizer(sub_domain)

    async def optimize_by_domain(problem: str, domain: str = None, sub_domain: str = "general", constraints: dict = None, **kwargs):
        """Optimize using domain-specific configuration (stub)."""
        return {}

    async def optimize_multi_domain(problem: str, domains: list, sub_domain: str = "general", constraints: dict = None, **kwargs):
        """Optimize for multiple domains (stub)."""
        return {}

    # Stub enums
    class EvolutionMode:
        STANDARD = "standard"
        PES = "pes"
        QD = "qd"
        MO = "mo"
        ADVERSARIAL = "adversarial"

    class DomainType:
        GENERAL = "general"
        FINANCE = "finance"
        TRADING = "trading"
        SCIENCE = "science"
        ENGINEERING = "engineering"
        PHARMA = "pharma"
        WEB = "web"
        WEB_DESIGN = "web_design"

    __all__ = [
        'FinanceOptimizer',
        'TradingOptimizer',
        'ScienceOptimizer',
        'EngineeringOptimizer',
        'PharmaOptimizer',
        'WebDesignOptimizer',
        'DomainOptimizer',
        'detect_domain',
        'get_optimizer',
        'optimize_by_domain',
        'optimize_multi_domain',
        'EvolutionMode',
        'DomainType',
    ]
