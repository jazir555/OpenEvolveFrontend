
# **ACTUAL INTEGRATION**: Adaptive MDAP for Propose Base
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

from abc import ABC, abstractmethod


class Proposer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def propose_instructions_for_program(self):
        pass

    def propose_instruction_for_predictor(self):
        pass
