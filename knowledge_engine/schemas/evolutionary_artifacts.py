"""Knowledge Engine Evolutionary Artifacts Schema."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class EvolutionaryArtifact:
    """Evolutionary artifact."""
    id: str = ""
    generation: int = 0
    fitness: float = 0.0
    genome: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.genome is None:
            self.genome = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ArtifactCollection:
    """Collection of evolutionary artifacts."""
    artifacts: List[EvolutionaryArtifact] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []
