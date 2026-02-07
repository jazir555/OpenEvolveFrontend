"""Task Complexity Classifier."""
from typing import Any, Dict, List, Optional
from enum import Enum

class ComplexityLevel(Enum):
    """Complexity level."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class TaskComplexityClassifier:
    """Task complexity classifier."""
    
    def classify(self, task: Dict[str, Any]) -> ComplexityLevel:
        """Classify task complexity."""
        return ComplexityLevel.MEDIUM
