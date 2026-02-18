"""Reproducibility context and controls for Layer 8."""

import random
import contextlib
from typing import Optional, Iterator
from dataclasses import dataclass

from .utils import optional_import

@dataclass
class ReproducibilityState:
    """Captured state of PRNGs."""
    random_state: object
    numpy_state: Optional[object] = None
    torch_state: Optional[object] = None
    cuda_state: Optional[object] = None

class ReproducibilityContext:
    """Enforces deterministic context across various backends."""
    
    def __init__(self, seed: int = 42, tier: int = 1):
        self.seed = seed
        self.tier = tier
        self._snapshot: Optional[ReproducibilityState] = None
        
        self.numpy = optional_import("numpy")
        self.torch = optional_import("torch")

    @contextlib.contextmanager
    def enforce(self) -> Iterator[None]:
        """Apply deterministic controls."""
        # 1. Take snapshot
        self._snapshot = ReproducibilityState(
            random_state=random.getstate(),
            numpy_state=self.numpy.random.get_state() if self.numpy else None,
            torch_state=self.torch.random.get_rng_state() if self.torch else None,
            cuda_state=self.torch.cuda.get_rng_state_all() if (self.torch and self.torch.cuda.is_available()) else None
        )
        
        # 2. Apply seeds
        random.seed(self.seed)
        if self.numpy:
            self.numpy.random.seed(self.seed)
        if self.torch:
            self.torch.manual_seed(self.seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(self.seed)
            
            # 3. Tier-based algorithm enforcement
            if self.tier >= 1:
                # Force deterministic algorithms
                if hasattr(self.torch, "use_deterministic_algorithms"):
                    try:
                        self.torch.use_deterministic_algorithms(True)
                    except Exception:
                        pass
                
                # CUDA specific controls
                if self.torch.cuda.is_available():
                    import os
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        try:
            yield
        finally:
            # 4. Restore state
            if self._snapshot:
                random.setstate(self._snapshot.random_state)
                if self.numpy and self._snapshot.numpy_state is not None:
                    self.numpy.random.set_state(self._snapshot.numpy_state)
                if self.torch and self._snapshot.torch_state is not None:
                    self.torch.random.set_rng_state(self._snapshot.torch_state)
                if self.torch and self._snapshot.cuda_state is not None:
                    self.torch.cuda.set_rng_state_all(self._snapshot.cuda_state)
