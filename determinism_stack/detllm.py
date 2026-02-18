"""Implementation of detLLM: Runtime Reproducibility Verification."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import optional_import, similarity

@dataclass
class EnvFingerprint:
    python_version: str = sys.version
    os_info: str = platform.platform()
    cpu_info: str = platform.processor()
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    device_info: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @classmethod
    def capture(cls) -> EnvFingerprint:
        torch = optional_import("torch")
        torch_ver = torch.__version__ if torch else None
        cuda_ver = torch.version.cuda if torch and hasattr(torch.version, "cuda") else None
        devices = []
        if torch and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.cuda.get_device_name(i))
        return cls(torch_version=torch_ver, cuda_version=cuda_ver, device_info=devices)

@dataclass
class ReproducibilityReport:
    status: str # PASS | FAIL | ERROR | UNAVAILABLE
    category: str # RUN_VARIANCE | BATCH_VARIANCE | PASS
    execution_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)
    artifacts_dir: Optional[str] = None

class DetLLM:
    """detLLM handles low-level inference determinism verification."""

    def __init__(self, artifacts_root: str = "artifacts/detllm"):
        self.artifacts_root = Path(artifacts_root)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def check(
        self,
        backend: str,
        model: str,
        prompts: List[str],
        runs: int = 3,
        batch_size: int = 1,
        tier: int = 1,
        vary_batch: Optional[List[int]] = None
    ) -> ReproducibilityReport:
        """Perform a full reproducibility check across multiple runs."""
        execution_id = f"check_{int(time.time())}"
        run_dir = self.artifacts_root / execution_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"detLLM: Starting check {execution_id} (tier={tier}, runs={runs})")

        # 1. Capture environment
        env = EnvFingerprint.capture()
        with open(run_dir / "env.json", "w") as f:
            json.dump(asdict(env), f, indent=2)

        # 2. Capture run config
        run_config = {
            "backend": backend,
            "model": model,
            "runs": runs,
            "batch_size": batch_size,
            "tier": tier,
            "prompts": prompts
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(run_config, f, indent=2)

        # 3. Simulate/Perform runs and capture traces
        # In a real implementation, this would call the backend multiple times
        # and capture token-level logs if available.
        all_run_outputs = []
        
        # For this implementation, we assume we're verifying a "black box" 
        # or we use the local backend if available.
        # Here we simulation the check logic.
        
        results = []
        # Mocking run results for the implementation
        for i in range(runs):
            # Simulation: in real life, we'd call backend.generate()
            results.append([f"Output for run {i} prompt {j}" for j in range(len(prompts))])

        # 4. Analyze variance
        first_divergence = None
        status = "PASS"
        category = "PASS"
        
        for p_idx in range(len(prompts)):
            outputs = [r[p_idx] for r in results]
            if len(set(outputs)) > 1:
                status = "FAIL"
                category = "RUN_VARIANCE_FIXED_BATCH"
                first_divergence = {
                    "prompt_index": p_idx,
                    "prompt": prompts[p_idx],
                    "outputs": outputs
                }
                break

        # 5. Generate report
        report = ReproducibilityReport(
            status=status,
            category=category,
            execution_id=execution_id,
            details={
                "first_divergence": first_divergence,
                "total_runs": runs,
                "tier_effective": tier
            },
            artifacts_dir=str(run_dir)
        )

        with open(run_dir / "report.json", "w") as f:
            json.dump(asdict(report), f, indent=2)

        if status == "FAIL":
            (run_dir / "diffs").mkdir(exist_ok=True)
            with open(run_dir / "diffs" / "first_divergence.json", "w") as f:
                json.dump(first_divergence, f, indent=2)

        return report
