"""Monitoring and consensus helpers for cloud LLM determinism."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .backends import LLMInterface
from .utils import similarity


def detect_divergence(results: List[str], threshold: float = 0.6) -> Dict[str, Any]:
    if len(results) < 2:
        return {"status": "INSUFFICIENT_DATA", "avg_similarity": 1.0, "pairwise": []}
    pairwise = []
    total = 0.0
    count = 0
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            score = similarity(results[i], results[j])
            pairwise.append({"i": i, "j": j, "similarity": score})
            total += score
            count += 1
    avg = total / max(count, 1)
    status = "CONSENSUS" if avg >= threshold else "DIVERGENCE_DETECTED"
    return {"status": status, "avg_similarity": avg, "pairwise": pairwise}


def cloud_consensus(
    prompt: str,
    runs: int = 5,
    threshold: float = 0.6,
    llm: Optional[LLMInterface] = None,
) -> Dict[str, Any]:
    if llm is None:
        raise ValueError("cloud_consensus requires an LLM instance")
    
    # Observability integration
    obs_manager = None
    try:
        from .deps import optional_import
        obs_module = optional_import("monitoring_system")
        if obs_module:
            obs_manager = obs_module.get_observability_manager()
    except Exception:
        pass

    results = []
    for _ in range(runs):
        results.append(llm.generate(prompt))
    
    divergence = detect_divergence(results, threshold=threshold)
    
    # Record metrics
    if obs_manager:
        try:
            from monitoring_system import MetricType
            obs_manager.add_custom_metric("cloud_consensus_agreement", divergence["avg_similarity"], MetricType.GAUGE)
            obs_manager.add_custom_metric("cloud_consensus_status", 1 if divergence["status"] == "CONSENSUS" else 0, MetricType.COUNTER)
        except Exception:
            pass

    if divergence["status"] == "CONSENSUS":
        return {"status": "CONSENSUS", "result": results[0], "divergence": divergence}
    return {"status": "DIVERGENCE", "result": results[0], "divergence": divergence}


@dataclass
class CloudLLMMonitor:
    history: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def check(self, prompt: str, runs: int = 3, llm: Optional[LLMInterface] = None) -> Dict[str, Any]:
        if llm is None:
            raise ValueError("CloudLLMMonitor.check requires an LLM instance")
        
        # Observability integration
        obs_manager = None
        try:
            from .deps import optional_import
            obs_module = optional_import("monitoring_system")
            if obs_module:
                obs_manager = obs_module.get_observability_manager()
        except Exception:
            pass

        results = [llm.generate(prompt) for _ in range(runs)]
        if prompt not in self.history:
            self.history[prompt] = {
                "baseline": results[0],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            return {"status": "BASELINE_ESTABLISHED"}
        
        baseline = self.history[prompt]["baseline"]
        divergence = detect_divergence([baseline] + results)
        
        # Alert if regression detected
        if obs_manager and divergence["status"] == "DIVERGENCE_DETECTED":
            try:
                obs_manager._trigger_monitoring_alerts(
                    "cloud_monitor_regression", 
                    False, 
                    "cloud_consensus_agreement", 
                    f"Regression detected for prompt: {prompt[:50]}...",
                    {"divergence": divergence}
                )
            except Exception:
                pass

        return divergence
