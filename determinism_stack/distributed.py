"""Distributed determinism coordinator with best-effort fallbacks."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from .utils import optional_import


class DistributedDeterminismCoordinator:
    def __init__(self, cluster_config: Dict[str, Any]):
        self.cluster_config = cluster_config
        self._cache: Dict[str, Any] = {}
        self._etcd = None
        self._producer = None

        etcd3 = optional_import("etcd3")
        kafka = optional_import("kafka")
        if etcd3:
            try:
                self._etcd = etcd3.client()
            except Exception:
                self._etcd = None
        if kafka and cluster_config.get("kafka_brokers"):
            try:
                self._producer = kafka.KafkaProducer(bootstrap_servers=cluster_config["kafka_brokers"])
            except Exception:
                self._producer = None

    def coordinate_generation(self, prompt: str, require_consensus: bool = True, llm: Optional[Any] = None) -> Any:
        prompt_id = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()
        cached = self._read_cache(prompt_id)
        if cached is not None:
            return cached
        if require_consensus:
            result = self._consensus_generate(prompt, llm=llm)
        else:
            result = self._single_generate(prompt, llm=llm)
        self._write_cache(prompt_id, result)
        return result

    def _read_cache(self, key: str) -> Optional[Any]:
        if self._etcd is not None:
            try:
                value, _ = self._etcd.get(f"/results/{key}")
                if value is not None:
                    return value.decode("utf-8", errors="ignore")
            except Exception:
                pass
        return self._cache.get(key)

    def _write_cache(self, key: str, value: Any) -> None:
        self._cache[key] = value
        if self._etcd is not None:
            try:
                self._etcd.put(f"/results/{key}", str(value))
            except Exception:
                pass

    def _single_generate(self, prompt: str, llm: Optional[Any] = None) -> str:
        if llm:
            return llm.generate(prompt)
        return f"[distributed-single] {prompt}"

    def _consensus_generate(self, prompt: str, llm: Optional[Any] = None) -> str:
        nodes = self.cluster_config.get("nodes", ["node-1", "node-2"])
        if llm:
            results = [llm.generate(prompt) for _ in nodes]
        else:
            results = [f"[{node}] {prompt}" for node in nodes]
        return results[0]
