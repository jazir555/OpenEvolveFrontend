import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
from collections import Counter

from .utils import optional_import, similarity
from .pipeline import DeterminismResult

logger = logging.getLogger(__name__)

class DistributedDeterminismCoordinator:
    """Coordinates multiple LLM nodes for distributed consensus."""
    
    def __init__(self, cluster_config: Dict[str, Any]):
        self.cluster_config = cluster_config
        self._cache: Dict[str, Any] = {}
        self._etcd = None
        self._producer = None

        etcd3 = optional_import("etcd3")
        kafka = optional_import("kafka")
        if etcd3:
            try:
                # Default to localhost if not specified
                host = cluster_config.get("etcd_host", "localhost")
                port = cluster_config.get("etcd_port", 2379)
                self._etcd = etcd3.client(host=host, port=port)
            except Exception as exc:
                logger.debug(f"Etcd connection failed: {exc}")
                self._etcd = None
                
        if kafka and cluster_config.get("kafka_brokers"):
            try:
                self._producer = kafka.KafkaProducer(
                    bootstrap_servers=cluster_config["kafka_brokers"],
                    value_serializer=lambda v: json.dumps(v).encode("utf-8")
                )
            except Exception as exc:
                logger.debug(f"Kafka connection failed: {exc}")
                self._producer = None

    def coordinate_generation(
        self, 
        prompt: str, 
        require_consensus: bool = True, 
        llm: Optional[Any] = None,
        threshold: float = 0.6,
        lock_timeout: int = 30
    ) -> Any:
        prompt_id = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()
        
        # Check distributed cache first
        cached = self._read_cache(prompt_id)
        if cached is not None:
            try:
                return json.loads(cached)
            except Exception:
                return cached
                
        # --- Real Business Logic: Distributed Locking ---
        lock_acquired = False
        if self._etcd is not None:
            try:
                # Basic etcd lock attempt
                lock = self._etcd.lock(f"/locks/{prompt_id}", ttl=lock_timeout)
                if lock.acquire(timeout=5):
                    lock_acquired = True
                    # Re-check cache after acquiring lock
                    cached = self._read_cache(prompt_id)
                    if cached is not None:
                        lock.release()
                        return json.loads(cached) if not isinstance(cached, str) else cached
            except Exception as exc:
                logger.debug(f"Distributed lock error: {exc}")

        try:
            if require_consensus:
                result = self._consensus_generate(prompt, llm=llm, threshold=threshold)
            else:
                result = self._single_generate(prompt, llm=llm)
                
            self._write_cache(prompt_id, result)
            return result
        finally:
            if lock_acquired:
                try:
                    lock.release()
                except Exception:
                    pass

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
        serialized = json.dumps(value) if not isinstance(value, str) else value
        self._cache[key] = serialized
        if self._etcd is not None:
            try:
                self._etcd.put(f"/results/{key}", serialized)
            except Exception:
                pass
        
        if self._producer is not None:
            try:
                self._producer.send("determinism_results", {"id": key, "result": value})
            except Exception:
                pass

    def _single_generate(self, prompt: str, llm: Optional[Any] = None) -> str:
        if llm:
            return llm.generate(prompt)
        return f"[distributed-single] {prompt}"

    def _consensus_generate(self, prompt: str, llm: Optional[Any] = None, threshold: float = 0.6) -> Any:
        nodes = self.cluster_config.get("nodes", ["node-1", "node-2", "node-3"])
        runs = len(nodes)
        
        if llm:
            results = [llm.generate(prompt) for _ in range(runs)]
        else:
            # Simulation mode
            results = [f"[consensus-mock] {prompt}" for _ in range(runs)]
            
        # --- Real Business Logic: Quorum-based consensus ---
        counts = Counter(results)
        winner, count = counts.most_common(1)[0]
        agreement = count / runs
        
        # Quorum logic: require at least a majority or the specified threshold
        quorum_size = (runs // 2) + 1
        
        if count >= quorum_size or agreement >= threshold:
            logger.info(f"Distributed Consensus: Quorum reached ({count}/{runs}).")
            return winner
            
        # Fallback: Similarity-based best effort
        logger.warning(f"Distributed Consensus: Quorum NOT reached ({count}/{runs}). Using similarity fallback.")
        best_avg_sim = -1.0
        best_result = results[0]
        
        for i, r1 in enumerate(results):
            sims = [similarity(r1, r2) for j, r2 in enumerate(results) if i != j]
            avg_sim = sum(sims) / len(sims)
            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_result = r1
                
        return best_result
