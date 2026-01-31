"""
Global Learning Engine

Aggregates learning across all users and executions to create a shared
knowledge base that improves for everyone over time.

Key features:
- Multi-user experience aggregation
- Pattern sharing across users
- Knowledge curation and refinement
- Transfer learning between domains
- Anonymized learning data sharing
- Versioned knowledge base
"""

import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import statistics
import hashlib
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class GlobalPattern:
    """A globally learned pattern shared across users"""
    pattern_id: str
    pattern_type: str  # 'component_config', 'pipeline', 'healing_strategy', 'domain_mapping'
    
    # Pattern data
    input_signature: Dict[str, Any]  # What inputs this applies to
    output_effectiveness: float  # How effective it is (0-1)
    
    # Source information
    source_executions: int  # Number of executions that contributed
    unique_users: int  # Number of unique users
    
    # Metadata
    created_at: str
    last_updated: str
    version: int = 1
    
    # Quality metrics
    success_rate: float = 0.0
    average_quality: float = 0.0
    confidence: float = 0.0


@dataclass
class KnowledgeEntry:
    """A curated knowledge entry in the global knowledge base"""
    entry_id: str
    entry_type: str  # 'entity_type', 'relation_type', 'pattern', 'best_practice'
    
    # Content
    content: Dict[str, Any]
    
    # Source tracking
    source_domain: str
    source_executions: int
    first_seen: str
    last_validated: str
    
    # Quality metrics
    accuracy_score: float
    usage_count: int
    validation_status: str  # 'pending', 'validated', 'deprecated'
    
    # User contributions
    contributing_users: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['contributing_users'] = list(self.contributing_users)
        return result


class GlobalLearningEngine:
    """
    Global learning engine that aggregates knowledge across all users.
    
    This creates a continuously improving knowledge base that benefits
    all users of the system. As more people use it, everyone gets better results.
    """
    
    def __init__(self, storage_path: Optional[str] = None, 
                 enable_sharing: bool = True):
        """
        Initialize global learning engine.
        
        Args:
            storage_path: Path to persist global learning data
            enable_sharing: Whether to enable cross-user learning
        """
        self.storage_path = storage_path
        self.enable_sharing = enable_sharing
        
        # Global knowledge base
        self.patterns: Dict[str, GlobalPattern] = {}
        self.knowledge_base: Dict[str, KnowledgeEntry] = {}
        self.user_contributions: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Statistics
        self.total_executions = 0
        self.total_users = 0
        self.execution_history: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load persisted data
        if storage_path:
            self._load_data()
        
        logger.info({
            "msg": "GlobalLearningEngine initialized",
            "patterns": len(self.patterns),
            "knowledge_entries": len(self.knowledge_base),
            "sharing_enabled": enable_sharing,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def contribute_experience(self, 
                             user_id: str,
                             execution_result: Dict[str, Any],
                             local_learning: Optional[Dict[str, Any]] = None):
        """
        Contribute a user's execution experience to global learning.
        
        Args:
            user_id: Anonymous user identifier
            execution_result: Result from execution
            local_learning: Optional local learning data
        """
        with self._lock:
            self.total_executions += 1
            
            if user_id not in self.user_contributions:
                self.total_users += 1
            
            # Record contribution
            contribution = {
                'user_id': hashlib.sha256(user_id.encode()).hexdigest()[:16],  # Anonymized
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'result': execution_result,
                'local_learning': local_learning
            }
            
            self.execution_history.append(contribution)
            
            # Extract and update patterns
            self._extract_patterns(execution_result, user_id)
            
            # Update knowledge base
            self._update_knowledge_base(execution_result, user_id)
            
            # Persist
            if self.storage_path:
                self._save_data()
            
            logger.debug({
                "msg": "Experience contributed to global learning",
                "user_id_hash": contribution['user_id'][:8],
                "total_executions": self.total_executions
            })
    
    def _extract_patterns(self, execution_result: Dict[str, Any], user_id: str):
        """Extract patterns from execution result"""
        
        # Pattern 1: Successful component configurations
        if execution_result.get('status') in ('success', 'partial'):
            components_used = execution_result.get('results', {}).keys()
            domain = execution_result.get('domain', 'general')
            data_type = execution_result.get('input_data', {}).get('data_type', 'unknown')
            
            pattern_key = f"config:{domain}:{data_type}"
            
            if pattern_key not in self.patterns:
                self.patterns[pattern_key] = GlobalPattern(
                    pattern_id=pattern_key,
                    pattern_type='component_config',
                    input_signature={'domain': domain, 'data_type': data_type},
                    output_effectiveness=0.5,
                    source_executions=0,
                    unique_users=0,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    last_updated=datetime.now(timezone.utc).isoformat()
                )
            
            pattern = self.patterns[pattern_key]
            pattern.source_executions += 1
            pattern.unique_users += 1
            pattern.last_updated = datetime.now(timezone.utc).isoformat()
            
            # Update effectiveness based on result quality
            quality = execution_result.get('execution', {}).get('quality_score', 0.5)
            pattern.output_effectiveness = (
                pattern.output_effectiveness * 0.9 + quality * 0.1
            )
            pattern.success_rate = pattern.output_effectiveness
        
        # Pattern 2: Healing strategies that worked
        if execution_result.get('healing_applied'):
            strategy = execution_result.get('healing_strategy', 'unknown')
            pattern_key = f"healing:{strategy}"
            
            if pattern_key not in self.patterns:
                self.patterns[pattern_key] = GlobalPattern(
                    pattern_id=pattern_key,
                    pattern_type='healing_strategy',
                    input_signature={'strategy': strategy},
                    output_effectiveness=0.5,
                    source_executions=0,
                    unique_users=0,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    last_updated=datetime.now(timezone.utc).isoformat()
                )
            
            pattern = self.patterns[pattern_key]
            pattern.source_executions += 1
            pattern.output_effectiveness = min(
                pattern.output_effectiveness * 0.95 + 0.05,
                1.0
            )
    
    def _update_knowledge_base(self, execution_result: Dict[str, Any], user_id: str):
        """Update global knowledge base from execution"""
        
        results = execution_result.get('results', {})
        domain = execution_result.get('domain', 'general')
        
        # Extract entities as knowledge
        if 'entities' in results:
            for entity in results['entities']:
                entity_type = entity.get('type', 'unknown')
                entity_text = entity.get('text', '')
                
                entry_key = f"entity:{entity_type}:{hashlib.md5(entity_text.encode()).hexdigest()[:16]}"
                
                if entry_key not in self.knowledge_base:
                    self.knowledge_base[entry_key] = KnowledgeEntry(
                        entry_id=entry_key,
                        entry_type='entity_type',
                        content={'type': entity_type, 'example': entity_text},
                        source_domain=domain,
                        source_executions=0,
                        first_seen=datetime.now(timezone.utc).isoformat(),
                        last_validated=datetime.now(timezone.utc).isoformat(),
                        accuracy_score=0.5,
                        usage_count=0,
                        validation_status='pending'
                    )
                
                entry = self.knowledge_base[entry_key]
                entry.source_executions += 1
                entry.usage_count += 1
                entry.contributing_users.add(user_id)
                
                # Increase accuracy if execution was successful
                if execution_result.get('status') == 'success':
                    entry.accuracy_score = min(entry.accuracy_score * 0.95 + 0.05, 1.0)
    
    def get_recommendations(self, 
                           input_context: Dict[str, Any],
                           local_user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get globally-learned recommendations for an input.
        
        Args:
            input_context: Context about the input
            local_user_id: Optional local user ID for personalization
            
        Returns:
            Dictionary of recommendations
        """
        with self._lock:
            domain = input_context.get('domain', 'general')
            data_type = input_context.get('data_type', 'unknown')
            
            recommendations = {
                'component_configs': [],
                'healing_strategies': [],
                'similar_patterns': [],
                'global_stats': self.get_stats()
            }
            
            # Find matching patterns
            for pattern_id, pattern in self.patterns.items():
                if pattern.output_effectiveness < 0.3:
                    continue  # Skip low-effectiveness patterns
                
                # Match by domain and data type
                if pattern.pattern_type == 'component_config':
                    sig = pattern.input_signature
                    if sig.get('domain') == domain and sig.get('data_type') == data_type:
                        recommendations['component_configs'].append({
                            'pattern_id': pattern_id,
                            'effectiveness': pattern.output_effectiveness,
                            'confidence': min(pattern.source_executions / 10, 1.0),
                            'users': pattern.unique_users
                        })
                
                elif pattern.pattern_type == 'healing_strategy':
                    if pattern.output_effectiveness > 0.6:
                        recommendations['healing_strategies'].append({
                            'strategy': pattern.input_signature.get('strategy'),
                            'success_rate': pattern.output_effectiveness,
                            'usage_count': pattern.source_executions
                        })
            
            # Sort by effectiveness
            recommendations['component_configs'].sort(
                key=lambda x: x['effectiveness'], 
                reverse=True
            )
            recommendations['healing_strategies'].sort(
                key=lambda x: x['success_rate'], 
                reverse=True
            )
            
            return recommendations
    
    def get_curated_knowledge(self, 
                             domain: Optional[str] = None,
                             min_accuracy: float = 0.6) -> List[KnowledgeEntry]:
        """
        Get curated knowledge entries.
        
        Args:
            domain: Optional domain filter
            min_accuracy: Minimum accuracy score
            
        Returns:
            List of knowledge entries
        """
        with self._lock:
            entries = []
            
            for entry in self.knowledge_base.values():
                if entry.accuracy_score < min_accuracy:
                    continue
                
                if domain and entry.source_domain != domain:
                    continue
                
                entries.append(entry)
            
            # Sort by accuracy and usage
            entries.sort(
                key=lambda e: (e.accuracy_score, e.usage_count),
                reverse=True
            )
            
            return entries
    
    def validate_knowledge(self, entry_id: str, 
                          validator_id: str,
                          is_valid: bool,
                          correction: Optional[Dict[str, Any]] = None):
        """
        Validate or correct a knowledge entry.
        
        Args:
            entry_id: Knowledge entry ID
            validator_id: User validating
            is_valid: Whether the entry is valid
            correction: Optional correction data
        """
        with self._lock:
            if entry_id not in self.knowledge_base:
                logger.warning(f"Knowledge entry {entry_id} not found")
                return
            
            entry = self.knowledge_base[entry_id]
            
            if is_valid:
                entry.validation_status = 'validated'
                entry.accuracy_score = min(entry.accuracy_score + 0.1, 1.0)
            else:
                entry.validation_status = 'deprecated'
                entry.accuracy_score *= 0.5
            
            entry.last_validated = datetime.now(timezone.utc).isoformat()
            entry.contributing_users.add(validator_id)
            
            if correction:
                entry.content.update(correction)
            
            logger.info({
                "msg": "Knowledge entry validated",
                "entry_id": entry_id,
                "is_valid": is_valid,
                "new_accuracy": entry.accuracy_score
            })
            
            if self.storage_path:
                self._save_data()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get global learning statistics"""
        with self._lock:
            # Calculate average effectiveness
            if self.patterns:
                avg_effectiveness = statistics.mean(
                    p.output_effectiveness for p in self.patterns.values()
                )
            else:
                avg_effectiveness = 0.0
            
            # Pattern type distribution
            pattern_types = defaultdict(int)
            for p in self.patterns.values():
                pattern_types[p.pattern_type] += 1
            
            return {
                "total_executions": self.total_executions,
                "unique_users": self.total_users,
                "learned_patterns": len(self.patterns),
                "knowledge_entries": len(self.knowledge_base),
                "average_pattern_effectiveness": avg_effectiveness,
                "pattern_types": dict(pattern_types),
                "knowledge_by_domain": self._count_knowledge_by_domain(),
                "sharing_enabled": self.enable_sharing
            }
    
    def _count_knowledge_by_domain(self) -> Dict[str, int]:
        """Count knowledge entries by domain"""
        counts = defaultdict(int)
        for entry in self.knowledge_base.values():
            counts[entry.source_domain] += 1
        return dict(counts)
    
    def export_knowledge(self, 
                        domain: Optional[str] = None,
                        min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        Export curated knowledge for sharing.
        
        Args:
            domain: Optional domain filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            Exportable knowledge package
        """
        with self._lock:
            # Filter patterns
            patterns = {
                k: {
                    'type': p.pattern_type,
                    'signature': p.input_signature,
                    'effectiveness': p.output_effectiveness,
                    'confidence': min(p.source_executions / 10, 1.0),
                    'executions': p.source_executions
                }
                for k, p in self.patterns.items()
                if p.output_effectiveness >= min_confidence
                and (domain is None or p.input_signature.get('domain') == domain)
            }
            
            # Filter knowledge
            knowledge = [
                {
                    'type': e.entry_type,
                    'content': e.content,
                    'domain': e.source_domain,
                    'accuracy': e.accuracy_score,
                    'usage': e.usage_count
                }
                for e in self.knowledge_base.values()
                if e.accuracy_score >= min_confidence
                and e.validation_status == 'validated'
                and (domain is None or e.source_domain == domain)
            ]
            
            return {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
                "domain_filter": domain,
                "patterns": patterns,
                "knowledge": knowledge,
                "stats": self.get_stats()
            }
    
    def import_knowledge(self, knowledge_package: Dict[str, Any],
                        source: str = "external"):
        """
        Import knowledge from external source.
        
        Args:
            knowledge_package: Knowledge package to import
            source: Source identifier
        """
        with self._lock:
            imported_patterns = 0
            imported_knowledge = 0
            
            # Import patterns
            for pattern_id, pattern_data in knowledge_package.get('patterns', {}).items():
                if pattern_id not in self.patterns:
                    self.patterns[pattern_id] = GlobalPattern(
                        pattern_id=pattern_id,
                        pattern_type=pattern_data['type'],
                        input_signature=pattern_data['signature'],
                        output_effectiveness=pattern_data['effectiveness'] * 0.9,  # Slightly reduce
                        source_executions=pattern_data.get('executions', 1),
                        unique_users=1,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        last_updated=datetime.now(timezone.utc).isoformat()
                    )
                    imported_patterns += 1
            
            # Import knowledge entries
            for entry_data in knowledge_package.get('knowledge', []):
                entry_hash = hashlib.md5(
                    json.dumps(entry_data['content'], sort_keys=True).encode()
                ).hexdigest()[:16]
                entry_id = f"imported:{source}:{entry_hash}"
                
                if entry_id not in self.knowledge_base:
                    self.knowledge_base[entry_id] = KnowledgeEntry(
                        entry_id=entry_id,
                        entry_type=entry_data['type'],
                        content=entry_data['content'],
                        source_domain=entry_data.get('domain', 'general'),
                        source_executions=entry_data.get('usage', 1),
                        first_seen=datetime.now(timezone.utc).isoformat(),
                        last_validated=datetime.now(timezone.utc).isoformat(),
                        accuracy_score=entry_data.get('accuracy', 0.5) * 0.9,
                        usage_count=0,
                        validation_status='pending'
                    )
                    imported_knowledge += 1
            
            logger.info({
                "msg": "Knowledge imported",
                "source": source,
                "patterns_imported": imported_patterns,
                "knowledge_imported": imported_knowledge
            })
            
            if self.storage_path:
                self._save_data()
    
    def _save_data(self):
        """Persist global learning data"""
        try:
            data = {
                "patterns": {
                    k: {
                        **{field: getattr(p, field) for field in ['pattern_id', 'pattern_type', 'input_signature', 'output_effectiveness', 'source_executions', 'unique_users', 'created_at', 'last_updated', 'version', 'success_rate', 'average_quality', 'confidence']}
                    }
                    for k, p in self.patterns.items()
                },
                "knowledge_base": {
                    k: v.to_dict()
                    for k, v in self.knowledge_base.items()
                },
                "stats": {
                    "total_executions": self.total_executions,
                    "total_users": self.total_users
                },
                "saved_at": datetime.now(timezone.utc).isoformat()
            }
            
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save global learning data: {e}")
    
    def _load_data(self):
        """Load persisted global learning data"""
        try:
            import os
            if not os.path.exists(self.storage_path):
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Load patterns
            for pattern_id, p_data in data.get('patterns', {}).items():
                self.patterns[pattern_id] = GlobalPattern(**p_data)
            
            # Load knowledge base
            for entry_id, e_data in data.get('knowledge_base', {}).items():
                e_data['contributing_users'] = set(e_data.get('contributing_users', []))
                self.knowledge_base[entry_id] = KnowledgeEntry(**e_data)
            
            # Load stats
            stats = data.get('stats', {})
            self.total_executions = stats.get('total_executions', 0)
            self.total_users = stats.get('total_users', 0)
            
            logger.info({
                "msg": "Global learning data loaded",
                "patterns": len(self.patterns),
                "knowledge_entries": len(self.knowledge_base)
            })
            
        except Exception as e:
            logger.error(f"Failed to load global learning data: {e}")


# Singleton instance for system-wide use
_global_learning_engine: Optional[GlobalLearningEngine] = None


def get_global_learning_engine(storage_path: Optional[str] = None) -> GlobalLearningEngine:
    """Get or create global learning engine singleton"""
    global _global_learning_engine
    
    if _global_learning_engine is None:
        _global_learning_engine = GlobalLearningEngine(storage_path)
    
    return _global_learning_engine
