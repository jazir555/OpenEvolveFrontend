"""
Agentic Context Engine (ACE) Integration for OpenEvolve Knowledge Engine

This module provides integration with the Agentic Context Engine (ACE) system,
enabling adaptive learning, reflection, and skill management capabilities.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Iterable
from dataclasses import dataclass
import uuid

# Try to import ACE classes for test patching compatibility
try:
    from ace import Sample, SimpleEnvironment
    _ace_available = True
except ImportError:
    # Create stub classes for patching
    class Sample:
        pass
    class SimpleEnvironment:
        pass
    _ace_available = False


logger = logging.getLogger(__name__)


@dataclass
class ACEIntegrationResult:
    """Result of an ACE integration operation."""
    success: bool
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    skills: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'entities': self.entities,
            'relations': self.relations,
            'skills': self.skills,
            'metadata': self.metadata,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error
        }


class AgenticContextEngine:
    """
    Integration with Agentic Context Engine (ACE) for adaptive knowledge processing.
    
    Provides methods for:
    - Adaptive learning from interactions
    - Reflection and self-improvement
    - Skill management and evolution
    - Context-aware processing
    - Continuous improvement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ACE integration.

        Args:
            config: Configuration for ACE components
        """
        self.config = config or self._get_default_config()

        # Initialize ACE components
        self.skillbook = None
        self.agent = None
        self.reflector = None
        self.skill_manager = None
        self.offline_ace = None
        self.online_ace = None

        # Store ACE classes for use in methods
        self.Sample = None
        self.SimpleEnvironment = None

        # Initialize based on configuration
        self._initialize_components()

        logger.info({
            "msg": "AgenticContextEngineIntegration initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ACE integration."""
        return {
            "model": "gpt-4o",
            "max_refinement_rounds": 1,
            "reflection_window": 3,
            "async_learning": False,
            "max_reflector_workers": 3,
            "enable_observability": True,
            "deduplication": {
                "enabled": True,
                "similarity_threshold": 0.85,
                "embedding_model": "text-embedding-3-small"
            },
            "offline_training": {
                "default_epochs": 3,
                "checkpoint_interval": 10,
                "batch_size": 32
            },
            "online_learning": {
                "enabled": True,
                "max_samples_before_update": 5
            }
        }
    
    def _initialize_components(self):
        """Initialize ACE components based on configuration."""
        try:
            # Import ACE components
            from ace import (
                Skillbook,
                Agent,
                Reflector,
                SkillManager,
                OfflineACE,
                OnlineACE,
                SimpleEnvironment,
                Sample
            )
            from ace.llm import LLMClient

            # Store ACE classes for use in methods
            self.Sample = Sample
            self.SimpleEnvironment = SimpleEnvironment

            # Initialize LLM client
            model = self.config.get("model", "gpt-4o")
            llm_client = LLMClient(model=model)

            # Initialize components
            self.skillbook = Skillbook()
            self.agent = Agent(llm_client)
            self.reflector = Reflector(llm_client)
            self.skill_manager = SkillManager(llm_client)

            # Initialize ACE orchestrators
            self.offline_ace = OfflineACE(
                skillbook=self.skillbook,
                agent=self.agent,
                reflector=self.reflector,
                skill_manager=self.skill_manager,
                max_refinement_rounds=self.config.get("max_refinement_rounds", 1),
                reflection_window=self.config.get("reflection_window", 3),
                async_learning=self.config.get("async_learning", False)
            )

            self.online_ace = OnlineACE(
                skillbook=self.skillbook,
                agent=self.agent,
                reflector=self.reflector,
                skill_manager=self.skill_manager,
                max_refinement_rounds=self.config.get("max_refinement_rounds", 1),
                reflection_window=self.config.get("reflection_window", 3),
                async_learning=self.config.get("async_learning", False)
            )

            logger.info({
                "msg": "ACE components initialized successfully",
                "model": model,
                "async_learning": self.config.get("async_learning", False),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        except ImportError as e:
            logger.warning({
                "msg": f"ACE not available, using mock implementation: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Initialize with mock components
            self._initialize_mock_components()
        except Exception as e:
            logger.error({
                "msg": f"Failed to initialize ACE components: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise
    
    def _initialize_mock_components(self):
        """Initialize mock components when ACE is not available."""
        logger.warning({
            "msg": "ACE (Agentic Context Engine) not available - components will fail on use",
            "install": "pip install ace-framework",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Create failing mock implementations
        from ..optional_imports import create_failing_mock
        
        MockSkillbook = create_failing_mock(
            package_name='ace-framework',
            feature_name='ACE Skillbook',
            install_command='pip install ace-framework'
        )
        
        MockAgent = create_failing_mock(
            package_name='ace-framework',
            feature_name='ACE Agent',
            install_command='pip install ace-framework'
        )
        
        MockReflector = create_failing_mock(
            package_name='ace-framework',
            feature_name='ACE Reflector',
            install_command='pip install ace-framework'
        )
        
        MockSkillManager = create_failing_mock(
            package_name='ace-framework',
            feature_name='ACE SkillManager',
            install_command='pip install ace-framework'
        )
        
        self._mock_classes = {
            'skillbook': MockSkillbook,
            'agent': MockAgent,
            'reflector': MockReflector,
            'skill_manager': MockSkillManager
        }
        self.skillbook = None
        self.agent = None
        self.reflector = None
        self.skill_manager = None
        self.offline_ace = None
        self.online_ace = None
    
    async def process_with_adaptive_learning(
        self,
        text: str,
        context: str = "",
        ground_truth: Optional[str] = None,
        enable_reflection: bool = True,
        enable_skill_update: bool = True,
        correlation_id: Optional[str] = None
    ) -> ACEIntegrationResult:
        """
        Process text with adaptive learning using ACE.
        
        Args:
            text: Input text to process
            context: Context information
            ground_truth: Ground truth for evaluation (optional)
            enable_reflection: Enable reflection and improvement
            enable_skill_update: Enable skill updates
            correlation_id: Correlation ID for tracking
            
        Returns:
            ACEIntegrationResult with processing results and learned skills
        """
        correlation_id = correlation_id or f"ace_adaptive_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting ACE adaptive processing",
            "text_length": len(text),
            "context_length": len(context),
            "enable_reflection": enable_reflection,
            "enable_skill_update": enable_skill_update,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.agent or not self.reflector:
                raise RuntimeError("ACE components not initialized")

            # Import Sample and SimpleEnvironment - they may be patched by tests
            try:
                # Use module-level imports (which can be patched by tests)
                from knowledge_engine.integrations.agentic_context_integration import Sample, SimpleEnvironment
            except ImportError:
                raise RuntimeError("ACE Sample/SimpleEnvironment not available")

            # Create a sample for ACE processing
            sample = Sample(
                question=text,
                context=context,
                ground_truth=ground_truth
            )

            # Create a simple environment for evaluation
            environment = SimpleEnvironment()
            
            # Process with ACE components
            agent_output = self.agent.generate(
                question=sample.question,
                context=sample.context,
                skillbook=self.skillbook,
                reflection="" if not enable_reflection else self._get_recent_reflections()
            )
            
            # Evaluate the output
            env_result = environment.evaluate(sample, agent_output)
            
            reflection_output = None
            skill_manager_output = None
            
            if enable_reflection:
                reflection_output = self.reflector.reflect(
                    question=sample.question,
                    agent_output=agent_output,
                    skillbook=self.skillbook,
                    ground_truth=env_result.ground_truth,
                    feedback=env_result.feedback,
                    max_refinement_rounds=self.config.get("max_refinement_rounds", 1)
                )
            
            if enable_skill_update and reflection_output:
                skill_manager_output = self.skill_manager.update_skills(
                    reflection=reflection_output,
                    skillbook=self.skillbook,
                    question_context=f"Question: {sample.question}\nContext: {sample.context}\nFeedback: {env_result.feedback}",
                    progress=f"Processing sample with ACE"
                )
            
            # Extract entities and relations from the output
            entities, relations = self._extract_entities_and_relations_from_output(agent_output)
            
            # Get current skills
            current_skills = []
            if hasattr(self.skillbook, 'get_skills'):
                current_skills = self.skillbook.get_skills()
            elif hasattr(self.skillbook, 'skills'):
                current_skills = self.skillbook.skills()
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = ACEIntegrationResult(
                success=True,
                entities=entities,
                relations=relations,
                skills=current_skills,
                metadata={
                    "model_used": self.config.get("model", "unknown"),
                    "reflection_enabled": enable_reflection,
                    "skill_update_enabled": enable_skill_update,
                    "ground_truth_provided": ground_truth is not None,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "ACE adaptive processing completed",
                "correlation_id": correlation_id,
                "entities_count": len(entities),
                "relations_count": len(relations),
                "skills_count": len(current_skills),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "ACE adaptive processing failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return ACEIntegrationResult(
                success=False,
                entities=[],
                relations=[],
                skills=[],
                metadata={"processing_time_ms": processing_time_ms},
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    def _extract_entities_and_relations_from_output(self, agent_output) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entities and relations from agent output.
        
        Args:
            agent_output: Output from ACE agent
            
        Returns:
            Tuple of (entities, relations)
        """
        entities = []
        relations = []
        
        # Extract from the final answer using simple pattern matching
        answer_text = getattr(agent_output, 'final_answer', '')
        
        # This is a simplified extraction - in a real implementation, 
        # this would use more sophisticated NLP techniques
        import re
        
        # Extract potential entities (capitalized words/phrases)
        entity_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Person names (John Doe)
            r'\b[A-Z][A-Z]+\b',  # Organizations (NASA, FBI)
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*(?:\s+(?:Inc|Corp|LLC|Ltd|Company|University|College|Hospital|Government|Department|Agency|Board|Institute|Lab|Center|Council|Association|Society|Union|Party|Movement|Company|Corp|Ltd|GmbH|SA|BV|Pty|LLP|LLC|Inc\.?))\b',  # Organizations
        ]
        
        seen_entities = set()
        for pattern in entity_patterns:
            matches = re.findall(pattern, answer_text)
            for match in matches:
                if match not in seen_entities:
                    entities.append({
                        "name": match.strip(),
                        "type": self._infer_entity_type(match),
                        "confidence": 0.8,
                        "position": answer_text.find(match)
                    })
                    seen_entities.add(match)
        
        # Extract potential relations (simple pattern matching)
        relation_patterns = [
            r'(\w+)\s+(?:is|was|are|were)\s+(?:a|an|the)\s+(\w+)',  # X is a Y
            r'(\w+)\s+(?:works|worked)\s+(?:at|for)\s+(\w+)',  # X works at Y
            r'(\w+)\s+(?:located|based)\s+(?:in|at)\s+(\w+)',  # X located in Y
        ]
        
        for pattern in relation_patterns:
            matches = re.findall(pattern, answer_text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    relations.append({
                        "subject": match[0],
                        "predicate": "related_to",  # Default predicate
                        "object": match[1],
                        "confidence": 0.7
                    })
        
        return entities, relations
    
    def _infer_entity_type(self, entity: str) -> str:
        """Infer entity type based on pattern."""
        entity_lower = entity.lower()
        
        # Check for common organization patterns
        org_patterns = ['inc', 'corp', 'ltd', 'llc', 'company', 'university', 'college', 'hospital', 'department', 'agency']
        if any(pattern in entity_lower for pattern in org_patterns):
            return "ORGANIZATION"
        
        # Check for person patterns (proper names)
        words = entity.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words):
            return "PERSON"
        
        # Default to entity
        return "ENTITY"
    
    def _get_recent_reflections(self) -> str:
        """Get recent reflections for context."""
        # This would be implemented based on the actual ACE implementation
        # For now, return an empty string
        return ""
    
    async def train_offline(
        self,
        training_samples: List[Dict[str, Any]],
        epochs: int = 3,
        correlation_id: Optional[str] = None
    ) -> ACEIntegrationResult:
        """
        Train the ACE system offline with provided samples.
        
        Args:
            training_samples: List of training samples with 'text' and 'ground_truth'
            epochs: Number of training epochs
            correlation_id: Correlation ID for tracking
            
        Returns:
            ACEIntegrationResult with training results
        """
        correlation_id = correlation_id or f"ace_train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        if not self.offline_ace:
            raise RuntimeError("Offline ACE not available")
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting ACE offline training",
            "samples_count": len(training_samples),
            "epochs": epochs,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Convert training samples to ACE Sample format
            ace_samples = []
            for sample_data in training_samples:
                text = sample_data.get('text', '')
                ground_truth = sample_data.get('ground_truth', '')
                context = sample_data.get('context', '')
                
                ace_sample = Sample(
                    question=text,
                    context=context,
                    ground_truth=ground_truth
                )
                ace_samples.append(ace_sample)
            
            # Create a simple environment for training
            from ace import SimpleEnvironment
            environment = SimpleEnvironment()
            
            # Run offline training
            results = self.offline_ace.run(
                samples=ace_samples,
                environment=environment,
                epochs=epochs
            )
            
            # Extract entities and relations from training results
            entities = []
            relations = []
            skills = []
            
            # Get current skills after training
            if hasattr(self.skillbook, 'get_skills'):
                skills = self.skillbook.get_skills()
            elif hasattr(self.skillbook, 'skills'):
                skills = self.skillbook.skills()
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = ACEIntegrationResult(
                success=True,
                entities=entities,
                relations=relations,
                skills=skills,
                metadata={
                    "training_samples": len(training_samples),
                    "epochs": epochs,
                    "results_count": len(results),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "ACE offline training completed",
                "correlation_id": correlation_id,
                "training_samples": len(training_samples),
                "epochs": epochs,
                "skills_count": len(skills),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "ACE offline training failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return ACEIntegrationResult(
                success=False,
                entities=[],
                relations=[],
                skills=[],
                metadata={"processing_time_ms": processing_time_ms},
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def process_online(
        self,
        text: str,
        context: str = "",
        ground_truth: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> ACEIntegrationResult:
        """
        Process text with online learning using ACE.
        
        Args:
            text: Input text to process
            context: Context information
            ground_truth: Ground truth for evaluation (optional)
            correlation_id: Correlation ID for tracking
            
        Returns:
            ACEIntegrationResult with processing results
        """
        correlation_id = correlation_id or f"ace_online_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        if not self.online_ace:
            raise RuntimeError("Online ACE not available")
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting ACE online processing",
            "text_length": len(text),
            "context_length": len(context),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Create a sample for ACE processing
            sample = Sample(
                question=text,
                context=context,
                ground_truth=ground_truth
            )
            
            # Create a simple environment for evaluation
            from ace import SimpleEnvironment
            environment = SimpleEnvironment()
            
            # Process with online ACE
            results = self.online_ace.run(
                samples=[sample],
                environment=environment
            )
            
            # Extract entities and relations from results
            entities = []
            relations = []
            
            if results:
                # Extract from the first result
                result = results[0]
                if hasattr(result, 'agent_output'):
                    entities, relations = self._extract_entities_and_relations_from_output(result.agent_output)
            
            # Get current skills
            current_skills = []
            if hasattr(self.skillbook, 'get_skills'):
                current_skills = self.skillbook.get_skills()
            elif hasattr(self.skillbook, 'skills'):
                current_skills = self.skillbook.skills()
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            ace_result = ACEIntegrationResult(
                success=True,
                entities=entities,
                relations=relations,
                skills=current_skills,
                metadata={
                    "model_used": self.config.get("model", "unknown"),
                    "results_count": len(results),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "ACE online processing completed",
                "correlation_id": correlation_id,
                "entities_count": len(entities),
                "relations_count": len(relations),
                "skills_count": len(current_skills),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return ace_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "ACE online processing failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return ACEIntegrationResult(
                success=False,
                entities=[],
                relations=[],
                skills=[],
                metadata={"processing_time_ms": processing_time_ms},
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def get_skillbook_state(self) -> Dict[str, Any]:
        """
        Get the current state of the skillbook.
        
        Returns:
            Dictionary with skillbook state information
        """
        if not self.skillbook:
            return {"error": "Skillbook not initialized", "timestamp": datetime.now(timezone.utc).isoformat()}
        
        try:
            skills = []
            if hasattr(self.skillbook, 'get_skills'):
                skills = self.skillbook.get_skills()
            elif hasattr(self.skillbook, 'skills'):
                skills = self.skillbook.skills()
            
            state = {
                "skill_count": len(skills),
                "skills": skills,
                "prompt_representation": self.skillbook.as_prompt() if hasattr(self.skillbook, 'as_prompt') else "No prompt representation",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return state
        except Exception as e:
            logger.error({
                "msg": "Failed to get skillbook state",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
    
    async def reset_skillbook(self):
        """Reset the skillbook to initial state."""
        try:
            from ace import Skillbook
            self.skillbook = Skillbook()
            
            logger.info({
                "msg": "Skillbook reset to initial state",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error({
                "msg": "Failed to reset skillbook",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise
    
    def get_ace_status(self) -> Dict[str, Any]:
        """
        Get the status of the ACE integration.
        
        Returns:
            Dictionary with integration status
        """
        return {
            "available": self.agent is not None and self.reflector is not None,
            "offline_ace_available": self.offline_ace is not None,
            "online_ace_available": self.online_ace is not None,
            "skillbook_initialized": self.skillbook is not None,
            "model": self.config.get("model", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def close(self):
        """Close resources used by the integration."""
        logger.info({
            "msg": "Closing ACE integration resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # If we have async learning running, stop it
        if self.offline_ace and hasattr(self.offline_ace, 'stop_async_learning'):
            self.offline_ace.stop_async_learning()
        
        if self.online_ace and hasattr(self.online_ace, 'stop_async_learning'):
            self.online_ace.stop_async_learning()
        
        logger.info({
            "msg": "ACE integration resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

# Availability flag
try:
    from ace import SkillBook
    ACE_INTEGRATION_AVAILABLE = True
except ImportError:
    ACE_INTEGRATION_AVAILABLE = False
