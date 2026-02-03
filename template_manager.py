"""
Template Manager Module

This module handles workflow configuration templates, allowing users to save,
load, and manage reusable workflow configurations.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

# **ACTUAL INTEGRATION**: Adaptive MDAP for template complexity analysis
try:
    from adaptive_mdap import TaskComplexityClassifier
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    SubProblem = None


class TemplateManager:
    """Manages workflow configuration templates."""
    
    def __init__(self, storage_path: str = "./workflow_templates"):
        """
        Initialize the Template Manager.
        
        Args:
            storage_path: Path to store template files
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.templates: Dict[str, Dict[str, Any]] = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all templates from storage."""
        templates = {}
        
        if not os.path.exists(self.storage_path):
            return templates
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                filepath = os.path.join(self.storage_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        template = json.load(f)
                        templates[template["id"]] = template
                except (OSError, IOError, json.JSONDecodeError, KeyError) as e:
                    print(f"Error loading template {filename}: {e}")
        
        return templates
    
    def _save_template(self, template: Dict[str, Any]) -> None:
        """Save a template to storage."""
        filepath = os.path.join(self.storage_path, f"{template['id']}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(template, f, indent=2)
        except (OSError, IOError, TypeError) as e:
            print(f"Error saving template: {e}")
            raise
    
    def create_template(
        self,
        name: str,
        description: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a new template.
        
        Args:
            name: Template name
            description: Template description
            config: Workflow configuration
            tags: Optional tags for categorization
            
        Returns:
            Template ID
        """
        # Generate template ID
        template_id = hashlib.md5(f"{name}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        template = {
            "id": template_id,
            "name": name,
            "description": description,
            "version": "1.0",
            "config": config,
            "usage_count": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "tags": tags or []
        }
        
        self.templates[template_id] = template
        self._save_template(template)
        
        return template_id
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a template by ID.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template dictionary or None if not found
        """
        return self.templates.get(template_id)
    
    def get_all_templates(self) -> List[Dict[str, Any]]:
        """
        Get all templates.
        
        Returns:
            List of template dictionaries
        """
        return list(self.templates.values())
    
    def update_template(
        self,
        template_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Update an existing template.
        
        Args:
            template_id: Template ID
            name: New name (optional)
            description: New description (optional)
            config: New configuration (optional)
            tags: New tags (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if template_id not in self.templates:
            return False
        
        template = self.templates[template_id]
        
        if name is not None:
            template["name"] = name
        if description is not None:
            template["description"] = description
        if config is not None:
            template["config"] = config
        if tags is not None:
            template["tags"] = tags
        
        template["updated_at"] = datetime.now().isoformat()
        
        self._save_template(template)
        return True
    
    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template.
        
        Args:
            template_id: Template ID
            
        Returns:
            True if successful, False otherwise
        """
        if template_id not in self.templates:
            return False
        
        # Remove from memory
        del self.templates[template_id]
        
        # Remove from storage
        filepath = os.path.join(self.storage_path, f"{template_id}.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except (OSError, IOError) as e:
            print(f"Error deleting template: {e}")
            return False
    
    def increment_usage(self, template_id: str) -> None:
        """
        Increment the usage count for a template.
        
        Args:
            template_id: Template ID
        """
        if template_id in self.templates:
            self.templates[template_id]["usage_count"] += 1
            self._save_template(self.templates[template_id])
    
    def export_template(self, template_id: str) -> Optional[str]:
        """
        Export a template as JSON string.
        
        Args:
            template_id: Template ID
            
        Returns:
            JSON string or None if template not found
        """
        template = self.get_template(template_id)
        if template:
            return json.dumps(template, indent=2)
        return None
    
    def import_template(self, template_json: str) -> Optional[str]:
        """
        Import a template from JSON string.
        
        Args:
            template_json: JSON string containing template
            
        Returns:
            Template ID if successful, None otherwise
        """
        try:
            template = json.loads(template_json)
            
            # Validate required fields
            required_fields = ["name", "description", "config"]
            if not all(field in template for field in required_fields):
                raise ValueError("Missing required fields")
            
            # Generate new ID to avoid conflicts
            template_id = hashlib.md5(
                f"{template['name']}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            
            template["id"] = template_id
            template["created_at"] = datetime.now().isoformat()
            template["updated_at"] = datetime.now().isoformat()
            template["usage_count"] = 0
            
            self.templates[template_id] = template
            self._save_template(template)
            
            return template_id
        
        except (OSError, IOError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error importing template: {e}")
            return None
    
    def validate_template(self, template: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a template configuration.
        
        Args:
            template: Template dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        required_fields = ["name", "description", "config"]
        for field in required_fields:
            if field not in template:
                errors.append(f"Missing required field: {field}")
        
        # Validate config structure
        if "config" in template:
            config = template["config"]
            
            # Check for required config fields
            if "max_depth" not in config:
                errors.append("Config missing 'max_depth'")
            
            if "teams" not in config:
                errors.append("Config missing 'teams'")
            
            if "gauntlets" not in config:
                errors.append("Config missing 'gauntlets'")
        
        return len(errors) == 0, errors
    
    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """
        Search templates by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching templates
        """
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            if (query_lower in template["name"].lower() or
                query_lower in template["description"].lower() or
                any(query_lower in tag.lower() for tag in template.get("tags", []))):
                results.append(template)
        
        return results

    
    def add_openevolve_preset_templates(self):
        """Add comprehensive OpenEvolve configuration preset templates"""
        presets = {
            "fast": {
                "name": "Fast Evolution",
                "description": "Quick evolution with minimal iterations for rapid prototyping",
                "tags": ["openevolve", "fast", "preset"],
                "config": {
                    "max_iterations": 5,
                    "population_size": 10,
                    "archive_size": 50,
                    "temperature": 0.5,
                    "elite_ratio": 0.2,
                    "exploration_ratio": 0.3,
                    "exploitation_ratio": 0.5,
                    "checkpoint_interval": 5,
                    "enable_artifacts": False,
                    "enable_cascade_evaluation": False,
                    "parallel_evaluations": 2
                }
            },
            "balanced": {
                "name": "Balanced Evolution",
                "description": "Balanced performance and quality for general use",
                "tags": ["openevolve", "balanced", "preset"],
                "config": {
                    "max_iterations": 20,
                    "population_size": 30,
                    "archive_size": 100,
                    "temperature": 0.7,
                    "elite_ratio": 0.15,
                    "exploration_ratio": 0.35,
                    "exploitation_ratio": 0.5,
                    "checkpoint_interval": 10,
                    "enable_artifacts": True,
                    "enable_cascade_evaluation": True,
                    "cascade_thresholds": [0.5, 0.75, 0.9],
                    "parallel_evaluations": 4
                }
            },
            "thorough": {
                "name": "Thorough Evolution",
                "description": "Thorough exploration with high quality for production use",
                "tags": ["openevolve", "thorough", "preset"],
                "config": {
                    "max_iterations": 50,
                    "population_size": 50,
                    "archive_size": 200,
                    "temperature": 0.8,
                    "elite_ratio": 0.1,
                    "exploration_ratio": 0.4,
                    "exploitation_ratio": 0.5,
                    "checkpoint_interval": 10,
                    "enable_artifacts": True,
                    "enable_cascade_evaluation": True,
                    "cascade_thresholds": [0.6, 0.8, 0.95],
                    "parallel_evaluations": 8,
                    "enable_quality_diversity": True,
                    "feature_dimensions": ["complexity", "novelty", "quality"],
                    "feature_bins": 10
                }
            },
            "research": {
                "name": "Research Evolution",
                "description": "Research mode with all features enabled for maximum exploration",
                "tags": ["openevolve", "research", "preset"],
                "config": {
                    "max_iterations": 100,
                    "population_size": 100,
                    "archive_size": 500,
                    "temperature": 0.9,
                    "elite_ratio": 0.05,
                    "exploration_ratio": 0.5,
                    "exploitation_ratio": 0.45,
                    "checkpoint_interval": 5,
                    "enable_artifacts": True,
                    "enable_cascade_evaluation": True,
                    "cascade_thresholds": [0.5, 0.7, 0.85, 0.95],
                    "parallel_evaluations": 16,
                    "enable_quality_diversity": True,
                    "feature_dimensions": ["complexity", "novelty", "quality", "diversity"],
                    "feature_bins": 20,
                    "enable_island_model": True,
                    "num_islands": 4,
                    "migration_interval": 10,
                    "migration_size": 5,
                    "enable_meta_prompting": True,
                    "enable_template_stochasticity": True
                }
            },
            "quality_diversity": {
                "name": "Quality Diversity",
                "description": "Focus on diverse, high-quality solutions using MAP-Elites",
                "tags": ["openevolve", "quality-diversity", "preset"],
                "config": {
                    "max_iterations": 30,
                    "population_size": 40,
                    "archive_size": 300,
                    "temperature": 0.75,
                    "elite_ratio": 0.1,
                    "exploration_ratio": 0.5,
                    "exploitation_ratio": 0.4,
                    "checkpoint_interval": 10,
                    "enable_quality_diversity": True,
                    "feature_dimensions": ["complexity", "novelty", "readability"],
                    "feature_bins": 15,
                    "enable_artifacts": True,
                    "parallel_evaluations": 6
                }
            },
            "ensemble": {
                "name": "Ensemble Evaluation",
                "description": "Use ensemble of evaluators for robust assessment",
                "tags": ["openevolve", "ensemble", "preset"],
                "config": {
                    "max_iterations": 15,
                    "population_size": 25,
                    "archive_size": 100,
                    "temperature": 0.7,
                    "enable_cascade_evaluation": True,
                    "cascade_thresholds": [0.6, 0.8, 0.95],
                    "parallel_evaluations": 5,
                    "ensemble_size": 5,
                    "consensus_threshold": 0.7,
                    "enable_artifacts": True
                }
            }
        }
        
        for preset_id, preset_data in presets.items():
            template_id = f"openevolve_{preset_id}"
            if template_id not in self.templates:
                self.create_template(
                    name=preset_data["name"],
                    description=preset_data["description"],
                    config=preset_data["config"],
                    tags=preset_data["tags"]
                )
    
    def get_openevolve_template(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get an OpenEvolve preset template by name.
        
        Args:
            preset_name: Preset name (fast, balanced, thorough, research, quality_diversity, ensemble)
            
        Returns:
            Template configuration or None if not found
        """
        template_id = f"openevolve_{preset_name}"
        return self.get_template(template_id)
    
    def validate_openevolve_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate an OpenEvolve configuration.
        
        Args:
            config: OpenEvolve configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        required_fields = ["max_iterations", "population_size"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate numeric ranges
        if "max_iterations" in config:
            if not isinstance(config["max_iterations"], int) or config["max_iterations"] < 1:
                errors.append("max_iterations must be a positive integer")
        
        if "population_size" in config:
            if not isinstance(config["population_size"], int) or config["population_size"] < 1:
                errors.append("population_size must be a positive integer")
        
        if "temperature" in config:
            if not 0 <= config["temperature"] <= 2:
                errors.append("temperature must be between 0 and 2")
        
        if "elite_ratio" in config:
            if not 0 <= config["elite_ratio"] <= 1:
                errors.append("elite_ratio must be between 0 and 1")
        
        if "exploration_ratio" in config:
            if not 0 <= config["exploration_ratio"] <= 1:
                errors.append("exploration_ratio must be between 0 and 1")
        
        if "exploitation_ratio" in config:
            if not 0 <= config["exploitation_ratio"] <= 1:
                errors.append("exploitation_ratio must be between 0 and 1")
        
        # Check that ratios sum to approximately 1
        if all(k in config for k in ["elite_ratio", "exploration_ratio", "exploitation_ratio"]):
            ratio_sum = config["elite_ratio"] + config["exploration_ratio"] + config["exploitation_ratio"]
            if not 0.99 <= ratio_sum <= 1.01:
                errors.append(f"elite_ratio + exploration_ratio + exploitation_ratio must sum to 1.0 (got {ratio_sum})")
        
        # Validate quality diversity settings
        if config.get("enable_quality_diversity"):
            if "feature_dimensions" not in config:
                errors.append("feature_dimensions required when enable_quality_diversity is True")
            if "feature_bins" not in config:
                errors.append("feature_bins required when enable_quality_diversity is True")
        
        # Validate cascade evaluation settings
        if config.get("enable_cascade_evaluation"):
            if "cascade_thresholds" not in config:
                errors.append("cascade_thresholds required when enable_cascade_evaluation is True")
            elif not isinstance(config["cascade_thresholds"], list):
                errors.append("cascade_thresholds must be a list")
        
        # Validate island model settings
        if config.get("enable_island_model"):
            if "num_islands" not in config:
                errors.append("num_islands required when enable_island_model is True")
            if "migration_interval" not in config:
                errors.append("migration_interval required when enable_island_model is True")
        
        return len(errors) == 0, errors
    
    def create_custom_openevolve_template(
        self,
        name: str,
        description: str,
        base_preset: str = "balanced",
        overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a custom OpenEvolve template based on a preset.
        
        Args:
            name: Template name
            description: Template description
            base_preset: Base preset to start from (fast, balanced, thorough, research)
            overrides: Configuration overrides
            
        Returns:
            Template ID
        """
        # Get base preset
        base_template = self.get_openevolve_template(base_preset)
        if not base_template:
            raise ValueError(f"Base preset '{base_preset}' not found")
        
        # Create new config with overrides
        config = base_template["config"].copy()
        if overrides:
            config.update(overrides)
        
        # Validate the configuration
        is_valid, errors = self.validate_openevolve_config(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")
        
        # Create the template
        return self.create_template(
            name=name,
            description=description,
            config=config,
            tags=["openevolve", "custom"]
        )
