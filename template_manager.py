import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class WorkflowTemplate:
    """Represents a configurable workflow template.
    """
    name: str
    description: Optional[str]
    parameters: Dict[str, Any]

class TemplateManager:
    """
    Manages the persistent storage and retrieval of workflow templates.
    Templates are stored as JSON files in a dedicated directory.
    """
    def __init__(self, base_dir: str = "templates"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_template_path(self, template_name: str) -> str:
        """Returns the file path for a given template name."""
        return os.path.join(self.base_dir, f"{template_name.replace(' ', '_').lower()}.json")

    def save_template(self, template: WorkflowTemplate):
        """Saves a workflow template to a JSON file."""
        file_path = self._get_template_path(template.name)
        try:
            with open(file_path, 'w') as f:
                json.dump(asdict(template), f, indent=2)
        except IOError as e:
            print(f"Error saving template {template.name}: {e}")

    def load_template(self, template_name: str) -> Optional[WorkflowTemplate]:
        """Loads a workflow template from a JSON file."""
        file_path = self._get_template_path(template_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    return WorkflowTemplate(**data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading template {template_name}: Invalid JSON format or missing keys: {e}")
            except IOError as e:
                print(f"Error reading template file {template_name}: {e}")
        return None

    def get_all_templates(self) -> List[WorkflowTemplate]:
        """Retrieves all saved workflow templates."""
        templates = []
        for filename in os.listdir(self.base_dir):
            if filename.endswith(".json"):
                template_name = filename.replace(".json", "").replace('_', ' ').title()
                template = self.load_template(template_name)
                if template:
                    templates.append(template)
        return templates

    def delete_template(self, template_name: str):
        """Deletes a workflow template file."""
        file_path = self._get_template_path(template_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting template file {template_name}: {e}")