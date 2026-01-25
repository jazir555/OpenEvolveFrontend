"""
Curie Bridge - Integration with SOP Generator and Validation Systems

This module provides the bridge between Curie's experimentation framework
and OpenEvolve's SOP Generator and validation systems.

Key Integration Points:
- SOP Generator for experiment protocol generation
- Validation systems for result verification
- Statistical framework for analysis
- Reflection-based refinement

Author: Agent 3 (Curie Integration Specialist)
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from integrations.base.experimentation_interface import (
    ExperimentProtocol,
    ExperimentDomain
)


logger = logging.getLogger(__name__)


class CurieBridge:
    """
    Bridge between Curie and OpenEvolve systems.

    This bridge handles:
    - Protocol generation using SOP Generator
    - Protocol execution and simulation
    - Result validation
    - Template management
    """

    def __init__(
        self,
        openai_api_key: str,
        workspace_dir: str = "./curie_workspace",
        cache_enabled: bool = True
    ):
        """
        Initialize Curie bridge.

        Args:
            openai_api_key: OpenAI API key for LLM operations
            workspace_dir: Workspace directory for experiments
            cache_enabled: Enable result caching
        """
        self.openai_api_key = openai_api_key
        self.workspace_dir = Path(workspace_dir)
        self.cache_enabled = cache_enabled
        self._cache = {}
        self._templates = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the bridge"""
        if self._initialized:
            return

        logger.info("Initializing Curie bridge...")

        # Create workspace directories
        os.makedirs(self.workspace_dir / "protocols", exist_ok=True)
        os.makedirs(self.workspace_dir / "results", exist_ok=True)
        os.makedirs(self.workspace_dir / "logs", exist_ok=True)

        # Load experiment templates
        await self._load_all_templates()

        self._initialized = True
        logger.info("Curie bridge initialization complete")

    async def _load_all_templates(self) -> None:
        """Load all experiment templates"""
        template_dir = Path(__file__).parent / "templates"

        if not template_dir.exists():
            logger.warning(f"Template directory not found: {template_dir}")
            return

        for template_file in template_dir.glob("*.yaml"):
            domain = template_file.stem
            try:
                with open(template_file, 'r') as f:
                    self._templates[domain] = yaml.safe_load(f)
                logger.info(f"Loaded template for domain: {domain}")
            except Exception as e:
                logger.error(f"Failed to load template {domain}: {e}")

    async def generate_protocol(
        self,
        hypothesis: str,
        domain: str,
        constraints: List[str],
        available_equipment: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate experimental protocol using SOP Generator integration.

        Args:
            hypothesis: Hypothesis statement
            domain: Scientific domain
            constraints: Experimental constraints
            available_equipment: Available equipment

        Returns:
            List of protocol steps
        """
        logger.info(f"Generating protocol for domain: {domain}")

        # Check cache
        cache_key = f"protocol:{hash(hypothesis)}:{domain}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        # Try to load template-based protocol
        if domain in self._templates:
            protocol = await self._generate_protocol_from_template(
                hypothesis,
                domain,
                constraints,
                available_equipment
            )
        else:
            # Fallback to LLM-based protocol generation
            protocol = await self._generate_protocol_with_llm(
                hypothesis,
                domain,
                constraints,
                available_equipment
            )

        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = protocol

        return protocol

    async def _generate_protocol_from_template(
        self,
        hypothesis: str,
        domain: str,
        constraints: List[str],
        available_equipment: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate protocol from domain template"""
        template = self._templates[domain]

        # Get protocol template
        protocol_template = template.get("protocol_template", [])

        # Customize protocol based on hypothesis and constraints
        protocol = []
        for i, step_template in enumerate(protocol_template):
            step = {
                "step_number": i + 1,
                "title": step_template.get("title", f"Step {i+1}"),
                "description": step_template.get("description", ""),
                "action": step_template.get("action", ""),
                "parameters": self._customize_parameters(
                    step_template.get("parameters", {}),
                    hypothesis,
                    constraints
                ),
                "materials": step_template.get("materials", []),
                "equipment": self._filter_equipment(
                    step_template.get("equipment", []),
                    available_equipment
                ),
                "duration": step_template.get("duration", 300),
                "safety_notes": step_template.get("safety_notes", []),
                "validation_criteria": step_template.get("validation_criteria", {})
            }
            protocol.append(step)

        # Save generated protocol
        protocol_id = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        protocol_file = self.workspace_dir / "protocols" / f"{protocol_id}.json"
        with open(protocol_file, 'w') as f:
            json.dump({
                "protocol_id": protocol_id,
                "hypothesis": hypothesis,
                "domain": domain,
                "steps": protocol,
                "generated_at": datetime.now().isoformat()
            }, f, indent=2)

        logger.info(f"Generated protocol from template: {protocol_id}")
        return protocol

    async def _generate_protocol_with_llm(
        self,
        hypothesis: str,
        domain: str,
        constraints: List[str],
        available_equipment: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate protocol using LLM"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not available, using fallback protocol")
            return self._fallback_protocol(hypothesis, domain)

        prompt = f"""
Generate a detailed experimental protocol to test this hypothesis:

Hypothesis: {hypothesis}
Domain: {domain}
Constraints: {', '.join(constraints) if constraints else 'None'}
Available Equipment: {', '.join(available_equipment) if available_equipment else 'Standard lab equipment'}

Generate a step-by-step protocol with:
1. Clear experimental steps
2. Required materials and equipment
3. Parameters to control
4. Safety considerations
5. Validation criteria

Return as JSON array of steps:
[
  {{
    "step_number": 1,
    "title": "Step title",
    "description": "Detailed description",
    "action": "Action to perform",
    "parameters": {{"param1": "value1"}},
    "materials": ["material1", "material2"],
    "equipment": ["equipment1"],
    "duration": 300,
    "safety_notes": ["note1"],
    "validation_criteria": {{"metric": "target"}}
  }}
]
"""

        try:
            openai.api_key = self.openai_api_key
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert experimental protocol generator. Generate detailed, scientifically rigorous protocols."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                timeout=30
            )

            protocol = json.loads(response.choices[0].message.content)
            logger.info(f"Generated protocol with LLM: {len(protocol)} steps")
            return protocol

        except Exception as e:
            logger.error(f"Failed to generate protocol with LLM: {e}")
            return self._fallback_protocol(hypothesis, domain)

    def _fallback_protocol(self, hypothesis: str, domain: str) -> List[Dict[str, Any]]:
        """Generate fallback protocol when LLM unavailable"""
        return [
            {
                "step_number": 1,
                "title": "Experimental Setup",
                "description": f"Set up experiment to test: {hypothesis}",
                "action": "Configure experimental apparatus",
                "parameters": {},
                "materials": [],
                "equipment": [],
                "duration": 600,
                "safety_notes": ["Follow standard safety procedures"],
                "validation_criteria": {}
            },
            {
                "step_number": 2,
                "title": "Data Collection",
                "description": "Collect experimental data",
                "action": "Execute experimental protocol",
                "parameters": {},
                "materials": [],
                "equipment": [],
                "duration": 3600,
                "safety_notes": [],
                "validation_criteria": {}
            },
            {
                "step_number": 3,
                "title": "Analysis",
                "description": "Analyze collected data",
                "action": "Process and analyze results",
                "parameters": {},
                "materials": [],
                "equipment": [],
                "duration": 1800,
                "safety_notes": [],
                "validation_criteria": {}
            }
        ]

    def _customize_parameters(
        self,
        template_params: Dict[str, Any],
        hypothesis: str,
        constraints: List[str]
    ) -> Dict[str, Any]:
        """Customize parameters based on hypothesis and constraints"""
        # For now, return template parameters as-is
        # In production, use LLM to customize based on hypothesis
        return template_params

    def _filter_equipment(
        self,
        template_equipment: List[str],
        available_equipment: List[str]
    ) -> List[str]:
        """Filter equipment based on availability"""
        if not available_equipment:
            return template_equipment

        # Return equipment that's either available or in template
        return [
            eq for eq in template_equipment
            if eq in available_equipment or "standard" in eq.lower()
        ]

    async def execute_protocol(
        self,
        protocol: ExperimentProtocol,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Execute experimental protocol.

        In production, this would interface with real laboratory equipment
        or simulation frameworks. For now, it simulates execution.

        Args:
            protocol: Protocol to execute
            iteration: Iteration number

        Returns:
            Execution results
        """
        logger.info(f"Executing protocol {protocol.protocol_id}, iteration {iteration}")

        # Simulate protocol execution
        await asyncio.sleep(0.1)  # Simulate work

        # Generate simulated results
        results = {
            "iteration": iteration,
            "data": self._simulate_data(protocol),
            "observations": [
                f"Experiment {iteration} completed successfully",
                "All steps executed according to protocol"
            ],
            "execution_time": protocol.duration_estimate,
            "timestamp": datetime.now().isoformat()
        }

        # Save results
        result_file = self.workspace_dir / "results" / f"{protocol.protocol_id}_iter{iteration}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _simulate_data(self, protocol: ExperimentProtocol) -> Dict[str, Any]:
        """Simulate experimental data"""
        # Generate simulated data based on hypothesis variables
        data = {}

        for var in protocol.hypothesis.independent_variables:
            data[var] = {
                "values": [1.0, 2.0, 3.0, 4.0, 5.0],
                "unit": "arbitrary",
                "uncertainty": 0.1
            }

        for var in protocol.hypothesis.dependent_variables:
            data[var] = {
                "values": [2.0, 4.0, 6.0, 8.0, 10.0],
                "unit": "arbitrary",
                "uncertainty": 0.2
            }

        return data

    async def validate_results(
        self,
        results: Dict[str, Any],
        protocol: ExperimentProtocol
    ) -> Dict[str, Any]:
        """
        Validate experimental results.

        Args:
            results: Experimental results
            protocol: Protocol that was executed

        Returns:
            Validation report
        """
        validation_report = {
            "valid": True,
            "checks_performed": [],
            "issues": [],
            "warnings": []
        }

        # Check if data exists
        if "data" in results and len(results["data"]) > 0:
            validation_report["checks_performed"].append("Data presence check: PASSED")
        else:
            validation_report["valid"] = False
            validation_report["issues"].append("No data collected")

        # Check reproducibility
        if "observations" in results:
            validation_report["checks_performed"].append("Observations recorded: PASSED")
        else:
            validation_report["warnings"].append("No observations recorded")

        # Log validation
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "protocol_id": protocol.protocol_id,
            "validation": validation_report
        }

        log_file = self.workspace_dir / "logs" / "validation_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

        return validation_report

    async def get_template(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get experiment template for domain"""
        return self._templates.get(domain)

    async def list_supported_domains(self) -> List[str]:
        """List supported experiment domains"""
        return list(self._templates.keys())

    async def shutdown(self) -> None:
        """Shutdown the bridge and cleanup"""
        logger.info("Shutting down Curie bridge")

        # Clear cache
        if self.cache_enabled:
            self._cache.clear()

        self._initialized = False

        logger.info("Curie bridge shutdown complete")

    async def validate(self) -> Dict[str, Any]:
        """Validate bridge configuration"""
        issues = []

        # Check OpenAI availability
        if not OPENAI_AVAILABLE:
            issues.append("OpenAI library not available")

        # Check workspace
        if not self.workspace_dir.exists():
            issues.append(f"Workspace directory not found: {self.workspace_dir}")

        # Check templates
        if not self._templates:
            issues.append("No experiment templates loaded")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "templates_loaded": len(self._templates),
            "supported_domains": list(self._templates.keys())
        }
