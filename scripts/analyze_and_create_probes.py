#!/usr/bin/env python3
"""
Intelligent Probe Generator for Core Projects
Analyzes each core project to determine its API nature and creates appropriate probes
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FRONTEND_DIR = Path(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend")
CORE_PROJECTS_DIR = FRONTEND_DIR / "core-projects"
ADAPTERS_DIR = FRONTEND_DIR / "glue" / "adapters"

# Projects needing probes
MISSING_PROJECTS = [
    "adaptive_mdap", "agentic-context-engine", "agentjson", "ai-knowledge-graph",
    "arbor", "causal-learn", "cav-nlp", "claudiomiro", "cognitive-hydraulics",
    "crewAI", "datapizza", "DeepKE", "deep-research-agent", "detllm", "drift",
    "dspy", "dspy-helm", "DTS", "Formal-Reasoning-Mode", "foundry",
    "Generic-Knowledge-Extraction-Tool", "guardrails", "Iterative-Contextual-Refinements",
    "jsonformer", "kg-gen", "lagrange-mapper", "Lean4-LLM-Ai-Agent-Mooc",
    "LoongFlow", "Matryoshka", "mrs-core", "NeuralKG", "neuromancer",
    "OneKE", "outlines", "PAMI", "pygraphistry", "rlm", "ROMA",
    "slither", "steer", "uqsa", "valkey"
]

class ProjectAnalyzer:
    """Analyzes a project to determine its integration characteristics"""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_dir = CORE_PROJECTS_DIR / project_name
        self.api_type = "unknown"
        self.default_port = None
        self.health_endpoint = None
        self.integration_type = "unknown"  # api, library, cli, hybrid
        self.language = "unknown"
        self.notes = []

    def analyze(self) -> Dict:
        """Main analysis method"""
        if not self.project_dir.exists():
            return {"status": "error", "message": f"Project directory not found: {self.project_dir}"}

        # Detect language
        self._detect_language()

        # Check for API server characteristics
        self._check_for_server()

        # Check for library characteristics
        self._check_for_library()

        # Check for CLI characteristics
        self._check_for_cli()

        # Look for configuration files
        self._parse_config_files()

        return {
            "project": self.project_name,
            "api_type": self.api_type,
            "integration_type": self.integration_type,
            "language": self.language,
            "default_port": self.default_port,
            "health_endpoint": self.health_endpoint,
            "notes": self.notes
        }

    def _detect_language(self):
        """Detect project language from file patterns"""
        py_files = list(self.project_dir.rglob("*.py"))
        js_files = list(self.project_dir.rglob("*.js"))
        ts_files = list(self.project_dir.rglob("*.ts"))
        rs_files = list(self.project_dir.rglob("*.rs"))
        go_files = list(self.project_dir.rglob("*.go"))

        if len(py_files) > 10:
            self.language = "python"
        elif len(ts_files) > 10 or len(js_files) > 10:
            self.language = "javascript"
        elif len(rs_files) > 10:
            self.language = "rust"
        elif len(go_files) > 10:
            self.language = "go"
        else:
            self.language = "mixed"

    def _check_for_server(self):
        """Check if project runs as an HTTP server"""
        # Look for common server indicators
        server_keywords = [
            "fastapi", "flask", "django", "tornado", "aiohttp",  # Python
            "express", "koa", "hapi", "nest",  # JavaScript/Node
            "actix", "rocket", "warp",  # Rust
            "gin", "echo", "fiber",  # Go
        ]

        # Check requirements.txt, package.json, Cargo.toml
        for config_file in ["requirements.txt", "package.json", "Cargo.toml", "go.mod"]:
            config_path = self.project_dir / config_file
            if config_path.exists():
                content = config_path.read_text(errors="ignore")
                for keyword in server_keywords:
                    if keyword.lower() in content.lower():
                        self.api_type = "http"
                        self.integration_type = "api"
                        self.notes.append(f"Detected server framework: {keyword}")
                        break

        # Check for port defaults in common files
        port_patterns = [
            (r"port\s*[=:]\s*(\d+)", None),
            (r"PORT\s*[=:]\s*(\d+)", None),
            (r"localhost:(\d+)", None),
            (r"0\.0\.0\.0:(\d+)", None),
            (r"host.*?(\d{4,5})", None),
        ]

        for file_path in self.project_dir.rglob("*.py"):
            if "config" in file_path.name.lower() or "main" in file_path.name.lower() or "server" in file_path.name.lower():
                try:
                    content = file_path.read_text(errors="ignore")
                    for pattern, _ in port_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            self.default_port = int(matches[0])
                            self.notes.append(f"Detected port: {self.default_port}")
                            break
                except:
                    pass

        # Set default port if not found
        if self.default_port is None and self.api_type == "http":
            self.default_port = 8080
            self.notes.append("Using default port 8080")

    def _check_for_library(self):
        """Check if project is a library/package"""
        lib_indicators = [
            self.project_dir / "setup.py",
            self.project_dir / "pyproject.toml",
            self.project_dir / "package.json",
            self.project_dir / "Cargo.toml",
        ]

        for indicator in lib_indicators:
            if indicator.exists():
                if self.integration_type == "unknown":
                    self.integration_type = "library"
                self.notes.append(f"Detected library: {indicator.name}")
                break

    def _check_for_cli(self):
        """Check if project provides CLI tools"""
        cli_indicators = [
            self.project_dir / "bin",
            self.project_dir / "cli",
        ]

        for indicator in cli_indicators:
            if indicator.exists() and indicator.is_dir():
                if self.integration_type == "library":
                    self.integration_type = "hybrid"
                elif self.integration_type == "unknown":
                    self.integration_type = "cli"
                self.notes.append(f"Detected CLI: {indicator.name}")
                break

        # Check for CLI entry points in pyproject.toml or setup.py
        for config_file in ["pyproject.toml", "setup.py"]:
            config_path = self.project_dir / config_file
            if config_path.exists():
                content = config_path.read_text(errors="ignore")
                if "console_scripts" in content or "cli" in content.lower():
                    if self.integration_type == "library":
                        self.integration_type = "hybrid"
                    elif self.integration_type == "unknown":
                        self.integration_type = "cli"
                    self.notes.append("Detected CLI entry points")
                    break

    def _parse_config_files(self):
        """Parse configuration files for API info"""
        # Try to find README
        readme_path = self.project_dir / "README.md"
        if readme_path.exists():
            content = readme_path.read_text(errors="ignore")

            # Look for port mentions
            port_matches = re.findall(r"port\s*:?\s*(\d+)", content, re.IGNORECASE)
            if port_matches and not self.default_port:
                self.default_port = int(port_matches[0])
                self.notes.append(f"Found port in README: {self.default_port}")

            # Look for endpoint mentions
            endpoint_matches = re.findall(r"/(api|health|ping|status|v1)", content, re.IGNORECASE)
            if endpoint_matches and not self.health_endpoint:
                self.health_endpoint = f"/{endpoint_matches[0]}"
                self.notes.append(f"Found endpoint in README: {self.health_endpoint}")

        # Check for docker-compose
        docker_compose = self.project_dir / "docker-compose.yml"
        if docker_compose.exists():
            content = docker_compose.read_text(errors="ignore")
            port_matches = re.findall(r"(\d+):\d+", content)
            if port_matches:
                self.default_port = int(port_matches[0])
                self.notes.append(f"Found port in docker-compose.yml: {self.default_port}")


def create_probe_script(analysis: Dict) -> str:
    """Generate a probe script based on project analysis"""

    project = analysis["project"]
    integration_type = analysis["integration_type"]
    api_type = analysis.get("api_type", "unknown")
    default_port = analysis.get("default_port", 8080)
    health_endpoint = analysis.get("health_endpoint", "/health")
    notes = analysis.get("notes", [])

    # Build the probe script
    script_lines = [
        "#!/bin/bash",
        f"# Probe for {project}",
        f"# Integration Type: {integration_type.upper()}",
        f"# API Type: {api_type.upper()}",
        "# Law of Runtime Truth: This probe must successfully execute before implementing the adapter",
        "",
        f"CONTAINER_NAME=\"{project}-core\"",
    ]

    if integration_type == "api":
        # HTTP API probe
        endpoint = f"http://localhost:{default_port}{health_endpoint}"
        script_lines.extend([
            f'API_ENDPOINT="{endpoint}"',
            "",
            f"echo \"Probing {project} ({integration_type} - {api_type})...\"",
            "",
            "# Try to curl the API endpoint",
            "if curl -f -s \"${API_ENDPOINT}\" > /dev/null 2>&1; then",
            f"    echo \"[OK] {project} API is accessible\"",
            "    exit 0",
            "else",
            f"    echo \"[FAIL] {project} API is NOT accessible\"",
            f'    echo "  Expected endpoint: ${API_ENDPOINT}"',
            "    echo \"  Please verify:\"",
            f"    echo \"  1. Container is running: docker ps | grep {CONTAINER_NAME}\"",
            f"    echo \"  2. Port is correct: Check core-projects/{project}/README.md\"",
            f"    echo \"  3. Update API_ENDPOINT in this script if different\"",
            "    exit 1",
            "fi"
        ])

    elif integration_type == "library":
        # Library probe - check if importable
        if analysis["language"] == "python":
            script_lines.extend([
                "",
                f"echo \"Probing {project} ({integration_type} - {analysis['language']})...\"",
                "",
                "# Libraries don't have HTTP APIs - check if package is importable",
                f"python3 -c \"import {project.replace('-', '_')}\" 2>/dev/null",
                "if [ $? -eq 0 ]; then",
                f"    echo \"✓ {project} library is importable\"",
                "    exit 0",
                "else",
                f"    echo \"✗ {project} library is NOT importable\"",
                "    echo \"  This is a LIBRARY project - integration requires:\"",
                "    echo \"  1. Install as dependency in adapter\"",
                "    echo \"  2. Import and use directly in Python code\"",
                "    echo \"  3. No HTTP API expected\"",
                "    exit 1",
                "fi"
            ])
        else:
            script_lines.extend([
                "",
                f"echo \"Probing {project} ({integration_type})...\"",
                f"echo \"⚠ {project} is a library - no HTTP API expected\"",
                f"echo \"  Integration strategy: Use as dependency in adapter\"",
                "exit 0"
            ])

    elif integration_type == "cli":
        # CLI probe
        script_lines.extend([
            "",
            f"echo \"Probing {project} ({integration_type})...\"",
            "",
            "# Try to run CLI command",
            f"if command -v {project} &> /dev/null; then",
            f"    {project} --version",
            f"    echo \"✓ {project} CLI is available\"",
            "    exit 0",
            "else",
            f"    echo \"✗ {project} CLI is NOT available\"",
            "    echo \"  This is a CLI project - integration requires:\"",
            "    echo \"  1. Install CLI tool in container\"",
            "    echo \"  2. Execute via subprocess in adapter\"",
            "    exit 1",
            "fi"
        ])

    else:  # hybrid or unknown
        script_lines.extend([
            "",
            f"echo \"Probing {project} ({integration_type})...\"",
            f"echo \"⚠ {project} integration type needs manual verification\"",
            "echo \"  Please check core-projects/{project}/README.md for integration details\"",
            "exit 0"
        ])

    # Add notes section
    if notes:
        script_lines.extend([
            "",
            "# Discovery Notes:",
        ])
        for note in notes:
            script_lines.append(f"#   - {note}")

    return "\n".join(script_lines) + "\n"


def main():
    """Main execution"""

    print("=" * 80)
    print("INTELLIGENT PROBE GENERATOR FOR CORE PROJECTS")
    print("=" * 80)
    print()

    results = []
    created_count = 0
    skipped_count = 0

    for project_name in MISSING_PROJECTS:
        print(f"Analyzing {project_name}...")

        # Analyze project
        analyzer = ProjectAnalyzer(project_name)
        analysis = analyzer.analyze()

        if analysis.get("status") == "error":
            print(f"  ✗ {analysis['message']}")
            skipped_count += 1
            continue

        # Create adapter directory structure
        adapter_dir = ADAPTERS_DIR / f"{project_name}-adapter"
        probes_dir = adapter_dir / "probes"
        probes_dir.mkdir(parents=True, exist_ok=True)

        # Generate probe script
        probe_script = create_probe_script(analysis)
        probe_path = probes_dir / "check_api.sh"

        # Write probe script
        probe_path.write_text(probe_script, encoding="utf-8")

        # Make executable
        os.chmod(probe_path, 0o755)

        print(f"  ✓ Created: {probe_path}")
        print(f"    Type: {analysis['integration_type'].upper()}")
        print(f"    Language: {analysis['language']}")

        results.append(analysis)
        created_count += 1

    # Generate summary report
    print()
    print("=" * 80)
    print("PROBE GENERATION SUMMARY")
    print("=" * 80)
    print(f"Total projects analyzed: {len(MISSING_PROJECTS)}")
    print(f"Probes created: {created_count}")
    print(f"Projects skipped: {skipped_count}")
    print()

    # Count by integration type
    type_counts = {}
    for result in results:
        itype = result["integration_type"]
        type_counts[itype] = type_counts.get(itype, 0) + 1

    print("Distribution by Integration Type:")
    for itype, count in sorted(type_counts.items()):
        print(f"  {itype.upper()}: {count}")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review generated probes in: glue/adapters/*/probes/check_api.sh")
    print("2. For each project:")
    print("   a. Read core-projects/{project}/README.md")
    print("   b. Update API_ENDPOINT or integration strategy in probe script")
    print("   c. Run the probe: ./glue/adapters/{project}-adapter/probes/check_api.sh")
    print("   d. If probe passes, proceed with adapter implementation")
    print("   e. If probe fails, investigate and update probe or mark as non-API project")
    print()

    # Generate detailed report
    report_path = FRONTEND_DIR / "scripts" / "probe_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Detailed analysis saved to: {report_path}")

    return results


if __name__ == "__main__":
    main()
