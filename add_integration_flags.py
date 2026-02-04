#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add availability flags to integration files that don't have them.
"""

import io
import sys
import re
from pathlib import Path

# Setup encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def add_availability_flag(file_path: str, flag_name: str, import_statement: str):
    """
    Add availability flag to the end of a Python file.

    Args:
        file_path: Path to the Python file
        flag_name: Name of the availability flag (e.g., 'DSPY_INTEGRATION_AVAILABLE')
        import_statement: Import statement to check (e.g., 'import dspy')
    """
    path = Path(file_path)

    if not path.exists():
        print(f"  [X] File not found: {file_path}")
        return False

    # Read the file
    content = path.read_text(encoding='utf-8')

    # Check if flag already exists
    if flag_name in content:
        print(f"  [->] Flag {flag_name} already exists in {file_path}")
        return True

    # Add the availability flag at the end
    flag_block = f"""

# Availability flag
try:
    {import_statement}
    {flag_name} = True
except ImportError:
    {flag_name} = False
"""

    content += flag_block

    # Write back
    path.write_text(content, encoding='utf-8')
    print(f"  [OK] Added {flag_name} to {file_path}")
    return True


def main():
    """Add availability flags to key integrations."""
    integrations = [
        ("knowledge_engine/integrations/dspy_integration.py", "DSPY_INTEGRATION_AVAILABLE", "import dspy"),
        ("knowledge_engine/integrations/ragbits_integration.py", "RAGBITS_INTEGRATION_AVAILABLE", "import ragbits"),
        ("knowledge_engine/integrations/agentic_context_integration.py", "ACE_INTEGRATION_AVAILABLE", "from ace import SkillBook"),
        ("knowledge_engine/integrations/agentjson_integration.py", "AGENTJSON_INTEGRATION_AVAILABLE", "import agentjson"),
        ("knowledge_engine/integrations/research_quest_integration.py", "RESEARCH_QUEST_INTEGRATION_AVAILABLE", "import research_quest"),
        ("knowledge_engine/integrations/mcp_gateway_integration.py", "MCP_GATEWAY_INTEGRATION_AVAILABLE", "import mcp_gateway"),
        ("knowledge_engine/integrations/openevolve_integration_library.py", "OPENEVOLVE_INTEGRATION_AVAILABLE", "import openevolve"),
    ]

    print("\nAdding availability flags to integrations:")
    print("="*80)

    for file_path, flag_name, import_statement in integrations:
        add_availability_flag(file_path, flag_name, import_statement)

    print("\n" + "="*80)
    print("Done!")


if __name__ == "__main__":
    main()
