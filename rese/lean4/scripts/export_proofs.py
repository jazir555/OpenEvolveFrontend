#!/usr/bin/env python3
"""
export_proofs.py

Export Lean 4 theorems and proofs to Python for documentation.

Usage:
    python scripts/export_proofs.py

Author: Agent O1 (Lean 4 Formalization Specialist)
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class TheoremInfo:
    """Information about a Lean 4 theorem"""
    name: str
    module: str
    statement: str
    proof_summary: str
    dependencies: List[str]
    verified: bool


class Lean4Exporter:
    """Export Lean 4 theorems to Python"""

    def __init__(self, lean_dir: Path):
        self.lean_dir = lean_dir
        self.theorems: List[TheoremInfo] = []

    def parse_lean_file(self, filepath: Path) -> List[TheoremInfo]:
        """
        Parse a Lean 4 file and extract theorems.

        Args:
            filepath: Path to Lean 4 file

        Returns:
            List of theorem information
        """
        theorems = []
        content = filepath.read_text()

        # Find theorem definitions
        # Pattern: theorem <name> : <type> := by <proof>
        theorem_pattern = r'theorem\s+(\w+)\s*:\s*(.+?)\s*:=\s*by\s*\n((?:\s+.+\n)*)'

        matches = re.finditer(theorem_pattern, content, re.MULTILINE)

        for match in matches:
            name = match.group(1)
            statement = match.group(2).strip()
            proof = match.group(3).strip()

            # Extract dependencies (simplified)
            dependencies = self._extract_dependencies(content, name)

            # Check if proof is complete (no 'sorry')
            verified = 'sorry' not in proof

            theorem = TheoremInfo(
                name=name,
                module=filepath.stem,
                statement=statement,
                proof_summary=self._summarize_proof(proof),
                dependencies=dependencies,
                verified=verified
            )
            theorems.append(theorem)

        return theorems

    def _extract_dependencies(self, content: str, theorem_name: str) -> List[str]:
        """Extract theorem dependencies (simplified)"""
        # Look for imports
        imports = re.findall(r'import\s+RESE\.(\w+)', content)
        return imports

    def _summarize_proof(self, proof: str) -> str:
        """Summarize proof strategy"""
        if not proof:
            return "No proof"

        # Count tactics used
        tactics = re.findall(r'\s+(by|apply|intro|cases|unfold|simp|rw|exact|sorry)', proof)

        if 'sorry' in tactics:
            return "Incomplete (contains 'sorry')"
        elif len(tactics) <= 3:
            return f"Simple proof ({len(tactics)} tactics)"
        else:
            return f"Complex proof ({len(tactics)} tactics)"

    def export_all(self) -> Dict[str, List[TheoremInfo]]:
        """
        Export all theorems from Lean 4 files.

        Returns:
            Dictionary mapping module names to theorems
        """
        modules = {
            'Basic': 'Basic.lean',
            'Constraint': 'Constraint.lean',
            'Templates': 'Templates.lean',
            'TestCases': 'TestCases.lean',
            'RESE': 'RESE.lean'
        }

        all_theorems = {}

        for module_name, filename in modules.items():
            filepath = self.lean_dir / filename
            if filepath.exists():
                theorems = self.parse_lean_file(filepath)
                all_theorems[module_name] = theorems
                self.theorems.extend(theorems)

        return all_theorems

    def export_to_json(self, output_path: Path) -> None:
        """
        Export theorems to JSON.

        Args:
            output_path: Path to output JSON file
        """
        theorems_dict = [asdict(t) for t in self.theorems]

        output_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_theorems': len(self.theorems),
            'verified_theorems': sum(1 for t in self.theorems if t.verified),
            'theorems': theorems_dict
        }

        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"Exported {len(self.theorems)} theorems to {output_path}")

    def export_to_markdown(self, output_path: Path) -> None:
        """
        Export theorems to Markdown documentation.

        Args:
            output_path: Path to output Markdown file
        """
        lines = [
            "# Lean 4 Theorems Documentation",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Theorems**: {len(self.theorems)}",
            f"**Verified**: {sum(1 for t in self.theorems if t.verified)}",
            "",
            "## Module Summary",
            ""
        ]

        # Group by module
        modules = {}
        for theorem in self.theorems:
            if theorem.module not in modules:
                modules[theorem.module] = []
            modules[theorem.module].append(theorem)

        # Summary table
        lines.extend([
            "| Module | Theorems | Verified |",
            "|--------|----------|----------|"
        ])

        for module_name, module_theorems in sorted(modules.items()):
            verified_count = sum(1 for t in module_theorems if t.verified)
            lines.append(f"| {module_name} | {len(module_theorems)} | {verified_count} |")

        lines.extend(["", "## Theorems", ""])

        # Detailed theorems
        for module_name, module_theorems in sorted(modules.items()):
            lines.extend([
                f"### {module_name}",
                ""
            ])

            for theorem in module_theorems:
                status = "✅ Verified" if theorem.verified else "⚠️ Incomplete"
                lines.extend([
                    f"#### {theorem.name} {status}",
                    "",
                    f"**Statement**: `{theorem.statement}`",
                    "",
                    f"**Proof**: {theorem.proof_summary}",
                    ""
                ])

                if theorem.dependencies:
                    lines.extend([
                        f"**Dependencies**: {', '.join(theorem.dependencies)}",
                        ""
                    ])

        output_path.write_text("\n".join(lines))
        print(f"Exported documentation to {output_path}")

    def generate_python_stub(self, output_path: Path) -> None:
        """
        Generate Python stub for using Lean 4 theorems.

        Args:
            output_path: Path to output Python file
        """
        lines = [
            '"""',
            'Lean 4 Theorem Stubs',
            '',
            'Auto-generated from Lean 4 formalizations.',
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            '"""',
            '',
            'from typing import Dict, List, Optional',
            'from dataclasses import dataclass',
            '',
            '',
            '@dataclass',
            'class LeanTheorem:',
            '    """Reference to a verified Lean 4 theorem"""',
            '    name: str',
            '    module: str',
            '    verified: bool = False',
            '    ',
            '    def __str__(self) -> str:',
            '        status = "✓" if self.verified else "✗"',
            '        return f"{status} {self.module}.{self.name}"',
            '',
            '',
            '# Verified theorems from Lean 4',
            'THEOREMS: Dict[str, LeanTheorem] = {'
        ]

        for theorem in self.theorems:
            lines.append(f'    "{theorem.name}": LeanTheorem(')
            lines.append(f'        name="{theorem.name}",')
            lines.append(f'        module="{theorem.module}",')
            lines.append(f'        verified={theorem.verified}')
            lines.append(f'    ),')

        lines.extend([
            '}',
            '',
            '',
            'def get_theorem(name: str) -> Optional[LeanTheorem]:',
            '    """Get a theorem by name"""',
            '    return THEOREMS.get(name)',
            '',
            '',
            'def list_verified() -> List[LeanTheorem]:',
            '    """List all verified theorems"""',
            '    return [t for t in THEOREMS.values() if t.verified]',
            '',
            '',
            'def list_module(module: str) -> List[LeanTheorem]:',
            '    """List theorems from a specific module"""',
            '    return [t for t in THEOREMS.values() if t.module == module]',
            '',
            '',
            'if __name__ == "__main__":',
            '    print("Lean 4 Theorems")',
            '    print("=" * 50)',
            '    ',
            '    verified = list_verified()',
            '    print(f"Total theorems: {len(THEOREMS)}")',
            '    print(f"Verified: {len(verified)}")',
            '    ',
            '    for module_name in ["Basic", "Constraint", "Templates", "TestCases", "RESE"]:',
            '        theorems = list_module(module_name)',
            '        if theorems:',
            '            print(f"\\n{module_name}: {len(theorems)} theorems")',
            '            for t in theorems:',
            '                print(f"  {t}")',
        ])

        output_path.write_text("\n".join(lines))
        print(f"Generated Python stub at {output_path}")


def main():
    """Main export process"""
    lean_dir = Path(__file__).parent.parent
    output_dir = lean_dir / "exported"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Create exporter
    exporter = Lean4Exporter(lean_dir)

    # Export all theorems
    print("Exporting Lean 4 theorems...")
    all_theorems = exporter.export_all()

    # Summary
    total = len(exporter.theorems)
    verified = sum(1 for t in exporter.theorems if t.verified)

    print(f"\nSummary:")
    print(f"  Total theorems: {total}")
    print(f"  Verified: {verified}")
    print(f"  Incomplete: {total - verified}")

    # Export to JSON
    json_path = output_dir / "theorems.json"
    exporter.export_to_json(json_path)

    # Export to Markdown
    md_path = output_dir / "theorems.md"
    exporter.export_to_markdown(md_path)

    # Generate Python stub
    py_path = output_dir / "lean_theorems.py"
    exporter.generate_python_stub(py_path)

    print(f"\nExport complete! Files in: {output_dir}")


if __name__ == "__main__":
    main()
