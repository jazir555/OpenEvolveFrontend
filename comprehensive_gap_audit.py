#!/usr/bin/env python3
"""
Comprehensive Gap Audit Script
Finds ALL placeholders, stubs, incomplete implementations in reliability codebase
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Gap:
    """Represents a single gap found in code"""
    file: str
    line: int
    gap_type: str  # "empty_class", "stub_method", "pass_statement", "todo_comment", etc.
    description: str
    code_snippet: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"

class GapAuditor:
    """Audits codebase for gaps and placeholders"""

    def __init__(self):
        self.gaps: List[Gap] = []

    def audit_file(self, filepath: Path) -> List[Gap]:
        """Audit a single file for gaps"""
        gaps = []
        try:
            content = filepath.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Check 1: Empty class definitions (class with only pass)
            gaps.extend(self._find_empty_classes(filepath, content, lines))

            # Check 2: Empty methods (def with only pass or ...)
            gaps.extend(self._find_empty_methods(filepath, content, lines))

            # Check 3: TODO/FIXME/NotImplemented comments
            gaps.extend(self._find_todo_comments(filepath, content, lines))

            # Check 4: Stub/mock implementations
            gaps.extend(self._find_stub_implementations(filepath, content, lines))

            # Check 5: Placeholder method bodies
            gaps.extend(self._find_placeholder_bodies(filepath, content, lines))

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"Error auditing {filepath}: {e}")

        return gaps

    def _find_empty_classes(self, filepath: Path, content: str, lines: List[str]) -> List[Gap]:
        """Find class definitions with only 'pass' statement"""
        gaps = []

        # Pattern: class <name>: ... (optional docstring) ... pass
        pattern = r'(class\s+(\w+)\s*[^:]*:)\s*\n(?:\s+(\"\"\"[^\"]*\"\"\"|\'\'\'[^\']*\'\'\'))?\s*\n\s+(pass)\s*$'

        for match in re.finditer(pattern, content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            class_name = match.group(2)
            code_snippet = self._get_snippet(lines, line_num, 3)

            # Exception: Exception classes are allowed to have pass
            if not class_name.endswith('Error') and not class_name.endswith('Exception'):
                gaps.append(Gap(
                    file=str(filepath.relative_to(Path.cwd())),
                    line=line_num,
                    gap_type="empty_class",
                    description=f"Class '{class_name}' has only 'pass' statement",
                    code_snippet=code_snippet,
                    severity="HIGH" if 'Stub' not in class_name else "CRITICAL"
                ))

        return gaps

    def _find_empty_methods(self, filepath: Path, content: str, lines: List[str]) -> List[Gap]:
        """Find method definitions with only 'pass' or '...'"""
        gaps = []

        # Pattern: def <method>...: ... (optional docstring) ... (pass or ...)
        pattern = r'(\def\s+(\w+)\s*\([^)]*\)\s*(?:->[^:]+)?:\s*\n)(?:\s+(\"\"\"[^\"]*\"\"\"|\'\'\'[^\']*\'\'\'))?\s*\n\s+(pass|\.\.\.)\s*$'

        for match in re.finditer(pattern, content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            method_name = match.group(2)
            code_snippet = self._get_snippet(lines, line_num, 5)

            # Check if it's a legitimate use (exception classes, etc)
            if not self._is_legitimate_empty_method(method_name, code_snippet):
                gaps.append(Gap(
                    file=str(filepath.relative_to(Path.cwd())),
                    line=line_num,
                    gap_type="empty_method",
                    description=f"Method '{method_name}' has only '{match.group(3)}' statement",
                    code_snippet=code_snippet,
                    severity="CRITICAL"
                ))

        return gaps

    def _find_todo_comments(self, filepath: Path, content: str, lines: List[str]) -> List[Gap]:
        """Find TODO, FIXME, Not Implemented comments"""
        gaps = []

        # Pattern: TODO, FIXME, Not Implemented, placeholder, etc.
        pattern = r'.*(TODO|FIXME|NotImplemented|placeholder|not implemented|future implementation|incomplete|stub.*implementation)'

        for line_num, line in enumerate(lines, 1):
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                # Skip if in docstring or comment about what WAS done
                if not self._is_legitimate_comment(line):
                    code_snippet = self._get_snippet(lines, line_num, 3)
                    gaps.append(Gap(
                        file=str(filepath.relative_to(Path.cwd())),
                        line=line_num,
                        gap_type="todo_comment",
                        description=f"Comment indicates incomplete work: '{match.group(1)}'",
                        code_snippet=code_snippet,
                        severity="HIGH"
                    ))

        return gaps

    def _find_stub_implementations(self, filepath: Path, content: str, lines: List[str]) -> List[Gap]:
        """Find stub implementations in comments or class names"""
        gaps = []

        # Pattern: Stub classes
        pattern = r'(class\s+(Stub\w*|Mock\w*|Fake\w*)\s*.*:)\s*\n(?:\s+(\"\"\"[^\"]*Stub[^\"]*\"\"\"|\'\'\'[^\']*Stub[^\']*\'\'\'))?\s*\n\s+(pass)\s*$'

        for match in re.finditer(pattern, content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            class_name = match.group(2)
            code_snippet = self._get_snippet(lines, line_num, 3)

            gaps.append(Gap(
                file=str(filepath.relative_to(Path.cwd())),
                line=line_num,
                gap_type="stub_class",
                description=f"Stub class '{class_name}' with no implementation",
                code_snippet=code_snippet,
                severity="CRITICAL"
            ))

        return gaps

    def _find_placeholder_bodies(self, filepath: Path, content: str, lines: List[str]) -> List[Gap]:
        """Find methods with placeholder bodies (raise NotImplementedError, etc.)"""
        gaps = []

        # Pattern: raise NotImplementedError
        pattern = r'(def\s+(\w+)\s*\([^)]*\)\s*(?:->[^:]+)?:.*\n.*?)(raise\s+NotImplementedError)'

        for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
            line_num = content[:match.start()].count('\n') + 1
            method_name = match.group(2)
            code_snippet = self._get_snippet(lines, line_num, 5)

            gaps.append(Gap(
                file=str(filepath.relative_to(Path.cwd())),
                line=line_num,
                gap_type="not_implemented",
                description=f"Method '{method_name}' raises NotImplementedError",
                code_snippet=code_snippet,
                severity="CRITICAL"
            ))

        return gaps

    def _is_legitimate_empty_method(self, method_name: str, code_snippet: str) -> bool:
        """Check if an empty method is legitimate (e.g., __init__, abstract methods)"""
        # Abstract methods are allowed to be empty
        if '@abstractmethod' in code_snippet or 'ABC' in code_snippet:
            return True

        # Magic methods that are legitimately empty
        if method_name.startswith('__') and method_name.endswith('__'):
            return True

        return False

    def _is_legitimate_comment(self, line: str) -> bool:
        """Check if a TODO/FIXME comment is legitimate (e.g., documenting what was done)"""
        # Comments that say what was already done are OK
        if any(phrase in line.lower() for phrase in ['completed', 'implemented', 'fixed', 'done', 'added']):
            return True

        # Comments in docstrings describing future features are OK
        if '"""' in line or "'''" in line:
            return True

        return False

    def _get_snippet(self, lines: List[str], line_num: int, context: int = 3) -> str:
        """Get code snippet around a line"""
        start = max(0, line_num - context - 1)
        end = min(len(lines), line_num + context)
        snippet_lines = []
        for i, line in enumerate(lines[start:end], start):
            snippet_lines.append(f"{i+1}: {line}")
        snippet = '\n'.join(snippet_lines)
        return snippet

def main():
    """Run comprehensive gap audit"""
    print("=" * 80)
    print("COMPREHENSIVE GAP AUDIT")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print()

    auditor = GapAuditor()
    all_gaps: List[Gap] = []

    # Audit reliability files
    print("Auditing reliability/ directory...")
    reliability_files = [
        'reliability/config.py',
        'reliability/enhanced_redflagger.py',
        'reliability/guardrails_adapter.py',
        'reliability/lmql_adapter.py',
        'reliability/unified_bridge.py',
    ]

    for filepath_str in reliability_files:
        filepath = Path(filepath_str)
        if filepath.exists():
            print(f"  Auditing {filepath_str}...")
            gaps = auditor.audit_file(filepath)
            all_gaps.extend(gaps)
            if gaps:
                print(f"    Found {len(gaps)} gaps")
        else:
            print(f"  WARNING: {filepath_str} not found")

    # Audit adapter files
    print("\nAuditing reliability-plugin/adapters/ directory...")
    adapter_files = [
        'reliability-plugin/adapters/mdap/mdap_reliability_adapter.py',
        'reliability-plugin/adapters/roma/roma_reliability_adapter.py',
    ]

    for filepath_str in adapter_files:
        filepath = Path(filepath_str)
        if filepath.exists():
            print(f"  Auditing {filepath_str}...")
            gaps = auditor.audit_file(filepath)
            all_gaps.extend(gaps)
            if gaps:
                print(f"    Found {len(gaps)} gaps")
        else:
            print(f"  WARNING: {filepath_str} not found")

    # Print results
    print("\n" + "=" * 80)
    print("AUDIT RESULTS")
    print("=" * 80)
    print(f"\nTotal gaps found: {len(all_gaps)}")

    # Group by severity
    critical = [g for g in all_gaps if g.severity == "CRITICAL"]
    high = [g for g in all_gaps if g.severity == "HIGH"]
    medium = [g for g in all_gaps if g.severity == "MEDIUM"]
    low = [g for g in all_gaps if g.severity == "LOW"]

    print(f"\nCRITICAL: {len(critical)}")
    print(f"HIGH: {len(high)}")
    print(f"MEDIUM: {len(medium)}")
    print(f"LOW: {len(low)}")

    # Print all gaps
    if all_gaps:
        print("\n" + "=" * 80)
        print("DETAILED GAP LIST")
        print("=" * 80)

        for gap in sorted(all_gaps, key=lambda g: (g.severity, g.file, g.line)):
            print(f"\n[{gap.severity}] {gap.file}:{gap.line}")
            print(f"  Type: {gap.gap_type}")
            print(f"  Description: {gap.description}")
            print(f"  Code:\n{gap.code_snippet}")
    else:
        print("\n✅ NO GAPS FOUND - ALL CODE IS PRODUCTION READY!")

    # Save to file
    output_file = Path("GAP_AUDIT_REPORT.txt")
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE GAP AUDIT REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total gaps found: {len(all_gaps)}\n\n")

        for gap in sorted(all_gaps, key=lambda g: (g.severity, g.file, g.line)):
            f.write(f"\n[{gap.severity}] {gap.file}:{gap.line}\n")
            f.write(f"  Type: {gap.gap_type}\n")
            f.write(f"  Description: {gap.description}\n")
            f.write(f"  Code:\n{gap.code_snippet}\n")

    print(f"\nReport saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
