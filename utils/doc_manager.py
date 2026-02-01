"""
Docstring evolution utilities for refinement workflows.

Ensures refined code includes updated docstrings and computes a simple
documentation fidelity score based on docstring coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re


@dataclass
class DocstringResult:
    updated_code: str
    fidelity_score: float
    changed: bool


class DocstringManager:
    """Manage docstring refinement for generated code."""

    def ensure_docstring_refinement(
        self,
        text: str,
        user_prompt_id: Optional[str] = None
    ) -> DocstringResult:
        """
        Ensure docstrings exist for functions in the provided text.

        Args:
            text: Code or mixed content to scan.
            user_prompt_id: Optional prompt identifier to include.

        Returns:
            DocstringResult with updated content and fidelity score.
        """
        if not text or "def " not in text:
            return DocstringResult(updated_code=text, fidelity_score=1.0, changed=False)

        updated, coverage, changed = self._process_text(text, user_prompt_id)
        fidelity = min(1.0, max(0.0, coverage))
        return DocstringResult(updated_code=updated, fidelity_score=fidelity, changed=changed)

    def _process_text(self, text: str, user_prompt_id: Optional[str]) -> Tuple[str, float, bool]:
        if "```" not in text:
            updated_code, coverage, changed = self._insert_docstrings(text, user_prompt_id)
            return updated_code, coverage, changed

        blocks = []
        last_idx = 0
        changed_any = False
        total_defs = 0
        documented_defs = 0

        pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
        for match in pattern.finditer(text):
            lang = (match.group(1) or "").lower()
            code = match.group(2)
            blocks.append(text[last_idx:match.start()])
            if lang in ("", "python", "py"):
                updated_code, coverage, changed = self._insert_docstrings(code, user_prompt_id)
                blocks.append(f"```{lang}\n{updated_code}```")
                changed_any = changed_any or changed
                defs, documented = self._count_docstrings(updated_code)
                total_defs += defs
                documented_defs += documented
            else:
                blocks.append(match.group(0))
            last_idx = match.end()

        blocks.append(text[last_idx:])
        coverage = 1.0 if total_defs == 0 else documented_defs / total_defs
        return "".join(blocks), coverage, changed_any

    def _insert_docstrings(self, code: str, user_prompt_id: Optional[str]) -> Tuple[str, float, bool]:
        lines = code.splitlines()
        changed = False
        total_defs = 0
        documented_defs = 0

        def _has_docstring(line_idx: int) -> bool:
            for idx in range(line_idx + 1, min(line_idx + 4, len(lines))):
                stripped = lines[idx].strip()
                if not stripped:
                    continue
                return stripped.startswith('"""') or stripped.startswith("'''")
            return False

        def _insert_at(line_idx: int, name: str) -> None:
            indent_match = re.match(r"(\s*)def\s+\w+", lines[line_idx])
            indent = indent_match.group(1) if indent_match else ""
            doc_indent = indent + "    "
            prompt_tag = f"User_Prompt_ID: {user_prompt_id}" if user_prompt_id else "User_Prompt_ID: unknown"
            doc_lines = [
                f'{doc_indent}"""',
                f"{doc_indent}TODO: Update docstring to reflect refined behavior.",
                f"{doc_indent}{prompt_tag}",
                f'{doc_indent}"""'
            ]
            insert_at = line_idx + 1
            lines[insert_at:insert_at] = doc_lines

        for idx, line in enumerate(list(lines)):
            if re.match(r"\s*def\s+\w+\s*\(", line):
                total_defs += 1
                if _has_docstring(idx):
                    documented_defs += 1
                else:
                    func_match = re.match(r"\s*def\s+(\w+)\s*\(", line)
                    func_name = func_match.group(1) if func_match else "function"
                    _insert_at(idx, func_name)
                    documented_defs += 1
                    changed = True

        coverage = 1.0 if total_defs == 0 else documented_defs / total_defs
        return "\n".join(lines), coverage, changed

    def _count_docstrings(self, code: str) -> Tuple[int, int]:
        lines = code.splitlines()
        total_defs = 0
        documented_defs = 0

        for idx, line in enumerate(lines):
            if re.match(r"\s*def\s+\w+\s*\(", line):
                total_defs += 1
                for look_ahead in range(idx + 1, min(idx + 4, len(lines))):
                    stripped = lines[look_ahead].strip()
                    if not stripped:
                        continue
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        documented_defs += 1
                    break

        return total_defs, documented_defs
