"""
Auto-tracking helpers to prevent agents from re-notifying themselves.

Stores per-agent seen node lists in a lightweight CSV file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Set


def get_agent_state_file(agent_name: str) -> Path:
    """Return the per-agent state file path."""
    state_dir = Path(__file__).parent / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    sanitized = agent_name.replace(" ", "_")
    return state_dir / f"seen_nodes_{sanitized}.csv"


def _load_seen_nodes(state_file: Path) -> Set[str]:
    if not state_file.exists():
        return set()
    try:
        with state_file.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            return {row[0] for row in reader if row}
    except (OSError, UnicodeDecodeError):
        return set()


def _save_seen_nodes(state_file: Path, nodes: Iterable[str]) -> None:
    try:
        with state_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for node in sorted(set(nodes)):
                writer.writerow([node])
    except OSError:
        return


def _list_markdown_files(vault_dir: Path) -> List[str]:
    return [
        str(path.relative_to(vault_dir))
        for path in vault_dir.rglob("*.md")
        if path.is_file()
    ]


def get_new_nodes(vault_dir: str, agent_name: str, save_state: bool = True) -> List[str]:
    """Return markdown files in vault that the agent has not yet seen."""
    vault_path = Path(vault_dir)
    if not vault_path.exists():
        return []
    state_file = get_agent_state_file(agent_name)
    seen_nodes = _load_seen_nodes(state_file)
    all_nodes = _list_markdown_files(vault_path)
    new_nodes = [node for node in all_nodes if node not in seen_nodes]
    if save_state:
        _save_seen_nodes(state_file, seen_nodes.union(all_nodes))
    return new_nodes


def mark_file_as_seen_by_agent(vault_dir: str, file_path: str, agent_name: str) -> None:
    """Record a file as seen by a specific agent."""
    vault_path = Path(vault_dir)
    try:
        relative_path = str(Path(file_path).relative_to(vault_path))
    except ValueError:
        relative_path = Path(file_path).name

    state_file = get_agent_state_file(agent_name)
    seen_nodes = _load_seen_nodes(state_file)
    seen_nodes.add(relative_path)
    _save_seen_nodes(state_file, seen_nodes)
