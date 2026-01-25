from __future__ import annotations

import random
import re
from typing import List, Tuple

GRID_COLUMNS = [12, 16, 24]
FLEX_DIRECTIONS = ['row', 'column', 'row-reverse']
CONTAINER_WIDTHS = ['960px', '1080px', '1200px', '1280px']
SPACING_SCALE = [0.8, 1.0, 1.2]
POSITION_VALUES = ['flex-start', 'center', 'flex-end', 'space-between']


class LayoutMutator:
    def mutate(self, html: str, css: str | None) -> Tuple[str, str | None, List[str]]:
        grid = random.choice(GRID_COLUMNS)
        flex_dir = random.choice(FLEX_DIRECTIONS)
        container = random.choice(CONTAINER_WIDTHS)
        spacing = random.choice(SPACING_SCALE)
        position = random.choice(POSITION_VALUES)
        changes = [
            f"Updated grid to {grid} columns",
            f"Updated flex direction to {flex_dir}",
            f"Updated container width to {container}",
            f"Adjusted spacing scale to {spacing}",
            f"Adjusted component alignment to {position}",
        ]

        html_out = self._replace_grid_columns(html, grid)
        css_out = self._replace_grid_columns(css, grid) if css else None

        html_out = self._replace_css_value(html_out, 'flex-direction', flex_dir)
        css_out = self._replace_css_value(css_out, 'flex-direction', flex_dir) if css_out else css_out

        html_out = self._replace_css_value(html_out, 'max-width', container)
        css_out = self._replace_css_value(css_out, 'max-width', container) if css_out else css_out

        html_out = self._scale_spacing(html_out, spacing)
        css_out = self._scale_spacing(css_out, spacing) if css_out else css_out

        html_out = self._replace_css_value(html_out, 'justify-content', position)
        css_out = self._replace_css_value(css_out, 'justify-content', position) if css_out else css_out

        html_out = self._replace_css_value(html_out, 'align-items', position)
        css_out = self._replace_css_value(css_out, 'align-items', position) if css_out else css_out

        return html_out, css_out, changes

    def _replace_grid_columns(self, content: str | None, columns: int) -> str:
        if content is None:
            return ''
        return re.sub(
            r"grid-template-columns\s*:\s*repeat\(\d+,",
            f"grid-template-columns: repeat({columns},",
            content,
            flags=re.IGNORECASE,
        )

    def _replace_css_value(self, content: str | None, prop: str, value: str) -> str:
        if content is None:
            return ''
        return re.sub(
            rf"({re.escape(prop)}\s*:\s*)([^;]+)",
            rf"\1{value}",
            content,
            flags=re.IGNORECASE,
        )

    def _scale_spacing(self, content: str | None, scale: float) -> str:
        if content is None:
            return ''

        return re.sub(r"(margin|padding)\s*:\s*(\d+(?:\.\d+)?)(px|rem|em)", lambda m: f"{m.group(1)}: {float(m.group(2)) * scale:.2f}{m.group(3)}", content, flags=re.IGNORECASE)
