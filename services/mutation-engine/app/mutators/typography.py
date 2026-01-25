from __future__ import annotations

import random
import re
from typing import List, Tuple

TYPE_SCALES = [0.95, 1.0, 1.05, 1.1, 1.15]
FONT_WEIGHTS = [300, 400, 500, 600, 700]
LINE_HEIGHTS = [1.2, 1.4, 1.6, 1.8]
LETTER_SPACING = ["-0.02em", "-0.01em", "0", "0.01em", "0.02em"]


class TypographyMutator:
    def mutate(self, html: str, css: str | None) -> Tuple[str, str | None, List[str]]:
        scale = random.choice(TYPE_SCALES)
        weight = random.choice(FONT_WEIGHTS)
        line_height = random.choice(LINE_HEIGHTS)
        letter_spacing = random.choice(LETTER_SPACING)
        changes = [
            f"Adjusted font scale to {scale}",
            f"Adjusted font weight to {weight}",
            f"Adjusted line height to {line_height}",
            f"Adjusted letter spacing to {letter_spacing}",
        ]

        html_out = self._scale_font_sizes(html, scale)
        css_out = self._scale_font_sizes(css, scale) if css else None

        html_out = self._replace_css_value(html_out, 'font-weight', str(weight))
        css_out = self._replace_css_value(css_out, 'font-weight', str(weight)) if css_out else css_out

        html_out = self._replace_css_value(html_out, 'line-height', str(line_height))
        css_out = self._replace_css_value(css_out, 'line-height', str(line_height)) if css_out else css_out

        html_out = self._replace_css_value(html_out, 'letter-spacing', letter_spacing)
        css_out = self._replace_css_value(css_out, 'letter-spacing', letter_spacing) if css_out else css_out

        return html_out, css_out, changes

    def _scale_font_sizes(self, content: str | None, scale: float) -> str:
        if content is None:
            return ''

        def repl(match: re.Match) -> str:
            value = float(match.group(1))
            unit = match.group(2)
            return f"font-size: {value * scale:.2f}{unit}"

        return re.sub(r"font-size\s*:\s*(\d+(?:\.\d+)?)(px|rem|em)", repl, content, flags=re.IGNORECASE)

    def _replace_css_value(self, content: str | None, prop: str, value: str) -> str:
        if content is None:
            return ''
        return re.sub(
            rf"({re.escape(prop)}\s*:\s*)([^;]+)",
            rf"\1{value}",
            content,
            flags=re.IGNORECASE,
        )
