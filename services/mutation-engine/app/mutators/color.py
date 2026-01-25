from __future__ import annotations

import random
import re
from typing import Dict, List, Tuple


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return '#%02x%02x%02x' % rgb


def _adjust_color(hex_color: str, factor: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    r = min(255, max(0, int(r * factor)))
    g = min(255, max(0, int(g * factor)))
    b = min(255, max(0, int(b * factor)))
    return _rgb_to_hex((r, g, b))


def _complementary_color(hex_color: str) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex((255 - r, 255 - g, 255 - b))


def _analogous_colors(hex_color: str) -> Tuple[str, str]:
    r, g, b = _hex_to_rgb(hex_color)
    return (
        _rgb_to_hex((min(255, r + 20), max(0, g - 10), b)),
        _rgb_to_hex((max(0, r - 20), min(255, g + 10), b)),
    )


BASE_PALETTES: List[Dict[str, str]] = [
    {
        '--primary': '#1f6feb',
        '--secondary': '#0d1117',
        '--accent': '#f78166',
        '--background': '#f6f8fa',
        '--foreground': '#0b1320',
    },
    {
        '--primary': '#2563eb',
        '--secondary': '#111827',
        '--accent': '#f97316',
        '--background': '#f8fafc',
        '--foreground': '#0f172a',
    },
    {
        '--primary': '#0f766e',
        '--secondary': '#134e4a',
        '--accent': '#facc15',
        '--background': '#f0fdfa',
        '--foreground': '#042f2e',
    },
    {
        '--primary': '#db2777',
        '--secondary': '#1f2937',
        '--accent': '#fde68a',
        '--background': '#fdf2f8',
        '--foreground': '#2b0a1f',
    },
    {
        '--primary': '#7c3aed',
        '--secondary': '#111827',
        '--accent': '#38bdf8',
        '--background': '#f5f3ff',
        '--foreground': '#1f1147',
    },
    {
        '--primary': '#059669',
        '--secondary': '#064e3b',
        '--accent': '#f59e0b',
        '--background': '#ecfdf5',
        '--foreground': '#052e24',
    },
    {
        '--primary': '#dc2626',
        '--secondary': '#1f2937',
        '--accent': '#fbbf24',
        '--background': '#fef2f2',
        '--foreground': '#320c0c',
    },
    {
        '--primary': '#0ea5e9',
        '--secondary': '#0f172a',
        '--accent': '#22c55e',
        '--background': '#f0f9ff',
        '--foreground': '#0b1f2d',
    },
    {
        '--primary': '#4f46e5',
        '--secondary': '#1e1b4b',
        '--accent': '#ec4899',
        '--background': '#eef2ff',
        '--foreground': '#0e0c2c',
    },
    {
        '--primary': '#ea580c',
        '--secondary': '#431407',
        '--accent': '#14b8a6',
        '--background': '#fff7ed',
        '--foreground': '#321207',
    },
]


def _generate_palettes() -> List[Dict[str, str]]:
    palettes: List[Dict[str, str]] = []
    for palette in BASE_PALETTES:
        palettes.append(palette)
        palettes.append(
            {key: _adjust_color(value, 1.1) for key, value in palette.items()}
        )
        palettes.append(
            {key: _adjust_color(value, 0.9) for key, value in palette.items()}
        )
        palettes.append(
            {
                **palette,
                '--accent': _adjust_color(palette['--accent'], 1.2),
                '--background': _adjust_color(palette['--background'], 0.95),
            }
        )
        palettes.append(
            {
                **palette,
                '--primary': _adjust_color(palette['--primary'], 0.8),
                '--foreground': _adjust_color(palette['--foreground'], 1.1),
            }
        )
    return palettes


PALETTES = _generate_palettes()


class ColorMutator:
    def __init__(self) -> None:
        self.palettes = PALETTES

    def mutate(self, html: str, css: str | None, constraints: dict | None) -> Tuple[str, str | None, List[str]]:
        palette = random.choice(self.palettes)
        changes: List[str] = [f"Applied palette {palette['--primary']} / {palette['--accent']}"]

        if constraints and isinstance(constraints.get('brand_colors'), dict):
            palette.update(constraints['brand_colors'])
            changes.append('Applied brand color constraints')
        else:
            if random.random() < 0.5:
                palette['--accent'] = _complementary_color(palette['--primary'])
                changes.append('Applied complementary accent color')
            if random.random() < 0.5:
                analog_one, analog_two = _analogous_colors(palette['--primary'])
                palette['--secondary'] = analog_one
                palette['--foreground'] = analog_two
                changes.append('Applied analogous secondary colors')

        html_out = self._apply_palette(html, palette)
        css_out = self._apply_palette(css, palette) if css else None
        return html_out, css_out, changes

    def _apply_palette(self, content: str | None, palette: Dict[str, str]) -> str:
        if content is None:
            return ''

        output = content
        for var_name, value in palette.items():
            output = re.sub(
                rf"({re.escape(var_name)}\s*:\s*)([^;]+);",
                rf"\1{value};",
                output,
                flags=re.IGNORECASE,
            )
        return output
