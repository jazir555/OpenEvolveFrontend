from __future__ import annotations

import random
import re
from typing import List, Tuple

BUTTON_VARIANTS = [
    'btn-primary',
    'btn-secondary',
    'btn-outline',
    'btn-solid',
]
NAV_POSITIONS = ['nav-left', 'nav-center', 'nav-right']
HERO_LAYOUTS = ['hero-split', 'hero-stacked', 'hero-overlay']


class ComponentMutator:
    def mutate(self, html: str) -> Tuple[str, List[str]]:
        changes: List[str] = []
        output = html

        variant = random.choice(BUTTON_VARIANTS)
        output, count = re.subn(
            r'class="([^"]*btn-[^\s\"]+)"',
            lambda m: f'class="{self._swap_button_class(m.group(1), variant)}"',
            output,
            flags=re.IGNORECASE,
        )
        if count:
            changes.append(f"Updated button style to {variant}")

        output, swapped = self._swap_sections(output)
        if swapped:
            changes.append('Reordered sections')

        nav_variant = random.choice(NAV_POSITIONS)
        output, count = re.subn(
            r'<nav([^>]*)class="([^"]*)"',
            lambda m: f'<nav{m.group(1)}class="{self._inject_class(m.group(2), nav_variant)}"',
            output,
            flags=re.IGNORECASE,
        )
        if count:
            changes.append(f"Updated navigation position to {nav_variant}")

        hero_variant = random.choice(HERO_LAYOUTS)
        output, count = re.subn(
            r'class="([^"]*hero[^"]*)"',
            lambda m: f'class="{self._inject_class(m.group(1), hero_variant)}"',
            output,
            flags=re.IGNORECASE,
        )
        if count:
            changes.append(f"Updated hero layout to {hero_variant}")

        return output, changes

    def _swap_button_class(self, class_name: str, replacement: str) -> str:
        classes = class_name.split()
        updated = [replacement if cls.startswith('btn-') else cls for cls in classes]
        return ' '.join(updated)

    def _inject_class(self, class_name: str, extra: str) -> str:
        classes = class_name.split()
        if extra not in classes:
            classes.append(extra)
        return ' '.join(classes)

    def _swap_sections(self, html: str) -> Tuple[str, bool]:
        sections = re.findall(r'<section[\s\S]*?</section>', html, flags=re.IGNORECASE)
        if len(sections) < 2:
            return html, False

        first, second = sections[0], sections[1]
        swapped_html = html.replace(first, '__SECTION_ONE__', 1)
        swapped_html = swapped_html.replace(second, first, 1)
        swapped_html = swapped_html.replace('__SECTION_ONE__', second, 1)
        return swapped_html, True
