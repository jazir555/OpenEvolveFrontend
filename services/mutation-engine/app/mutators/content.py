from __future__ import annotations

import random
import re
from typing import List, Tuple

CTA_VARIATIONS = [
    'Get started',
    'Start free trial',
    'Book a demo',
    'Join now',
    'Request access',
    'See pricing',
    'Talk to sales',
    'Launch project',
]

HEADLINE_VARIATIONS = [
    'Build faster with evolutionary design',
    'Design systems that evolve with your users',
    'Turn design experiments into measurable wins',
    'Evolve landing pages in hours, not weeks',
]


class ContentMutator:
    def mutate(self, html: str) -> Tuple[str, List[str]]:
        changes: List[str] = []
        output = html

        cta = random.choice(CTA_VARIATIONS)
        output, count = re.subn(
            r">(Get started|Start free trial|Book a demo|Join now)<",
            f">{cta}<",
            output,
            flags=re.IGNORECASE,
        )
        if count:
            changes.append(f"Updated CTA text to '{cta}'")

        headline = random.choice(HEADLINE_VARIATIONS)
        output, count = re.subn(
            r"<h1[^>]*>.*?</h1>",
            f"<h1>{headline}</h1>",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if count:
            changes.append("Updated primary headline")

        output, count = re.subn(
            r"<p([^>]*)>([^<]+)</p>",
            r"<p\\1><strong>\\2</strong></p>",
            output,
            count=1,
            flags=re.IGNORECASE,
        )
        if count:
            changes.append('Boosted content hierarchy in first paragraph')

        if 'Trusted by' not in output:
            output = re.sub(
                r"</body>",
                "<div class=\"trust-signal\">Trusted by teams at Atlas, Horizon, and Northwind</div></body>",
                output,
                flags=re.IGNORECASE,
            )
            changes.append('Added trust signal block')

        return output, changes
