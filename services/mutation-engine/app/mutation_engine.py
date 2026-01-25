from __future__ import annotations

import asyncio
from typing import List

from .schemas import DesignInput, MutationRequest, MutationResult
from .mutators.color import ColorMutator
from .mutators.typography import TypographyMutator
from .mutators.layout import LayoutMutator
from .mutators.content import ContentMutator
from .mutators.component import ComponentMutator

try:
    from evolutionary_optimization import EvolutionaryOptimizer
except Exception:  # pragma: no cover
    EvolutionaryOptimizer = None


class MutationEngine:
    def __init__(self) -> None:
        self.color_mutator = ColorMutator()
        self.typography_mutator = TypographyMutator()
        self.layout_mutator = LayoutMutator()
        self.content_mutator = ContentMutator()
        self.component_mutator = ComponentMutator()
        self.optimizer_available = EvolutionaryOptimizer is not None

    def mutate(self, request: MutationRequest) -> MutationResult:
        design: DesignInput = request.design
        html = design.html
        css = design.css
        changes: List[str] = []

        types = request.mutation_types or [
            'color',
            'typography',
            'layout',
            'content',
            'component',
        ]

        if request.constraints and isinstance(request.constraints.get('locked_mutations'), list):
            locked = set(request.constraints['locked_mutations'])
            types = [mutation for mutation in types if mutation not in locked]

        if 'color' in types:
            html, css, color_changes = self.color_mutator.mutate(
                html, css, request.constraints
            )
            changes.extend(color_changes)

        if 'typography' in types:
            html, css, type_changes = self.typography_mutator.mutate(html, css)
            changes.extend(type_changes)

        if 'layout' in types:
            html, css, layout_changes = self.layout_mutator.mutate(html, css)
            changes.extend(layout_changes)

        if 'content' in types:
            html, content_changes = self.content_mutator.mutate(html)
            changes.extend(content_changes)

        if 'component' in types:
            html, component_changes = self.component_mutator.mutate(html)
            changes.extend(component_changes)

        if self.optimizer_available:
            changes.append('OpenEvolve optimizer available for adaptive mutations')

        return MutationResult(html=html, css=css, changes=changes)

    async def mutate_batch(self, requests: List[MutationRequest], max_concurrency: int) -> List[MutationResult]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_one(request: MutationRequest) -> MutationResult:
            async with semaphore:
                return self.mutate(request)

        return await asyncio.gather(*[run_one(request) for request in requests])


mutation_engine = MutationEngine()
