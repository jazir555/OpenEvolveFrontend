"""
Proof of concept: SOP Generator + Research-Quest integration.
"""

import asyncio

from sop_generator_research_quest import ResearchQuestSOPGenerator


async def main() -> None:
    generator = ResearchQuestSOPGenerator(domain="research")
    research_question = "How can we improve reproducibility of LLM evaluation benchmarks?"

    stage_1_sop = await generator.generate_stage_sop(
        research_question=research_question,
        stage_id=1,
        constraints=["Include reproducibility checklist"],
        context="Focus on open-source evaluation datasets and tooling.",
    )

    print(stage_1_sop.to_markdown()[:1200])


if __name__ == "__main__":
    asyncio.run(main())
