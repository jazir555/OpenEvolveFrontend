# Research Notes

Annotated bibliography and research findings that inform the AI Council Framework. These are the papers and frameworks we studied, what they found, and how those findings shaped our design.

---

## Core Papers

### ReConcile: Round-Table Conference Improves Reasoning via Consensus
**Chen et al., ACL 2024** — [arXiv:2309.13007](https://arxiv.org/abs/2309.13007)

A multi-model, multi-agent framework designed as a round-table conference among diverse LLM agents. Uses multiple rounds of discussion where agents learn to convince others, combined with confidence-weighted voting.

**Key findings:**
- Up to 11.4% improvement over single-model baselines
- Even outperformed GPT-4 on three benchmark datasets
- Confidence-weighted voting (where more certain models have more influence) is critical
- Explicit agree/disagree with justification produces better outcomes than open discussion

**How we use it:** The structured response format (POSITION / CONFIDENCE / EVIDENCE) and confidence-weighted consensus calculation come directly from ReConcile.

---

### Multi-Agent Debate Improves Mathematical and Strategic Reasoning
**Du et al., 2023** — [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)

Proposes a "society of minds" approach where multiple LLM instances debate to improve factuality and reduce hallucinations.

**Key findings:**
- Debate between models reduces hallucination rates
- Independent reasoning followed by cross-examination outperforms single-pass generation
- The approach works across mathematical, strategic, and factual reasoning tasks

**How we use it:** The fundamental architecture — independent responses followed by structured debate — is the backbone of the framework.

---

### CONSENSAGENT: Addressing Sycophancy in Multi-Agent Debate
**Pitre et al., ACL 2025** — [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.1141/)

Directly addresses the sycophancy problem in multi-agent debate, where agents reinforce each other's responses instead of critically engaging.

**Key findings:**
- Sycophancy is the primary failure mode in multi-agent debate
- It inflates computational costs by requiring additional debate rounds
- Dynamic prompt refinement can mitigate (but not eliminate) sycophantic behavior

**How we use it:** The anti-sycophancy protocol (forbidden agreement phrases, required evidence for position changes, protected dissent) is our practical implementation of CONSENSAGENT's findings.

---

### Chain-of-Agents: Large Language Models Collaborating on Long-Context Tasks
**Google Research, NeurIPS 2024** — [arXiv:2406.02818](https://arxiv.org/abs/2406.02818)

Worker agents process information sequentially, passing summaries forward. A manager agent at the end receives all accumulated evidence and synthesizes the final output.

**Key findings:**
- Removing the manager agent "significantly hurt performance"
- The manager's value comes from seeing ALL context and synthesizing across perspectives
- Sequential processing with final aggregation outperforms parallel-only approaches

**How we use it:** The dedicated PM role (which sees all responses but doesn't vote) and the final synthesis step are directly inspired by CoA's manager agent pattern.

---

### Talk Isn't Always Cheap: Multi-Agent Debate Failure Modes
**Xiong et al., 2025** — [arXiv:2509.05396](https://arxiv.org/abs/2509.05396)

Examines when and why multi-agent debate fails, challenging the assumption that more debate is always better.

**Key findings:**
- Stronger agents flip from correct to incorrect answers in response to weaker peers' arguments more often than the reverse
- Extended debate causes confidence to increase while accuracy decreases
- Sycophancy through exhaustion is a real failure mode — contrarians capitulate to end the debate, not because they're convinced

**How we use it:** The three-round hard limit. This paper is why we don't allow Round 4+, regardless of whether consensus has been reached.

---

### Mixture-of-Agents
**Together AI, 2024** — [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)

Proposes an aggregate-and-synthesize pattern where multiple models contribute and the best model serves as the final-layer aggregator.

**Key findings:**
- Aggregating multiple model outputs through a synthesis layer consistently outperforms individual models
- The strongest model should be the synthesizer, not a participant
- Even weaker models contribute useful signal when their outputs are properly aggregated

**How we use it:** The PM selection guidance (use the strongest available model for synthesis) and the principle that all council members contribute value regardless of individual capability.

---

### CriticGPT: Finding GPT-4's Mistakes with GPT-4
**OpenAI, 2024** — [OpenAI Blog](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/)

Trained a model specifically to find bugs in code, then studied its failure modes.

**Key findings:**
- Critic agents hallucinate non-existent bugs at significant rates
- "Helpfulness" is more important than "discriminability" — critics should aim to help, not just judge
- Error-focused framing produces more false positives than constructive framing

**How we use it:** The Fresh Eyes validator uses constructive framing ("What's missing? What would you improve?") instead of error-hunting ("Find the bugs in this analysis") specifically because of CriticGPT's findings.

---

### MAD Framework Research
**Wu et al., 2025** — [EmergentMind Analysis](https://www.emergentmind.com/topics/multiagent-debate-framework)

Comprehensive analysis of what works and doesn't in multi-agent debate frameworks.

**Key findings:**
- Heterogeneous teams outperform homogeneous teams by approximately 6.8%
- Hiding confidence scores during debate prevents over-confidence cascades
- MAD cannot exceed the accuracy of its strongest participant
- Low-performing or over-confident agents can degrade team output

**How we use it:** Model diversity requirements, the guideline to hide confidence during debate rounds (reveal only for final vote), and the warning that weak models can actively harm council output.

---

## Additional Research

### Latent Collaboration in Multi-Agent Systems
**2025** — Preprint

Finds that sharing rich internal representations ("latent thoughts") between agents leads to +14.6% accuracy improvement over simple text-based debate.

**Relevance:** Supports the value of the PM synthesis phase, where the PM has access to the full reasoning of all participants, not just their conclusions.

---

### Memory in the Age of AI Agents
**arXiv:2512.13564** — December 2025

Definitive survey covering 200+ papers on memory systems for AI agents. Establishes that context windows are not memory and that intelligent retrieval is necessary for persistent, accurate AI systems.

**Relevance:** Informed the framework's approach to memory (planned feature) and validated the principle that curated context outperforms raw context.

---

## Open Questions

These are areas where the research is still evolving and the framework may need to adapt:

1. **Optimal council size:** Most research uses 3–5 agents, but is there a sweet spot for specific task types?
2. **Model selection criteria:** Beyond "different families," what specific model characteristics maximize diversity of thought?
3. **Automated sycophancy detection:** Can we measure sycophancy in real-time rather than relying on prompt-based prevention?
4. **Scaling beyond text:** How does multi-agent debate apply to multimodal tasks (image analysis, code review, creative work)?
5. **Longitudinal learning:** Can councils improve over time by learning from their own past decisions?
