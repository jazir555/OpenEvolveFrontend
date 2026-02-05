# AI Council Framework

**A research-backed methodology for multi-AI collaborative decision-making.**

The AI Council Framework is a structured approach to orchestrating multiple AI models into a deliberative council that produces higher-quality, lower-hallucination outputs through parallel consultation, structured debate, and consensus synthesis.

> *"The architecture is technically feasible and the results are measurable — near-zero identity hallucination across 7 different AI models, with structured disagreement consistently producing better analysis than any single AI."*

---

## The Problem

Single-AI interactions suffer from well-documented failure modes:

- **Hallucination** — Models confidently state incorrect information with no self-correction mechanism
- **Sycophancy** — Models agree with users even when the user is wrong ([Perez et al., 2023](https://arxiv.org/abs/2212.09251))
- **Blind spots** — Every model has training data gaps that go undetected in single-model use
- **Groupthink** — Even multi-agent systems converge on wrong answers through mutual reinforcement ([Xiong et al., 2025](https://arxiv.org/abs/2509.05396))

## The Solution

The AI Council Framework addresses these through a structured multi-model deliberation protocol:

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI COUNCIL FRAMEWORK                        │
│                                                                 │
│  1. DISTRIBUTE — Send prompt to all council members             │
│  2. COLLECT — Gather independent responses (isolated)           │
│  3. SYNTHESIZE — Manager AI aggregates and identifies consensus │
│  4. DEBATE — Share disagreements, request evidence (max 3 rds)  │
│  5. VERIFY — Fresh Eyes validation + web search verification    │
│  6. DELIVER — Final recommendation with confidence scores       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Innovations

### User-Controlled Consensus Depth

Not every question needs the same rigor. The framework provides five configurable consensus modes that let users trade off speed vs. thoroughness:

| Mode | Models | Rounds | Consensus Target | Estimated Time |
|------|--------|--------|------------------|----------------|
| ⚡ QUICK | 2 | 0 | 50%+ | 1–2 min |
| ⚖️ BALANCED | 3 | 1 | 66%+ | 3–5 min |
| 🎯 THOROUGH | 4 | 2–3 | 80%+ | 10–15 min |
| 🔬 RIGOROUS | 4 | 3–4 | 90%+ | 18–25 min |
| ⚗️ EXHAUSTIVE | 4–5 | 5+ | 95%+ | 30–45 min |

The system can auto-suggest depth based on query analysis (e.g., "What is X?" → QUICK, "Should I invest in X?" → RIGOROUS), with user override always available.

### Anti-Sycophancy Protocol

Research shows that in multi-agent debate, stronger models often flip from correct to incorrect answers under social pressure from weaker peers. The framework enforces:

- **Independent Round 1** — No model sees other responses before forming its position
- **Evidence-required position changes** — Models cannot change stance without citing new evidence
- **Confidence-weighted voting** — Prevents low-confidence models from drowning out high-confidence positions
- **Protected dissent** — Minority positions are preserved in the final output, not erased

### The "Gemini Principle"

Named after an observed phenomenon during development: in one council session, a single AI was outnumbered 6-to-1 on three hardware architecture questions. After structured debate with evidence, five of the six other AIs revised toward the contrarian's position.

**Principle:** A lone dissenter with evidence is more valuable than a unanimous but unchallenged consensus. The framework explicitly protects and amplifies contrarian views rather than suppressing them.

### Fresh Eyes Validation

A novel addition to the multi-agent debate literature. After the council reaches consensus, a separate AI receives:

- The original question
- The final synthesized answer
- **Zero context** from the debate itself (new session, no cache)

This AI's job is constructive validation — not error-hunting (which research shows leads to hallucinated bugs), but forward-looking improvement. It catches groupthink that context-heavy systems miss because it has no stake in the debate's outcome.

### Three-Round Hard Limit

Based on findings from "Talk Isn't Always Cheap" (Xiong et al., 2025): extended deliberation causes confidence to increase while accuracy decreases. Sycophancy through exhaustion causes contrarians to capitulate.

The framework enforces a maximum of three debate rounds, after which the PM must synthesize or escalate to the human.

---

## Architecture

### Council Structure

```
┌──────────────────────────────────────────┐
│            HUMAN (User)                  │
│          Question + Depth Mode           │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│         PROJECT MANAGER (PM)             │
│    Orchestration · Synthesis · No Vote   │
└──┬──────┬──────┬──────┬──────┬───────────┘
   │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼
┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐
│AI  A││AI  B││AI  C││AI  D││AI  E│
│     ││     ││     ││     ││     │
└─────┘└─────┘└─────┘└─────┘└─────┘
   Independent Council Members

               │ (After synthesis)
               ▼
┌──────────────────────────────────────────┐
│         FRESH EYES VALIDATOR             │
│     Zero-context constructive review     │
└──────────────────────────────────────────┘
```

### Response Format

Every council member must provide structured responses:

```
POSITION: [AGREE / DISAGREE / PARTIALLY AGREE]
CONFIDENCE: [HIGH / MEDIUM / LOW] (X%)
REASONING: [2-3 sentences explaining WHY]
EVIDENCE: [Citation, URL, or "Based on training data"]
WHAT WOULD CHANGE MY MIND: [Specific evidence needed]
```

### Consensus Calculation

```
For each claim in the final answer:
  Agreement Score = Models Agreeing / (Agreeing + Disagreeing)
  (Neutral/Abstain does not count against)

Overall Consensus = Average of all claim scores

If below target threshold:
  → Flag as "Split Decision"
  → Present majority AND minority views
  → Let human decide
```

---

## Research Foundation

This framework synthesizes findings from peer-reviewed research:

| Paper | Key Finding | How It's Applied |
|-------|-------------|------------------|
| [ReConcile](https://arxiv.org/abs/2309.13007) (Chen et al., ACL 2024) | Round-table conference with confidence-weighted voting improves reasoning by +11.4% | Confidence-weighted consensus voting |
| [Multi-Agent Debate](https://arxiv.org/abs/2305.14325) (Du et al., 2023) | "Society of minds" approach reduces hallucinations | Parallel independent consultation |
| [CONSENSAGENT](https://aclanthology.org/2025.findings-acl.1141/) (ACL 2025) | Sycophancy in multi-agent debate requires dynamic prompt refinement | Anti-sycophancy protocol |
| [Chain-of-Agents](https://arxiv.org/abs/2406.02818) (Google, NeurIPS 2024) | Manager agent synthesis is critical — removing it "significantly hurt performance" | Dedicated PM synthesis role |
| [Mixture-of-Agents](https://arxiv.org/abs/2406.04692) (Together AI, 2024) | Aggregate-and-synthesize pattern; best models as final-layer aggregators | Tiered model selection |
| [Talk Isn't Always Cheap](https://arxiv.org/abs/2509.05396) (Xiong et al., 2025) | Extended debate causes stronger agents to flip to wrong answers | 3-round hard limit |
| [CriticGPT](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/) (OpenAI, 2024) | Critic agents hallucinate non-existent bugs; need constructive framing | Fresh Eyes uses forward-looking validation |

---

## Validated Results

### Identity Verification

The framework was tested across 7 AI models (Claude, GPT, Gemini, DeepSeek, Grok, Kimi/Qwen). Key findings:

- **Near-zero identity hallucination** after implementing mandatory identity declaration
- **Identity spoofing detected and corrected** — Qwen initially claimed to be Claude 3.5 Sonnet; the protocol caught and corrected this
- Consistent structured output format maintained across all models from v2.2 onward

### Cross-Validation Catches Real Errors

In council sessions, cross-model validation caught errors that no single model would have self-corrected:

- **Hallucinated tools** — One model cited "CrewAI-Desktop 0.60 with drag-and-drop Council Builder" which does not exist
- **Inflated usability scores** — Based on the hallucinated tool, leading to cascading incorrect recommendations
- **Version number fabrication** — Specific software versions cited with confidence that had never been released

### Honest Pessimism Has Value

One model consistently gave the lowest scores (Overall: 5/10, Usability: 3/10) but was arguably the most accurate, identifying that no plug-and-play solution existed for non-programmers — a finding the optimistic models glossed over.

---

## Getting Started

### Prerequisites

- Access to 3+ AI models (cloud APIs or local via Ollama)
- A way to send the same prompt to multiple models
- A designated "PM" model for synthesis

### Quick Start

1. **Choose your depth mode** based on the stakes of your question
2. **Copy the council prompt** from [`examples/quick_start.md`](examples/quick_start.md)
3. **Send Round 1** to each AI independently (no cross-contamination)
4. **Collect responses** and send to your PM for synthesis
5. **Run Fresh Eyes** if using THOROUGH mode or above

See the [Getting Started Guide](docs/GETTING_STARTED.md) for detailed instructions.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | Step-by-step setup guide |
| [Methodology](docs/METHODOLOGY.md) | Detailed explanation of the framework's design decisions |
| [Research Notes](docs/RESEARCH_NOTES.md) | Annotated bibliography and research findings |
| [Lessons Learned](docs/LESSONS_LEARNED.md) | What worked, what didn't, and why |
| [FAQ](docs/FAQ.md) | Common questions and answers |

---

## Related Projects

The multi-AI council space is growing. Here are some related open-source implementations:

| Project | Approach | Key Difference from This Framework |
|---------|----------|-----------------------------------|
| [ai-council-mcp](https://github.com/0xAkuti/ai-council-mcp) | MCP server, parallel query + anonymous synthesis | No memory, no multi-round debate |
| [ai-counsel](https://github.com/blueman82/ai-counsel) | Multi-round deliberation with convergence detection | Closer to this framework; adds decision graph memory |
| [multi-ai-advisor-mcp](https://github.com/YuChenSSR/multi-ai-advisor-mcp) | Ollama-native with per-model personas | Simpler, role-based rather than debate-based |
| [second-opinion](https://github.com/ProCreations-Official/second-opinion) | Code review focused, multiple model consultation | Domain-specific (coding), not general-purpose |

---

## Roadmap

- [ ] Automated orchestration (Python-based council runner)
- [ ] MCP server integration for plug-and-play use
- [ ] Memory persistence layer for cross-session learning
- [ ] Benchmarking suite for measuring council accuracy vs. single-model
- [ ] Domain-specific prompt templates (business, technical, creative, personal)

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where help is especially valuable:
- Benchmarking against single-model baselines
- Domain-specific prompt templates
- Automated orchestration tooling
- Multi-language support

---

## Citation

If you use this framework in research or production, please cite:

```
@misc{fevrier2026aicouncil,
  title={AI Council Framework: Research-Backed Multi-AI Collaborative Decision-Making},
  author={Février, Stanley},
  year={2026},
  url={https://github.com/focuslead/ai-council-framework}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Stanley Février**

Framework design and iterative development through AI-directed methodology. Built through systematic experimentation with 7+ AI models, cross-validated against peer-reviewed research in multi-agent systems.

*This framework was developed using the methodology it describes — multiple AI perspectives, structured debate, and evidence-based consensus.*
