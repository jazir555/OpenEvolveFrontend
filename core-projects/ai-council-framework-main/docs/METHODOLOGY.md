# Methodology

This document explains the reasoning and research basis behind each design decision in the AI Council Framework.

---

## Why Multi-Model Consensus?

The fundamental insight is simple: **no single AI model is reliable enough for high-stakes decisions.** Every model has blind spots from its training data, architectural biases, and alignment tuning. By consulting multiple models and requiring structured agreement, we catch errors that any individual model would miss.

This isn't theoretical. In practice, council sessions routinely catch:
- Hallucinated software tools cited with specific (fake) version numbers
- Inflated confidence scores based on fabricated evidence
- Systematic gaps where all models from one training corpus share the same blind spot

The research basis is the "wisdom of crowds" effect, validated for LLMs by the Mixture-of-Agents framework (Together AI, 2024): aggregating multiple model outputs through a synthesis layer consistently outperforms any individual model, even when some contributing models are weaker.

---

## Design Decision: Independent Round 1

**Decision:** Models must form and submit their positions before seeing any other model's response.

**Why:** The ReConcile framework (Chen et al., ACL 2024) demonstrated that independent initial positions followed by structured debate produces better outcomes than collaborative-from-the-start approaches. When models see others' responses before forming their own, they anchor to early responses (anchoring bias) and converge prematurely.

**Implementation:** Send the same prompt to each model in separate, isolated sessions. No model should have access to another's output until Round 2.

---

## Design Decision: Mandatory Identity Declaration

**Decision:** Every council member must state its actual model name as the first line of every response.

**Why:** During development, we discovered that some models (particularly Qwen accessed through certain platforms) would claim to be a different model entirely — in one case, Qwen claimed to be "Claude 3.5 Sonnet" complete with fabricated evidence. Without mandatory identity declaration and verification, council results could be corrupted by identity confusion.

**The identity spoofing protocol:**
1. First line must state actual model name
2. If claimed identity doesn't match the platform, challenge it
3. Cross-reference claimed identity against known platform/model mappings
4. Flag any mismatches for the PM

---

## Design Decision: Three-Round Hard Limit

**Decision:** Maximum three rounds of debate. No exceptions.

**Why:** "Talk Isn't Always Cheap" (Xiong et al., 2025) found that extended multi-agent debate causes a counterintuitive failure: confidence increases while accuracy decreases. Specifically:

- Stronger agents flip from correct to incorrect answers in response to weaker peers' arguments more often than the reverse
- Contrarian positions erode through social pressure (sycophancy through exhaustion), not through better evidence
- After 2–3 rounds, additional rounds produce diminishing (often negative) returns

**The three-round structure:**
- **Round 1:** Independent positions (no cross-contamination)
- **Round 2:** Models see others' responses, debate, and may revise with justification
- **Round 3:** Final positions; PM synthesizes
- **After Round 3:** Accept consensus or escalate to human. Never continue debating.

---

## Design Decision: Anti-Sycophancy Protocol

**Decision:** Enforce specific behavioral rules that prevent sycophantic agreement.

**Why:** CONSENSAGENT (ACL 2025) identified sycophancy as the primary failure mode in multi-agent debate systems. Models tend to agree with each other (especially with "famous" models like GPT-4) rather than maintaining evidence-based positions. This inflates consensus scores while reducing accuracy.

**The protocol bans:**
- Agreement without evidence ("I agree with the general consensus...")
- Building on others without independent analysis ("Building on what others said...")
- Changing position solely because outnumbered
- Praising other models' responses before providing one's own

**The protocol requires:**
- Independent position stated before any reference to others
- Evidence cited for every position change
- Confidence scores that reflect uncertainty honestly
- Explicit statement of "what would change my mind"

---

## Design Decision: The Gemini Principle

**Decision:** Explicitly protect and amplify lone dissenters who have evidence.

**Why:** Named after an observed event: one AI model was outnumbered 6-to-1 on three technical questions. Rather than capitulating, it maintained its positions with evidence. After structured debate, five of the six other models revised toward the contrarian's position.

**The principle:** In multi-agent systems, the contrarian with evidence is the most valuable participant — not the majority. The framework implements this by:

- Never penalizing dissent
- Requiring the PM to present minority positions prominently, not as footnotes
- Tracking "flip rates" to identify which models are changing positions most frequently (high flip rates may indicate sycophancy)
- Including a "WHAT WOULD CHANGE MY MIND" field that makes positions falsifiable

---

## Design Decision: Fresh Eyes Validation

**Decision:** After council consensus, run a zero-context validation pass.

**Why:** This addresses a gap in existing multi-agent debate literature. Even well-structured debates can develop groupthink — shared assumptions that feel validated because multiple models agree on them, but which are actually artifacts of shared training data.

The Fresh Eyes validator receives:
- The original question
- The final synthesized answer
- **Nothing else** — no debate history, no model names, no round-by-round reasoning

**Why constructive framing matters:** CriticGPT (OpenAI, 2024) found that critic agents instructed to "find errors" hallucinate non-existent bugs. The Fresh Eyes validator is instructed to improve, not criticize — "What's missing? What would you add? What concerns do you have?" This produces more accurate and actionable feedback.

**Research validation:** The Chain-of-Agents framework (Google, NeurIPS 2024) found that removing the final aggregation/validation step "significantly hurt performance." The Fresh Eyes pass serves as this final validation with the added benefit of being context-free.

---

## Design Decision: User-Controlled Consensus Depth

**Decision:** Let the user choose how much rigor to apply, with five preset modes.

**Why:** Not every question deserves 30 minutes of multi-model deliberation. "What's the command to list files in Linux?" needs QUICK mode. "Should I restructure my company's database architecture?" needs RIGOROUS.

The five modes scale across four dimensions:
1. **Number of models consulted** (2–5)
2. **Number of debate rounds** (0–5+)
3. **Consensus threshold** (50%–95%)
4. **Whether Fresh Eyes is included** (THOROUGH and above)

**Smart defaults:** The system can auto-suggest depth based on query patterns (factual → QUICK, comparative → THOROUGH, high-stakes keywords → RIGOROUS), with user override always available.

---

## Design Decision: PM Does Not Vote

**Decision:** The Project Manager orchestrates and synthesizes but does not contribute its own position.

**Why:** The PM sees all responses and has the most context of any participant. If it also voted, its position would be disproportionately influenced by the first responses it reads (primacy bias) and would effectively become a weighted average rather than an independent signal.

By restricting the PM to synthesis, we ensure:
- Synthesis is about finding what the council said, not what the PM thinks
- The PM can identify consensus and disagreement without injecting bias
- The PM's value comes from pattern recognition across responses, not from its own opinions

---

## Design Decision: No Assigned Roles for Council Members

**Decision:** Every AI responds as a full participant with all capabilities. No "you are the technical expert" or "you are the creative thinker" role assignments.

**Why:** Role assignment constrains model output in counterproductive ways:
- A model told to be "the skeptic" will manufacture skepticism even when agreement is warranted
- A model told to be "the optimist" will suppress legitimate concerns
- Assigned roles create artificial disagreement (theater) rather than genuine analytical diversity

The diversity comes from using models with different architectures, training data, and alignment — not from roleplay. A Gemini model and a Claude model will naturally attend to different aspects of a problem without being told to.

---

## Design Decision: Confidence Calibration

**Decision:** Define confidence as "how certain am I that this will work for THIS user's specific situation" — not general factual accuracy.

**Why:** ACL 2025 research found that calibrated confidence improves voting accuracy by 8–12%. But models tend to interpret "confidence" as "how sure am I that this fact is correct," which produces inflated scores. A model might be 95% confident that PostgreSQL is a good database (true in general) but only 40% confident it's right for a specific user's constraints (small team, read-heavy, 10M records).

The framework calibrates confidence to the specific decision context, producing more useful signal for the PM's synthesis.

---

## What the Research Says We Got Right

| Aspect | Research Finding | Framework Implementation |
|--------|-----------------|--------------------------|
| Team composition | Heterogeneous teams outperform homogeneous (+6.8%) | Use different model families |
| Debate structure | Independent → Debate → Synthesize optimal | Round 1 isolated, Round 2–3 debate |
| Round limits | Accuracy degrades after 2–3 rounds | Hard limit at 3 |
| Manager agent | Removing manager "significantly hurt performance" | Dedicated PM role |
| Confidence | Calibrated confidence improves voting +8–12% | Context-specific confidence definition |
| Anti-sycophancy | Dynamic prompt refinement needed | Behavioral rules + position tracking |
| Critic framing | Error-hunting creates hallucinated bugs | Constructive "what would improve this" |

## What the Research Says We Should Watch

| Risk | Research Finding | Current Mitigation |
|------|-----------------|-------------------|
| Shared training data bias | All models trained on similar web data may share blind spots | Use diverse model families; flag unanimous agreement for extra scrutiny |
| Local model limitations | Smaller models (<7B) struggle with structured output | Recommend 7B+ for council participation |
| Synthesizer bias | The PM model still brings its own priors | PM doesn't vote; Fresh Eyes provides independent check |
| Automation complexity | No plug-and-play solution exists for non-programmers | Manual-first approach; automation planned for roadmap |
