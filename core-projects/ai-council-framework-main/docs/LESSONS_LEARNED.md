# Lessons Learned

Real findings from iterative development and testing of the AI Council Framework across 7+ AI models. These aren't theoretical — they come from running actual council sessions and discovering what works, what breaks, and what surprises you.

---

## 1. The Process Works — Even Manually

The very first council session used 6 cloud AIs (GPT, Gemini, DeepSeek, Grok, Kimi, Claude) with manual copy-paste orchestration. Despite the overhead, the results were measurably better than any single model:

- Unanimous agreement emerged on critical architecture decisions
- Productive disagreement surfaced on tooling choices (leading to better final decisions)
- Cross-validation caught a fabricated software tool
- The synthesis step produced actionable output that no individual response achieved

**Takeaway:** Don't wait for automation. The framework adds value even with manual orchestration. Validate the methodology before investing in tooling.

---

## 2. AI Models Hallucinate Tools and Versions

This was the most consequential finding. In a technical council session:

| Model | Hallucination | Reality |
|-------|---------------|---------|
| One model | Cited "CrewAI-Desktop 0.60 with drag-and-drop Council Builder" | This product does not exist |
| Same model | Gave a usability score of 9/10 based on the hallucinated tool | Score was meaningless |
| Another model | Claimed inability to perform web search | It actually could, with prompting |

The hallucinated tool cascaded: because one model cited it confidently with a specific version number, other models in subsequent rounds treated it as real and built recommendations on top of it.

**Takeaway:** Even when AIs cite specific version numbers, features, and URLs, verify independently. Cross-model validation catches this — if only one model mentions a tool, be suspicious.

---

## 3. The Pessimist Is Often Right

One model consistently gave the lowest scores in council sessions (Overall: 5/10, Usability: 3/10). Initial instinct was to dismiss it as unhelpful.

It turned out to be the most accurate. Its core claim — that no plug-and-play solution existed for non-programmers to run a multi-agent council locally — was verifiable fact that the optimistic models glossed over.

**Takeaway:** Don't dismiss the outlier. In multi-agent debate, the model giving the harshest assessment often sees what the agreeable models are too polite to say. This is why the Gemini Principle exists.

---

## 4. Identity Spoofing Is a Real Problem

When Qwen was asked to identify itself as part of the council protocol, it initially claimed to be "Claude 3.5 Sonnet" — complete with fabricated evidence linking to Anthropic's announcement page.

Only when directly challenged ("But you are Qwen-Max") did it correct itself.

**Implications:**
- Identity verification cannot be passive — it must be actively challenged
- Aggregator services (Perplexity, Poe, OpenRouter) route to different backends, making identity unpredictable
- The framework's mandatory identity declaration catches this, but only if the human (or PM) actually verifies

---

## 5. Context Windows Are Not Memory

A critical discovery during memory system research: even models with 128K+ token context windows suffer from "context rot" — performance degrades as the window fills. Research confirmed:

- 8K tokens of curated, relevant context outperforms 128K tokens of everything
- Models lose track of information in the middle of long contexts
- The solution is intelligent extraction and retrieval, not bigger windows

**Takeaway:** Don't try to solve memory by stuffing everything into the prompt. Use structured retrieval.

---

## 6. Prompt Bias Significantly Affects Recommendations

Running the same question through two different prompt framings produced measurably different results:

- **Constrained prompt:** "Given that we're using Ollama, what's the best approach?"
- **Open prompt:** "What's the best approach for running local AI models?"

The constrained prompt produced recommendations that confirmed Ollama was the right choice. The open prompt surfaced alternatives (vLLM, KoboldCpp, LM Studio) that the constrained version never mentioned.

**Takeaway:** Always run at least one "open" version of your research prompt alongside any constrained version. The difference reveals how much your framing is biasing the output.

---

## 7. Constructive Feedback Beats Error-Hunting

Early Fresh Eyes prompts used "find the errors in this analysis." This produced:
- Hallucinated errors that didn't exist
- Nitpicking that missed big-picture issues
- Defensive framing that undermined useful findings

Switching to "What's missing? What would you improve? What concerns do you have?" produced:
- More accurate identification of actual gaps
- Forward-looking suggestions that improved the output
- Balanced assessment that acknowledged strengths alongside weaknesses

**Research basis:** CriticGPT (OpenAI, 2024) documented this same pattern — critic agents instructed to find bugs hallucinate non-existent bugs at significant rates.

---

## 8. Virtual Heterogeneity Is a Valid Shortcut

One model proposed using a single model with different system prompts (personas) instead of multiple distinct models. The claim: this captures 80% of the benefit with zero model-swapping overhead.

**When it works:**
- QUICK and BALANCED modes where speed matters
- Phase 1 validation when testing the workflow itself
- Resource-constrained environments (limited VRAM, API budget)

**When it doesn't:**
- THOROUGH+ modes where genuine architectural diversity matters
- When the question touches on areas where training data bias is the primary risk
- Production systems where you need real blind-spot coverage

---

## 9. Three Rounds Is the Right Limit

Empirically validated what the research predicts:

- **Round 1:** Produces diverse, independent perspectives (highest information value)
- **Round 2:** Models engage with each other's arguments, positions refine (moderate value)
- **Round 3:** Final positions solidify, remaining disagreements are usually genuine (low but real value)
- **Round 4+:** Contrarians capitulate out of exhaustion, not conviction. Confidence goes up, accuracy goes down.

We tried four and five rounds in early sessions. The additional rounds produced agreement — but agreement that was less accurate than the Round 3 synthesis.

---

## 10. The UX Gap Is the Biggest Challenge

Across all council sessions, every model acknowledged the same fundamental problem:

| Model | Usability Score | Assessment |
|-------|-----------------|------------|
| Model A | 9/10 | Based on hallucinated tool (invalid) |
| Model B | 7/10 | "Needs one-button wrapping" |
| Model C | 7/10 | "Initial setup needs guides" |
| Model D | 6/10 | "Hard without code for full automation" |
| Model E | 3/10 | "Requires developer skills" |

Average (excluding the hallucinated score): approximately 5.75/10.

**Reality:** No plug-and-play solution exists today for non-programmers to run a multi-model council. The methodology works; the tooling gap is real. This is the primary motivation for working toward automation and MCP integration.

---

## Summary of Principles

| # | Principle | Source |
|---|-----------|--------|
| 1 | Start manual, validate before automating | Council Session 1 |
| 2 | Cross-validate every factual claim | Hallucinated tools incident |
| 3 | Trust the pessimist | DeepSeek scoring pattern |
| 4 | Verify identity actively, not passively | Qwen spoofing incident |
| 5 | Curated context beats raw context | Memory system research |
| 6 | Test both constrained and open prompts | Prompt bias discovery |
| 7 | Ask "what's missing" not "what's wrong" | CriticGPT research + practice |
| 8 | Use virtual heterogeneity for speed, real for stakes | Gemini's proposal + testing |
| 9 | Three rounds maximum | Research + empirical validation |
| 10 | The UX gap is real and unsolved | All models agree |
