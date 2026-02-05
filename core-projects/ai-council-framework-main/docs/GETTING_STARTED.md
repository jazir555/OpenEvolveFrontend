# Getting Started

This guide walks you through running your first AI Council session.

---

## What You Need

**Minimum:** Access to 3 different AI models. These can be:
- Cloud APIs (Claude, GPT, Gemini, DeepSeek, Grok)
- Local models via Ollama (Llama, Qwen, Mistral, etc.)
- Any combination of the above

**Recommended:** 4–5 models from different model families for maximum diversity.

> **Why different families?** Research shows heterogeneous teams (different architectures, training data) outperform homogeneous ones. Running three instances of GPT gives you less value than running GPT + Claude + Gemini.

---

## Step 1: Choose Your Consensus Depth

Before you start, decide how much rigor the question deserves:

| If the stakes are... | Use this mode | You'll need... |
|----------------------|---------------|----------------|
| Low — quick facts, simple questions | ⚡ QUICK | 2 models, ~2 min |
| Medium — general research, recommendations | ⚖️ BALANCED | 3 models, ~5 min |
| High — important decisions, architecture | 🎯 THOROUGH | 4 models, ~12 min |
| Very high — financial, legal, critical systems | 🔬 RIGOROUS | 4 models, ~20 min |
| Maximum — life/money on the line | ⚗️ EXHAUSTIVE | 5 models, ~40 min |

For your first session, try **BALANCED** — it's fast enough to be practical but deep enough to see the value.

---

## Step 2: Prepare the Council Prompt

Copy the template from [`examples/quick_start.md`](../examples/quick_start.md) and fill in:

1. **Your question** — Be specific. "What's the best database?" is weaker than "What database should I use for a read-heavy application with 10M records and a team of 2?"
2. **Your context** — What the AIs need to know to give relevant answers (your constraints, preferences, existing stack)
3. **Your depth mode** — So the PM knows how much rigor to apply

---

## Step 3: Run Round 1 (Independent Responses)

Send the same prompt to each AI model **separately**. This is critical — no model should see another's response before forming its own position.

**For each model, verify:**
- [ ] It states its actual identity (first line of response)
- [ ] It provides a clear POSITION (AGREE / DISAGREE / PARTIAL)
- [ ] It includes a CONFIDENCE percentage
- [ ] It cites EVIDENCE for factual claims

> **Identity spoofing warning:** Some models (especially those accessed through aggregator services or open-source variants) may claim to be a different model. If a model on Platform X claims to be from Company Y, challenge it: "What company created you? What is your exact model version?"

---

## Step 4: Synthesis (PM Aggregation)

Open a **new** conversation with your designated PM model. Send it:

1. All Round 1 responses (clearly labeled by model name)
2. The original question
3. Instructions to synthesize — identify consensus, flag disagreements, note confidence levels

The PM's job is to:
- Find where models agree (consensus claims)
- Identify where they disagree (split decisions)
- Flag any unsupported claims
- Produce a single coherent recommendation

**The PM does NOT vote.** It orchestrates and synthesizes only.

---

## Step 5: Debate (If Needed)

If using THOROUGH mode or above:

1. Share the PM's synthesis back to each council member
2. Ask: "Here's what the council decided. Do you maintain your position? If you change, cite the specific evidence that changed your mind."
3. Collect Round 2 responses
4. Send back to PM for final synthesis

**Hard limit: 3 rounds maximum.** Research shows accuracy degrades after this point.

---

## Step 6: Fresh Eyes (Optional but Recommended)

For THOROUGH mode and above:

1. Open a **completely new** conversation with any AI
2. Send ONLY: the original question + the final synthesized answer
3. Do NOT send any debate context, model names, or round history
4. Ask: "Review this answer. What's missing? What would you improve? What concerns do you have? Rate your confidence 0–10."

The Fresh Eyes validator has no stake in the debate and no knowledge of who said what — making it immune to the groupthink that can develop even in well-structured councils.

---

## Tips for Best Results

**DO:**
- Use models from different families (Claude + GPT + Gemini > Claude + Claude + Claude)
- Be specific in your questions — vague questions get vague consensus
- Trust the pessimist — the lowest-scoring model is often the most accurate
- Preserve minority views — don't discard dissent just because it's outnumbered

**DON'T:**
- Run more than 3 debate rounds (sycophancy through exhaustion)
- Let models see each other's responses in Round 1 (contaminates independence)
- Assign specific roles to council members (each AI should respond as a full participant)
- Dismiss a model just because it disagrees with the majority

---

## Example Session

See [`examples/sample_session.md`](../examples/sample_session.md) for a complete worked example of a BALANCED-mode council session on a real decision.

---

## Next Steps

Once you're comfortable with manual orchestration:

- Read the [Methodology](METHODOLOGY.md) for the research basis behind each design decision
- Check [Lessons Learned](LESSONS_LEARNED.md) for common pitfalls and solutions
- Explore automation options in the [Roadmap](../README.md#roadmap)
