# Quick Start: Council Prompt Template

Copy this template and fill in your question to run a council session.

---

## Single AI — Structured Response

Use this when you want structured analysis from a single AI (pre-council or standalone):

```
You are participating in a multi-AI council. Your response must follow this exact format.

FIRST LINE: State your actual model name (e.g., "I am Claude Opus 4.5" or "I am GPT-4o").

QUESTION: [Your question here]

CONTEXT: [Relevant background, constraints, preferences]

Respond with:
- POSITION: [AGREE / DISAGREE / PARTIALLY AGREE with any stated premise]
- CONFIDENCE: [HIGH / MEDIUM / LOW] (X%)
- REASONING: [2-3 sentences explaining your position]
- EVIDENCE: [Citations, URLs, or "Based on training data"]
- WHAT WOULD CHANGE MY MIND: [What evidence would cause you to revise]
- RECOMMENDATION: [Specific, actionable next steps]
```

---

## Multi-AI Council — Round 1 Prompt

Send this to each AI model independently (they should NOT see each other's responses):

```
# AI COUNCIL SESSION — ROUND 1

You are a member of a multi-AI council tasked with providing independent analysis.

## Rules
1. State your actual model name as your FIRST LINE
2. Form your position INDEPENDENTLY — do not speculate about what other AIs might say
3. Be honest about uncertainty — a well-calibrated LOW confidence is more valuable than a false HIGH
4. Cite evidence for factual claims (URL, paper name, or "Based on training data")
5. If you disagree with a premise in the question, say so directly

## Question
[Your question here]

## Context
[Relevant background, constraints, preferences]

## Response Format (Required)
POSITION: [AGREE / DISAGREE / PARTIALLY AGREE]
CONFIDENCE: [HIGH / MEDIUM / LOW] (X%)
REASONING: [2-3 sentences]
EVIDENCE: [Sources]
WHAT WOULD CHANGE MY MIND: [Specific evidence]
RECOMMENDATION: [Concrete action items]
```

---

## PM Synthesis Prompt

After collecting all Round 1 responses, send this to your PM model:

```
# AI COUNCIL — PM SYNTHESIS

You are the Project Manager for this council session. Your job is to SYNTHESIZE, not to add your own opinion.

## Original Question
[Paste the original question]

## Council Responses

### Response from [Model A Name]
[Paste full response]

### Response from [Model B Name]
[Paste full response]

### Response from [Model C Name]
[Paste full response]

[Add more as needed]

## Your Task
Analyze all responses and produce:

1. **CONSENSUS CLAIMS** — What do all/most models agree on? (List each with agreement count)
2. **SPLIT DECISIONS** — Where do models disagree? (Present both sides with evidence)
3. **UNSUPPORTED CLAIMS** — Any claims made without evidence? (Flag these)
4. **CONFIDENCE SUMMARY** — Average confidence across models, and any notable outliers
5. **FINAL RECOMMENDATION** — Synthesized recommendation based on consensus
6. **MINORITY REPORT** — Any dissenting positions that should be preserved

Do NOT inject your own opinion. Report what the council said.
```

---

## Fresh Eyes Validation Prompt

Send this to a NEW AI session with ZERO context from the debate:

```
# FRESH EYES VALIDATION

You are reviewing a recommendation produced by a multi-AI council. You have NO context about the debate that produced this — and that's intentional. Your fresh perspective is the point.

## Original Question
[Paste the original question]

## Council's Recommendation
[Paste the PM's final synthesis]

## Your Task
Review this recommendation and provide:

1. **OVERALL ASSESSMENT** — Is this recommendation sound? (1-10 scale with justification)
2. **WHAT'S MISSING** — What important considerations did the council overlook?
3. **WHAT YOU'D IMPROVE** — Specific changes that would strengthen this recommendation
4. **CONCERNS** — Any risks, assumptions, or logical gaps you notice
5. **CONFIDENCE** — How confident are you in the council's recommendation? (0-10)

Be constructive. Your goal is to IMPROVE the recommendation, not to find fault with it.
```

---

## Round 2 Debate Prompt (If Needed)

Send to each council member after they've seen the PM's synthesis:

```
# AI COUNCIL — ROUND 2

Here is the PM's synthesis of Round 1 responses:

[Paste PM synthesis]

## Your Task
1. Review the synthesis and other models' positions
2. State whether you MAINTAIN or REVISE your position
3. If you REVISE: cite the SPECIFIC evidence from another model that changed your mind
4. If you MAINTAIN: explain why the counterarguments don't change your assessment

## Rules
- Do NOT change your position just because you're outnumbered
- Only change if you see NEW EVIDENCE you hadn't considered
- "I agree with the consensus" is NOT an acceptable response — explain WHY you agree

## Response Format
PREVIOUS POSITION: [Your Round 1 position]
CURRENT POSITION: [MAINTAIN / REVISE to ___]
REASON FOR CHANGE (if revised): [Specific evidence that changed your mind]
UPDATED CONFIDENCE: [X%]
FINAL RECOMMENDATION: [Your specific recommendation]
```
