# Frequently Asked Questions

---

## General

**Q: What's the difference between this and just asking multiple AIs the same question?**

Asking multiple AIs gives you multiple answers. The AI Council Framework gives you a *structured process* for turning those answers into a single, validated recommendation. The difference is in the protocol: independent Round 1 (prevents anchoring), structured debate with evidence requirements (prevents sycophancy), PM synthesis (identifies real consensus vs. artificial agreement), and Fresh Eyes validation (catches groupthink). Without this structure, you just have a pile of opinions.

---

**Q: Does this actually work better than a single good AI?**

Yes, measurably. ReConcile (ACL 2024) showed up to 11.4% improvement over single-model baselines. In our testing, the framework catches hallucinated tools, fabricated version numbers, and overconfident claims that no single model self-corrects. The value scales with the stakes of the question — for "what's the capital of France," a single AI is fine. For "should I restructure my database architecture," the council process consistently produces better outcomes.

---

**Q: How long does a council session take?**

Depends on your depth mode. QUICK takes 1–2 minutes. BALANCED takes 3–5 minutes. THOROUGH takes 10–15 minutes. RIGOROUS takes 18–25 minutes. EXHAUSTIVE takes 30–45 minutes. Most questions are well-served by BALANCED or THOROUGH.

---

**Q: Can I use this with local models (Ollama, LM Studio)?**

Yes. The framework is model-agnostic. Any AI that can read a prompt and produce a structured response works. Local models via Ollama are particularly useful for privacy-sensitive queries. Research suggests using 7B+ parameter models for reliable structured output — smaller models may struggle with the response format.

---

## Design Choices

**Q: Why only 3 rounds of debate?**

Research shows accuracy degrades after 2–3 rounds. "Talk Isn't Always Cheap" (Xiong et al., 2025) found that extended debate causes stronger models to flip from correct to incorrect answers — sycophancy through exhaustion. We've validated this empirically: Round 4+ produces agreement that is less accurate than Round 3 synthesis.

---

**Q: Why doesn't the PM vote?**

The PM sees all responses and has the most context. If it also voted, its position would be biased by the first responses it reads (primacy effect). Restricting the PM to synthesis ensures its value comes from pattern recognition across responses, not from injecting its own opinions into the consensus.

---

**Q: Why no assigned roles (skeptic, optimist, etc.)?**

Assigned roles create artificial disagreement. A model told to be "the skeptic" will manufacture skepticism even when agreement is warranted. Real analytical diversity comes from using models with different architectures and training data, not from roleplay. Each council member should respond as a full participant with all capabilities.

---

**Q: What if all models agree? Is that good or bad?**

It depends. Unanimous agreement on a well-studied factual question is probably correct. Unanimous agreement on a novel, complex, or opinion-driven question should trigger extra scrutiny — it might mean all models share the same training data bias. The framework recommends running Fresh Eyes validation on unanimous results for THOROUGH+ modes.

---

## Practical

**Q: Which AI models should I use?**

Use models from different families for maximum diversity. A good starting combination: Claude + GPT + Gemini gives you three different architectures, training corpuses, and alignment approaches. Add DeepSeek or Grok for additional diversity. Avoid using multiple instances of the same model family (e.g., GPT-4 + GPT-4o) — they share blind spots.

---

**Q: Can I automate this?**

Not yet with a plug-and-play solution. The framework currently works best with manual orchestration (copy-paste between models). Automation is on the roadmap — see Related Projects in the README for existing MCP-based council implementations that partially automate the process.

---

**Q: How do I handle models that refuse to answer?**

Some models will decline certain queries based on their safety training. If a model refuses, note this in the council record and proceed with the remaining models. A refusal is itself useful signal — it tells you the question may be at the boundary of what AI systems are comfortable advising on, which should factor into your confidence assessment.

---

**Q: What about API costs?**

For cloud models, each council session uses N × (prompt tokens + response tokens), where N is the number of models. QUICK mode with 2 models is roughly 2x a single query. EXHAUSTIVE with 5 models and multiple rounds could be 15–20x. Local models via Ollama have zero API cost but require appropriate hardware (8GB+ VRAM recommended, 16GB+ for larger models).

---

**Q: Is there a minimum number of models needed?**

Two models give you a basic comparison. Three is the practical minimum for meaningful consensus (majority vote becomes possible). Four to five is the sweet spot for most questions. Beyond five, the marginal value of additional models drops significantly while orchestration complexity increases.
