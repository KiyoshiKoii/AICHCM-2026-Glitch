# LLM parser model evaluation

Last reviewed: 2026-07-20.

The parser is a short Vietnamese-to-English transformation with a strict JSON
schema. Long-context or deep reasoning performance matters less than Vietnamese
instruction following, structured-output reliability, latency, and cost.

| Candidate | Deployment | Strength for this parser | Main trade-off | Suggested role |
| --- | --- | --- | --- | --- |
| Qwen3 8B via Ollama | Local | Strong multilingual instruction following; no per-call fee; data stays local | Quality and latency depend on local hardware; must be benchmarked on Vietnamese queries | Default development model |
| Gemma 3 4B/12B via Ollama | Local | Small multilingual alternatives for constrained hardware | May need more prompt tuning than larger models | Local fallback / benchmark candidate |
| GPT-5 mini (or current low-latency successor) | Paid API | Structured outputs and strong adherence for a narrow, high-volume task | External dependency and per-token cost | Production quality/cost baseline |
| Claude Sonnet 5 | Paid API | High instruction-following quality and structured outputs | More capability and cost than this parser may need | Accuracy-oriented benchmark |
| Gemini 3.5 Flash | Paid API | Low-latency model with structured outputs and broad multilingual input | Separate provider integration and operational dependency | Price/latency benchmark |

## Recommendation

1. Develop with `qwen3:8b` through Ollama and JSON Schema enforcement.
2. Build a fixed Vietnamese evaluation set before choosing a paid provider.
3. Compare exact-schema rate, semantic coverage, hallucinated visual details,
   p50/p95 latency, and cost per 1,000 queries.
4. Promote a paid model only if it materially improves retrieval metrics such
   as Recall@20 or MRR, not merely because its prose looks better.

The current service keeps the parser behind a protocol so an OpenAI, Anthropic,
or Gemini adapter can be added without changing search fusion.

## Minimum evaluation set

- 150-300 Vietnamese queries across objects, people, actions, location, color,
  counting, negation, before/after language, slang, and spelling errors.
- Human-authored expected visual prompts and acceptable synonym sets.
- Three repeated runs per query for local models.
- Failure labels: invalid JSON, missing field, mistranslation, invented detail,
  lost temporal cue, and weak synonym coverage.

## Official references

- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama Qwen3 8B](https://ollama.com/library/qwen3:8b)
- [OpenAI GPT-5 mini](https://developers.openai.com/api/docs/models/gpt-5-mini)
- [Anthropic Claude Sonnet 5](https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5)
- [Google Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash)

