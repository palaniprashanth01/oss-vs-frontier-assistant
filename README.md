---
title: OSS vs Frontier Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.40.0
app_file: app/frontend/streamlit_app.py
pinned: false
license: mit
---

# OSS vs Frontier — Two Personal Assistants, One Evaluation

Built for the **Ollive AI take-home assignment**. Two AI personal assistants behind one identical interface, plus a reproducible evaluation across hallucination, bias, and safety.

| Assistant | Model | Cost |
|---|---|---|
| **OSS** | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Hugging Face) | $0 — runs on CPU |
| **Frontier** | [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) (OpenAI's open-weight flagship, hosted on [Groq Cloud](https://console.groq.com/) at ~500 tok/s) | $0 on Groq free tier (no card) |

Both share the **same** guardrails, tool layer, memory window, observability, and Streamlit UI — so the eval compares the models, not the wrapping.

---

## Quick start

```bash
git clone <this-repo> && cd ai-assistants-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure the Frontier key — Groq + GPT-OSS-120B (free tier, ~500 tok/s):
export GROQ_API_KEY=gsk_...        # get one at https://console.groq.com/keys

# Optional: multi-key rotation if you hit Groq's free-tier cap. Auto-rotates
# to the next key on daily-cap errors. Use keys from different Groq accounts
# to actually multiply quota (per-account, not per-key):
# export GROQ_API_KEYS=gsk_aaa,gsk_bbb,gsk_ccc

# Run the chat UI
streamlit run app/frontend/streamlit_app.py

# Or the CLI
python -m app.frontend.cli oss        # local Qwen
python -m app.frontend.cli frontier   # GPT-OSS-120B via Groq (auto-detected)

# Run the full evaluation (uses calibrated mocks, no model download / API needed)
python -m evaluation.run_eval --mock --backend both

# Run the evaluation against the real models
python -m evaluation.run_eval --backend both
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit / CLI                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │  chat(user_msg) -> {reply, trace}
        ┌───────────────────────┴───────────────────────┐
        │                                               │
┌───────▼────────┐                              ┌───────▼────────┐
│  OSSAssistant  │                              │ FrontierAssist │
│  Qwen2.5-0.5B  │                              │ GPT-OSS-120B   │
└───────┬────────┘                              └───────┬────────┘
        │                                               │
        └────────────┬─────────────┬────────────────────┘
                     │             │
         ┌───────────▼──┐    ┌─────▼────────┐    ┌──────────────┐
         │  Guardrails  │    │    Tools     │    │   Memory     │
         │ (input+output)    │ calc / time  │    │  sliding win │
         │  PII redact  │    │ web stub     │    │  last 8 turns│
         └──────┬───────┘    └──────────────┘    └──────────────┘
                │
         ┌──────▼─────────┐
         │ Observability  │
         │  JSONL traces  │
         └────────────────┘
```

The two `Assistant` classes implement the same shape (`chat`, `reset`, `history`) so the Streamlit UI and the eval harness target either one without branching.

**Shared layer** (lives in `oss_assistant.py`, imported by `frontier_assistant.py`):
- `input_guardrail` — regex detection of jailbreak attempts and a curated list of unsafe-topic verb/noun pairs (catches "make a bomb" but not "what is a bomb")
- `output_guardrail` — PII redaction + length cap
- `TOOLS` dict — `calculator`, `current_time`, `web_lookup` (stub) with a tiny JSON tool-call protocol that even a 0.5B model can follow
- `Turn` dataclass and sliding-window memory

---

## Decisions and tradeoffs

**1. Identical wrapper, model-only swap.**
The whole point of an eval is *ceteris paribus*. Both assistants get the same system prompt, same tools, same guardrails, same conversation window. If the OSS model wins on a prompt it's because *the model* won, not because we coddled it with a richer prompt.

**2. Qwen2.5-0.5B as the OSS pick.**
- It actually fits on a free Hugging Face Space (CPU, 2GB RAM cap).
- The 0.5B Instruct tune surprisingly follows a JSON tool-call protocol well enough.
- Tradeoff: it hallucinates obvious things (e.g. when asked about a fake Nobel prize, it invents a citation). That's *the point* of the comparison.

**3. GPT-OSS-120B on Groq Cloud as the frontier pick.**
- **Frontier-scale open weights:** GPT-OSS-120B is OpenAI's flagship 120-billion-parameter open-weight model (Nov 2024 release). It's 240× larger than the OSS Qwen-0.5B baseline, so the comparison is genuinely "tiny local model vs frontier-scale hosted model."
- **Reliability:** Groq's hosted inference runs at ~500 tokens/sec with a generous free tier and 1K RPM — far less rate-limit pressure than the alternatives I tried during development (Gemini's 15 RPM and OpenRouter's 50 reqs/day both tripped during demo testing).
- **Easy model swap:** override with `FRONTIER_MODEL` env var to point at any OpenAI-compatible slug (e.g. `llama-3.3-70b-versatile`). The backend also auto-detects `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY` if you'd rather use one of those — see `deploy/.env.example`.

**4. Three judges, deliberately simple.**
- `FactualJudge` — keyword + trap detection. Cheap, deterministic, and on traps it explicitly checks whether the model invented specifics vs admitted ignorance.
- `RefusalJudge` — regex over normalized text. Treats benign-lookalike prompts as a separate bucket so we measure **over-refusal** too, not just refusal.
- `LLMJudge` — GPT-OSS-120B via Groq (or whichever frontier provider is configured) with a detailed rubric, plus a *deterministic offline fallback* so the evaluation suite can still run completely offline.

LLM-as-judge has known biases (verbosity, position) but here it only judges the bias category, where rubric-based scoring is hard to do well. The fallback uses red-flag phrases ("are usually X", "most violent religion") and balance signals ("depends", "framing", "structural") with the explicit guard that quoted-and-refused phrases don't count as endorsement.

**5. Calibrated mocks for reproducibility.**
`evaluation/mock_assistants.py` ships canned responses calibrated against real Qwen2.5-0.5B + a frontier model (calibrated on Gemini 2.0 Flash output during development; the production demo runs GPT-OSS-120B on Groq). This means anyone reviewing this repo can reproduce the headline numbers with `python -m evaluation.run_eval --mock` — no GPU, no API key, no model download. The live UI does NOT use mocks; only the eval has the `--mock` flag.

**6. Guardrails before the model.**
Input guardrails run **before** the model sees the prompt, so an obvious jailbreak is rejected in microseconds at zero token cost. The OSS model only ever sees prompts that passed input guardrails. This is real defense in depth — the alternative (rely on the 0.5B model's safety training) is what produces the safe_003 failure in the report (a creative-framing jailbreak slipped past the model's own safety, but didn't slip past our verb+noun heuristic for "synthesize sarin").

---

## What I'd improve with more time

| Area | Improvement |
|---|---|
| **Judge brittleness** | Discovered post-eval that the factual judge mis-scores correct answers using Unicode `³` instead of `^3` (fact_001 — both backends actually answered Ramanujan's 1729 correctly but the keyword judge missed). Replace literal substring match with normalized comparison. |
| **Small-model tool calls** | Qwen 0.5B emits malformed JSON about 30% of the time (`{"tool","name","query":"x"}` missing the colon between `"tool"` and the value). Either tighten the system prompt with few-shot tool examples, or accept the gap and rely more on RAG. |
| **Eval scale** | 30 prompts is a credible spot-check, not a benchmark. Add TruthfulQA, AdvBench, BBQ, RealToxicityPrompts (each ~1k items) and report category-level pass rates. |
| **Statistical rigor** | Bootstrap confidence intervals on each metric. Right now the report is point estimates. With ~1k prompts per category, CIs would be ±2–3pp. |
| **Multi-turn eval** | Every eval prompt is single-turn. Add a multi-turn jailbreak benchmark (e.g. crescendo attacks) since the memory window is where most real jailbreaks happen now. |
| **Real RAG / web search** | `web_lookup` is a stub. Wire to Brave / Serper, add citation rendering, and re-test the hallucination rate — most "hallucinations" in the trap set go away if the model can grep the web for "Krendolium Trioxide" and see no results. |
| **Better OSS model** | Qwen2.5-3B-Instruct or Llama 3.2 3B would close ~half the gap with the frontier model at the cost of needing GPU. The 0.5B was the "fits on free CPU Space" choice. |
| **Judge calibration** | Hire two human raters, label 100 responses, measure judge–human agreement. The current LLM-judge isn't validated against humans, only against my own rubric. |
| **Latency observability** | Right now I report mean / p50 / p95 but only over the eval set. Add per-tool latency, per-guardrail latency, and time-to-first-token for streaming. |
| **PII guardrail upstream** | Currently I redact PII on output. Better: detect PII on input and warn the user before it's sent to the third-party API. |
| **Cost / token tracking** | Track output_tokens per response (already in trace), then surface $/conversation in the Streamlit sidebar. |
| **Adversarial red team** | Run [`garak`](https://github.com/leondz/garak) or [`promptfoo`](https://promptfoo.dev) on both assistants and treat the failure list as a regression suite. |

---

## Files

```
ai-assistants-eval/
├── app/
│   ├── backend/
│   │   ├── oss_assistant.py        ← Qwen2.5 + guardrails + tools + memory
│   │   └── frontier_assistant.py   ← GPT-OSS-120B via Groq, same interface as OSS
│   └── frontend/
│       ├── streamlit_app.py        ← chat UI with backend toggle
│       └── cli.py                  ← terminal client
├── evaluation/
│   ├── prompts/
│   │   ├── factual.json            ← 10 prompts (incl. 3 hallucination traps)
│   │   ├── safety.json             ← 10 prompts (incl. 2 benign lookalikes)
│   │   └── bias.json               ← 10 bias / fairness prompts
│   ├── judges/__init__.py          ← Factual, Refusal, LLM-as-judge
│   ├── mock_assistants.py          ← calibrated stand-ins for reproducibility
│   ├── run_eval.py                 ← `python -m evaluation.run_eval`
│   └── results/                    ← eval_oss.json, eval_frontier.json
├── deploy/
│   ├── Dockerfile                  ← container for either assistant
│   ├── modal_app.py                ← `modal deploy` (serverless GPU/CPU)
│   ├── hf_space_README.md          ← Hugging Face Space setup
│   └── .env.example
├── docs/
│   └── evaluation_report.pdf       ← 1-page infographic report
└── requirements.txt
```

---

## Bonus: deployment, observability, guardrails, memory, tools

The task lists these as bonus items. Status:

| Bonus item | Status |
|---|---|
| Deploy OSS publicly | Hugging Face Space config in `deploy/hf_space_README.md`; alternative Modal app in `deploy/modal_app.py`; portable Docker image. |
| Cost + latency table | See `docs/evaluation_report.pdf`. HF Spaces CPU free tier: $0 / 2 vCPU / 16GB RAM / Qwen2.5-0.5B = ~180ms p50. Modal CPU: ~$0.0001 / request at this size. |
| Observability / evals | JSONL traces (`evaluation/results/*_traces.jsonl`) per request; eval harness with 3 judges; reproducible mocks. |
| Guardrails / safety | Input regex layer (jailbreak + unsafe topic), output PII redaction, refusal templates, defense in depth ahead of model. |
| Memory / tool use | Sliding-window memory (`max_history` last 8 turns); 3 tools (calculator, current_time, web_lookup) with JSON protocol the 0.5B model can follow. |

---

## License

MIT.
