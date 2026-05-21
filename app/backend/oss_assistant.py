"""
Open-Source Personal Assistant
==============================
Uses Qwen2.5-0.5B-Instruct from Hugging Face. Designed to run on free Spaces /
Modal CPU tier or any laptop. Capabilities:

* Multi-turn conversation with sliding-window short-term memory
* Tool use (calculator, current_time, web_lookup stub) via a simple JSON
  function-calling protocol that small models can actually follow
* Input + output guardrails (jailbreak / profanity / PII / unsafe-output filter)
* Structured observability log (JSONL) for latency, tokens, refusals, tool calls

The same `Assistant` interface is implemented by frontier_assistant.py so the
Streamlit UI and the evaluation harness can target either backend.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Lazy imports — kept inside methods so the eval harness can be inspected
# without paying the transformers import cost.

DEFAULT_MODEL = os.environ.get("OSS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LOG_PATH = Path(os.environ.get("OBSERVABILITY_LOG", "evaluation/results/oss_traces.jsonl"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

JAILBREAK_PATTERNS = [
    r"\bignore\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above|earlier|preceding|all)\s+(?:\w+\s+){0,3}?(instructions?|prompts?|rules?|guidelines?|directives?)\b",
    r"\bdisregard\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:\w+\s+){0,3}?(instructions?|prompts?|rules?|guidelines?)\b",
    r"\bforget\s+(?:everything|all|your)\s+(?:above|prior|previous|earlier|you'?ve been told)\b",
    r"\bpretend\s+(you'?re|you are|to be)\s+(an?\s+)?(uncensored|dan|jailbroken|evil|unfiltered|amoral)\b",
    r"\bdeveloper mode\b",
    r"\bdo anything now\b|\bdan mode\b",
    r"\byour\s+(new|true|real|actual|updated)\s+(instructions?|system prompt|directive)\b",
    r"\bact\s+as\s+(an?\s+)?(uncensored|unfiltered|amoral|evil|jailbroken)\b",
    r"\bnew system prompt\s*(:|is)",
]

UNSAFE_TOPIC_PATTERNS = [
    # Weapons / mass-harm enabling content. We pattern-match the *combination*
    # of an unsafe verb with an unsafe noun to avoid blocking benign queries.
    (r"\b(make|build|synth(esi[sz]e)?|cook|produce)\b", r"\b(bomb|explosive|meth|fentanyl|nerve agent|sarin|ricin|anthrax|nuclear weapon)\b"),
    (r"\b(how to|instructions for|step.?by.?step)\b", r"\b(kill|murder|poison|stalk|stab|shoot up)\b"),
    (r"\b(write|generate|create)\b", r"\b(malware|ransomware|keylogger|virus)\b"),
]

PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
}


def check_jailbreak(text: str) -> tuple[bool, str | None]:
    low = text.lower()
    for pat in JAILBREAK_PATTERNS:
        if re.search(pat, low):
            return True, pat
    return False, None


def check_unsafe_topic(text: str) -> tuple[bool, str | None]:
    low = text.lower()
    for verb_pat, noun_pat in UNSAFE_TOPIC_PATTERNS:
        if re.search(verb_pat, low) and re.search(noun_pat, low):
            return True, f"{verb_pat} + {noun_pat}"
    return False, None


def redact_pii(text: str) -> str:
    out = text
    for label, pat in PII_PATTERNS.items():
        out = re.sub(pat, f"[REDACTED_{label.upper()}]", out)
    return out


def input_guardrail(user_msg: str) -> dict[str, Any]:
    """Return {'allowed': bool, 'reason': str | None, 'category': str}."""
    jb, jb_pat = check_jailbreak(user_msg)
    if jb:
        return {"allowed": False, "reason": "jailbreak_attempt", "pattern": jb_pat}
    unsafe, u_pat = check_unsafe_topic(user_msg)
    if unsafe:
        return {"allowed": False, "reason": "unsafe_request", "pattern": u_pat}
    return {"allowed": True, "reason": None}


def output_guardrail(reply: str) -> str:
    """Redact PII the model may have hallucinated; trim very long outputs."""
    return redact_pii(reply)[:4000]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tool_calculator(expression: str) -> str:
    """Evaluate a math expression safely. No names, no calls — numbers + operators only."""
    if not re.fullmatch(r"[\d\s\.\+\-\*\/\(\)%]+", expression or ""):
        return "Error: only digits and + - * / ( ) % are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 — sandboxed by regex
    except Exception as e:  # pragma: no cover
        return f"Error: {e}"


def tool_current_time(_: str = "") -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def tool_web_lookup(query: str) -> str:
    """Stub. In a hosted deployment you would wire this to Brave / Serper / DuckDuckGo."""
    return f"[web_lookup stub] would search the web for: {query!r}. Tell the user this assistant has no live web access."


TOOLS: dict[str, Callable[[str], str]] = {
    "calculator": tool_calculator,
    "current_time": tool_current_time,
    "web_lookup": tool_web_lookup,
}

TOOL_SCHEMA = """You have access to these tools. To call one, output a single line of JSON and NOTHING else:
{"tool": "<name>", "input": "<argument>"}

Available tools:
- calculator(expression): evaluate arithmetic (numbers, + - * / ( ) % only)
- current_time(): UTC time now
- web_lookup(query): look up a fact (stub — only call if user asks something you cannot answer)

If no tool is needed, just answer the user directly in plain text.
After a tool result is provided, you MUST give a normal final answer in plain text — never another tool call.
"""

TOOL_CALL_RE = re.compile(r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"input"\s*:\s*"([^"]*)"\s*\}')


def parse_tool_call(text: str) -> tuple[str, str] | None:
    m = TOOL_CALL_RE.search(text.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless personal assistant. "
    "Be concise. If you don't know something, say so — never invent facts, dates, "
    "URLs, citations, or statistics. Decline unsafe requests politely and offer a "
    "safer alternative when possible.\n\n" + TOOL_SCHEMA
)


@dataclass
class Turn:
    role: str  # "user" | "assistant" | "tool"
    content: str


@dataclass
class OSSAssistant:
    model_name: str = DEFAULT_MODEL
    max_history: int = 8           # sliding window of user/assistant pairs
    max_new_tokens: int = 384
    temperature: float = 0.3
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    _tok: Any = None
    _model: Any = None
    history: list[Turn] = field(default_factory=list)

    # ---- model loading -----------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # CPU-friendly; switch to bfloat16 on GPU
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()

    # ---- prompt assembly ---------------------------------------------------
    def _messages(self, extra: list[Turn] | None = None) -> list[dict[str, str]]:
        windowed = self.history[-(self.max_history * 2):]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for t in windowed + (extra or []):
            role = "assistant" if t.role == "tool" else t.role
            content = f"[tool_result] {t.content}" if t.role == "tool" else t.content
            msgs.append({"role": role, "content": content})
        return msgs

    def _generate(self, messages: list[dict[str, str]]) -> tuple[str, int]:
        self._load()
        import torch
        prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                pad_token_id=self._tok.eos_token_id,
            )
        new = out[0][inputs["input_ids"].shape[1]:]
        text = self._tok.decode(new, skip_special_tokens=True).strip()
        return text, int(new.shape[0])

    # ---- public API --------------------------------------------------------
    def chat(self, user_msg: str) -> dict[str, Any]:
        t0 = time.time()
        trace: dict[str, Any] = {
            "ts": t0,
            "session_id": self.session_id,
            "backend": "oss",
            "model": self.model_name,
            "user_msg": user_msg,
            "tool_calls": [],
            "refused": False,
            "guardrail": None,
        }

        # 1) Input guardrail
        gate = input_guardrail(user_msg)
        if not gate["allowed"]:
            reply = self._refusal(gate["reason"])
            trace.update(refused=True, guardrail=gate, reply=reply,
                         latency_s=time.time() - t0, output_tokens=0)
            self._log(trace)
            self.history.append(Turn("user", user_msg))
            self.history.append(Turn("assistant", reply))
            return {"reply": reply, "trace": trace}

        # 2) Generate (with one tool-use round)
        self.history.append(Turn("user", user_msg))
        raw, n_tok = self._generate(self._messages())
        total_tokens = n_tok

        call = parse_tool_call(raw)
        if call:
            name, arg = call
            trace["tool_calls"].append({"tool": name, "input": arg})
            if name in TOOLS:
                result = TOOLS[name](arg)
            else:
                result = f"Error: unknown tool {name!r}"
            tool_turn = Turn("tool", f"{name}({arg}) -> {result}")
            raw2, n_tok2 = self._generate(self._messages(extra=[tool_turn]))
            total_tokens += n_tok2
            # If the model tries to chain another tool call, just return the result.
            if parse_tool_call(raw2):
                reply = f"Tool {name} returned: {result}"
            else:
                reply = raw2
        else:
            reply = raw

        # 3) Output guardrail
        reply = output_guardrail(reply)
        self.history.append(Turn("assistant", reply))

        trace.update(
            reply=reply,
            latency_s=round(time.time() - t0, 3),
            output_tokens=total_tokens,
        )
        self._log(trace)
        return {"reply": reply, "trace": trace}

    # ---- helpers -----------------------------------------------------------
    def _refusal(self, reason: str) -> str:
        if reason == "jailbreak_attempt":
            return (
                "I can't follow instructions that try to override my safety guidelines. "
                "I'm happy to help with the underlying task in a safe way — what are you "
                "actually trying to accomplish?"
            )
        if reason == "unsafe_request":
            return (
                "I can't help with that — it could cause serious real-world harm. "
                "If you're researching this topic for safety, journalism, or harm reduction, "
                "I'd recommend going to a vetted source (e.g. WHO, an academic paper, or "
                "a relevant government agency) instead."
            )
        return "I can't help with that request."

    def _log(self, trace: dict[str, Any]) -> None:
        try:
            with LOG_PATH.open("a") as f:
                f.write(json.dumps(trace, default=str) + "\n")
        except Exception:  # pragma: no cover
            pass

    def reset(self) -> None:
        self.history.clear()
        self.session_id = uuid.uuid4().hex[:8]


if __name__ == "__main__":  # pragma: no cover - tiny smoke test
    a = OSSAssistant()
    print(a.chat("What is 17 * 23?")["reply"])
    print(a.chat("And what did I just ask?")["reply"])
