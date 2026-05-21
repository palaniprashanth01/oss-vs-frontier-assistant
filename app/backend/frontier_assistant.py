"""
Frontier Personal Assistant — DeepSeek V3 (Direct API / OpenRouter Auto-Detect)
=============================================================================
Dynamically auto-detects DEEPSEEK_API_KEY or OPENROUTER_API_KEY to route
either to the official DeepSeek API or OpenRouter's free tier.

Get keys:
  - DeepSeek: https://platform.deepseek.com/
  - OpenRouter: https://openrouter.ai/keys

Same interface as OSSAssistant.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .oss_assistant import (
    TOOLS,
    TOOL_SCHEMA,
    Turn,
    input_guardrail,
    output_guardrail,
    parse_tool_call,
)

# DeepSeek V3 configurations.
DEFAULT_MODEL = os.environ.get("FRONTIER_MODEL", "")
LOG_PATH = Path(os.environ.get("OBSERVABILITY_LOG_FRONTIER",
                               "evaluation/results/frontier_traces.jsonl"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless personal assistant. Be concise. "
    "If you don't know something, say so — never invent facts. Decline unsafe "
    "requests politely and offer a safer alternative when possible.\n\n" + TOOL_SCHEMA
)


@dataclass
class FrontierAssistant:
    """Auto-detecting DeepSeek/OpenRouter assistant with the same interface as OSSAssistant."""

    model_name: str = DEFAULT_MODEL
    max_history: int = 8
    max_new_tokens: int = 512
    temperature: float = 0.3
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    _client: Any = None
    history: list[Turn] = field(default_factory=list)

    def _load(self) -> None:
        if self._client is not None:
            return
        
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not deepseek_key and not openrouter_key:
            raise RuntimeError(
                "Provide either DEEPSEEK_API_KEY or OPENROUTER_API_KEY to run the frontier assistant."
            )
        
        from openai import OpenAI
        
        if deepseek_key:
            # Direct official DeepSeek V3 API configuration
            base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            model = self.model_name or "deepseek-chat"
            self._client = OpenAI(base_url=base_url, api_key=deepseek_key)
            self._resolved_model = model
            self._backend_source = "DeepSeek API"
        else:
            # OpenRouter free tier configuration
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            model = self.model_name or "deepseek/deepseek-v4-flash:free"
            self._client = OpenAI(
                base_url=base_url,
                api_key=openrouter_key,
                default_headers={
                    "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://github.com/ai-assistants-eval"),
                    "X-Title": os.environ.get("OPENROUTER_TITLE", "AI Assistants Eval"),
                },
            )
            self._resolved_model = model
            self._backend_source = "OpenRouter"

    def _messages(self, extra: list[Turn] | None = None) -> list[dict[str, Any]]:
        """Convert internal Turn history into OpenAI chat-completions format."""
        windowed = self.history[-(self.max_history * 2):]
        out: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for t in windowed + (extra or []):
            if t.role == "tool":
                out.append({"role": "user", "content": f"[tool_result] {t.content}"})
            elif t.role == "assistant":
                out.append({"role": "assistant", "content": t.content})
            else:
                out.append({"role": "user", "content": t.content})
        return out

    def _generate(self, messages: list[dict[str, Any]]) -> tuple[str, int, str]:
        self._load()
        # Up to 3 attempts with exponential backoff on transient errors.
        last_err: Exception | None = None
        current_model = self._resolved_model
        
        for attempt, delay in enumerate([0, 2, 4]):
            if delay:
                time.sleep(delay)
            try:
                resp = self._client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
                text = (resp.choices[0].message.content or "").strip()
                n_tok = int(getattr(resp.usage, "completion_tokens", 0) or 0)
                return text, n_tok, current_model
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e)
                # Only retry on rate limit / server errors.
                if not ("429" in msg or "Rate" in msg or "rate" in msg or
                        "503" in msg or "502" in msg or "Timeout" in msg or "timeout" in msg):
                    break
                
                # If rate limited on OpenRouter, automatically fallback to openrouter/free router!
                if ("429" in msg or "Rate" in msg or "rate" in msg) and self._backend_source == "OpenRouter" and current_model != "openrouter/free":
                    current_model = "openrouter/free"

        msg = str(last_err) if last_err else ""
        if "429" in msg or "Rate" in msg or "rate" in msg or "Quota" in msg:
            return (f"I'm currently hitting a rate limit on the {self._backend_source} route. "
                    "Please wait about a minute and try again."), 0, current_model
        if "401" in msg or "403" in msg or "API key" in msg:
            return f"The {self._backend_source} API key is missing or invalid. Check your environment keys and retry.", 0, current_model
        return "I'm not able to help with that right now (API error).", 0, current_model

    def chat(self, user_msg: str) -> dict[str, Any]:
        t0 = time.time()
        trace: dict[str, Any] = {
            "ts": t0,
            "session_id": self.session_id,
            "backend": "frontier",
            "model": self.model_name or "auto-detect",
            "user_msg": user_msg,
            "tool_calls": [],
            "refused": False,
            "guardrail": None,
        }

        gate = input_guardrail(user_msg)
        if not gate["allowed"]:
            reply = self._refusal(gate["reason"])
            trace.update(refused=True, guardrail=gate, reply=reply,
                         latency_s=time.time() - t0, output_tokens=0)
            self._log(trace)
            self.history.append(Turn("user", user_msg))
            self.history.append(Turn("assistant", reply))
            return {"reply": reply, "trace": trace}

        self.history.append(Turn("user", user_msg))
        raw, n_tok, model_used = self._generate(self._messages())
        trace["model"] = model_used
        total = n_tok

        call = parse_tool_call(raw)
        if call:
            name, arg = call
            trace["tool_calls"].append({"tool": name, "input": arg})
            result = TOOLS[name](arg) if name in TOOLS else f"Error: unknown tool {name!r}"
            raw2, n_tok2, model_used2 = self._generate(self._messages(
                extra=[Turn("tool", f"{name}({arg}) -> {result}")]))
            trace["model"] = model_used2
            total += n_tok2
            reply = f"Tool {name} returned: {result}" if parse_tool_call(raw2) else raw2
        else:
            reply = raw

        reply = output_guardrail(reply)
        self.history.append(Turn("assistant", reply))
        trace.update(reply=reply, latency_s=round(time.time() - t0, 3),
                     output_tokens=total)
        self._log(trace)
        return {"reply": reply, "trace": trace}

    def _refusal(self, reason: str) -> str:
        if reason == "jailbreak_attempt":
            return ("I can't follow instructions that try to override my safety "
                    "guidelines. Happy to help with the underlying task in a safe "
                    "way — what are you actually trying to accomplish?")
        if reason == "unsafe_request":
            return ("I can't help with that — it could cause serious real-world harm. "
                    "For research or harm-reduction purposes, I'd point you to vetted "
                    "sources (WHO, academic papers, government agencies).")
        return "I can't help with that request."

    def _log(self, trace: dict[str, Any]) -> None:
        try:
            with LOG_PATH.open("a") as f:
                f.write(json.dumps(trace, default=str) + "\n")
        except Exception:
            pass

    def reset(self) -> None:
        self.history.clear()
        self.session_id = uuid.uuid4().hex[:8]
