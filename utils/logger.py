from __future__ import annotations

import json
import os
import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any, Callable


Json = dict[str, Any]
StreamCallback = Callable[[Json], None]


@dataclass
class TraceStep:
    timestamp: float
    step: str
    input_data: Any | None = None
    output_data: Any | None = None
    confidence: float | None = None
    tool_name: str | None = None
    streaming_msg: str | None = None
    meta: Json | None = None

    def to_dict(self) -> Json:
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "confidence": self.confidence,
            "tool_name": self.tool_name,
            "streaming_msg": self.streaming_msg,
            "meta": self.meta or {},
        }


class Logger:
    """
    Captures the full agent execution trace (inputs/outputs/tool calls).
    Also optionally streams human-readable step updates to the UI.
    """

    def __init__(self, stream_callback: StreamCallback | None = None) -> None:
        self._steps: list[TraceStep] = []
        self._stream_callback = stream_callback

    @property
    def steps(self) -> list[TraceStep]:
        return self._steps

    def step(
        self,
        step: str,
        input_data: Any | None = None,
        output_data: Any | None = None,
        confidence: float | None = None,
        tool_name: str | None = None,
        streaming_msg: str | None = None,
        meta: Json | None = None,
    ) -> None:
        trace_step = TraceStep(
            timestamp=time.time(),
            step=step,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            tool_name=tool_name,
            streaming_msg=streaming_msg,
            meta=meta,
        )
        self._steps.append(trace_step)

        if self._stream_callback and streaming_msg:
            self._stream_callback(
                {
                    "type": "step",
                    "step": step,
                    "message": streaming_msg,
                    "confidence": confidence,
                    "tool_name": tool_name,
                    "meta": meta or {},
                }
            )

    def to_trace(self) -> list[Json]:
        return [s.to_dict() for s in self._steps]


def call_gemini_json(prompt: str, schema_hint: str | None = None, model_name: str = "gemini-1.5-flash") -> str | None:
    """
    Calls Gemini and asks for a JSON-only response.
    Returns the raw text response (caller should parse).
    If GEMINI_API_KEY is not set, returns None.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        sys = "You are a reliable JSON generator. Output strictly valid JSON and nothing else."
        if schema_hint:
            sys += f" Schema hint: {schema_hint}"

        def _run() -> str:
            resp = model.generate_content(
                [
                    {"role": "system", "parts": [sys]},
                    {"role": "user", "parts": [prompt]},
                ],
                generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
            )
            return resp.text

        # Hard timeout to avoid blocking the Streamlit UI indefinitely.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run)
            return fut.result(timeout=20.0)
    except concurrent.futures.TimeoutError:
        return None
    except Exception:
        # Gemini failures shouldn't break the whole app.
        return None


def safe_json_loads(text: str | None) -> Json | None:
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except Exception:
        return None

