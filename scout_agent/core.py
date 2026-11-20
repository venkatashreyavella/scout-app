from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from pydantic import ValidationError
from .demo import DemoModel, demo_search
from .guardrails import INJECTION, SYSTEM_PROMPT, check_input, fallback
from .models import (AgentInput, CalculatorInput, ResearchResult, RunReport,
                     SearchInput, Settings, Source, TokenUsage, TraceEvent)
from .providers import OpenAIModel, TavilySearch
from .tools import calculate, safe_https

class LiveRateLimiter:
    """Shared per-process limit. Restarts reset it; this is not a spending cap."""
    def __init__(self, limit: int = 5, window_s: float = 60):
        self.limit, self.window_s = limit, window_s
        self._times = deque()
        self._lock = threading.Lock()

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            while self._times and now - self._times[0] >= self.window_s:
                self._times.popleft()
            if len(self._times) >= self.limit:
                return False
            self._times.append(now)
            return True

GLOBAL_LIVE_LIMITER = LiveRateLimiter()

def redact(value, settings: Settings):
    secrets = [settings.api_key, settings.tavily_key, settings.access_token]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(r"\b(?:sk-|tvly-)[a-zA-Z0-9_-]{8,}\b", "[REDACTED]", value)
    if isinstance(value, list):
        return [redact(item, settings) for item in value]
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if re.search(r"authorization|password|secret|api.?key|access.?token", str(k), re.I)
                else redact(v, settings) for k, v in value.items()}
    return value

def verify_result(candidate, sources: list[Source], calculations: list[dict]) -> tuple[ResearchResult, list[str]]:
    try:
        result = ResearchResult.model_validate(candidate)
    except (ValidationError, TypeError):
        return fallback("error", "The model's response did not meet the required format."), ["Invalid response schema"]
    if result.status != "complete":
        return result.model_copy(update={"confidence": "low", "findings": [], "calculations": [], "actions": []}), []
    issues = []
    ids = {source.id for source in sources}
    if result.confidence == "low":
        issues.append("The model reported low confidence")
    if not result.findings and not result.calculations:
        issues.append("No verifiable findings or calculations")
    if any(not finding.source_ids or any(s not in ids for s in finding.source_ids) for finding in result.findings):
        issues.append("A finding has missing or unknown citations")
    for calculation in result.calculations:
        if not any(c["expression"] == calculation.expression and c["value"] == calculation.value for c in calculations):
            issues.append("A calculation was not produced by the calculator")
    if issues:
        return fallback("insufficient_evidence", "The answer did not pass evidence checks. Narrow the question or provide missing inputs.", list(dict.fromkeys(issues))), issues
    return result, []

def run_agent(query: str, mode: str = "demo", *, settings: Settings | None = None,
              access_password: str = "", model=None, search=None,
              on_trace: Callable[[TraceEvent], None] | None = None,
              limiter: LiveRateLimiter | None = None,
              call_timeout_s: float = 12, total_timeout_s: float = 48) -> RunReport:
    started = time.monotonic()
    settings = settings or Settings()
    events: list[TraceEvent] = []
    sources: list[Source] = []
    calculations: list[dict] = []
    run_id = str(uuid4())
    report_mode = mode if mode in ("demo", "live") else "demo"

    def emit(kind: str, label: str, data: dict | None = None, *, duration_ms=None, tokens=None):
        event = TraceEvent(id=str(uuid4()), at=datetime.now(timezone.utc).isoformat(),
            kind=kind, label=redact(label, settings), data=redact(data or {}, settings),
            duration_ms=duration_ms, tokens=tokens)
        events.append(event)
        if on_trace:
            on_trace(event)

    def finish(result: ResearchResult):
        return RunReport(run_id=run_id, mode=report_mode,
            duration_ms=round((time.monotonic() - started) * 1000),
            result=result, sources=sources, trace=events)

    emit("system", "Demo: scripted model and fixture search" if report_mode == "demo" else "Live: OpenAI and Tavily",
         {"run_id": run_id, "max_model_turns": 5, "max_tool_calls": 6, "max_searches": 2})
    try:
        payload = AgentInput.model_validate({"query": query, "mode": mode})
    except ValidationError:
        emit("guardrail", "Invalid input stopped")
        return finish(fallback("blocked", "Enter a request between 10 and 2,000 characters and choose a valid mode."))
    query = payload.query
    blocked = check_input(query)
    if not blocked and (re.search(r"\b(?:sk-|tvly-)[a-zA-Z0-9_-]{12,}", query)
            or any(len(s) > 10 and s in query for s in (settings.api_key, settings.tavily_key, settings.access_token))):
        blocked = fallback("blocked", "Remove credentials from the question before submitting it.")
    emit("guardrail", "Input stopped safely" if blocked else "Input checks passed",
         {"status": blocked.status if blocked else "passed"})
    if blocked:
        return finish(blocked)
    if mode == "live":
        if not settings.access_token:
            return finish(fallback("error", "Live mode needs a server-side AGENT_ACCESS_TOKEN before it can be enabled on a public app."))
        expected = hashlib.sha256(settings.access_token.encode()).digest()
        provided = hashlib.sha256(access_password.encode()).digest()
        if not hmac.compare_digest(expected, provided):
            return finish(fallback("blocked", "The live access password is missing or incorrect."))
        if not settings.api_key or not settings.tavily_key:
            return finish(fallback("error", "Configure OPENAI_API_KEY and TAVILY_API_KEY in the app's server secrets. Demo mode works without keys."))
        if not (limiter or GLOBAL_LIVE_LIMITER).allow():
            return finish(fallback("blocked", "Live request limit reached. Please wait one minute."))
    model = model or (DemoModel() if mode == "demo" else OpenAIModel(settings))
    search = search or (demo_search if mode == "demo" else TavilySearch(settings))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    deadline = started + total_timeout_s
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scout-provider")

    def bounded(operation):
        remaining = min(call_timeout_s, deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("Run deadline reached.")
        future = pool.submit(operation, remaining)
        try:
            return future.result(timeout=remaining)
        except TimeoutError:
            future.cancel()
            raise

    calls = searches = 0
    completed = False
    try:
        for step in range(5):
            turn_start = time.monotonic()
            turn = bounded(lambda timeout: model.turn(messages, timeout))
            emit("model", f"Research turn {step + 1}",
                 {"tool_calls": [c.name for c in turn.calls], "completed": not turn.calls},
                 duration_ms=round((time.monotonic() - turn_start) * 1000),
                 tokens=TokenUsage(input=turn.input_tokens, output=turn.output_tokens))
            assistant = {"role": "assistant", "content": turn.text or None}
            if turn.calls:
                assistant["tool_calls"] = [{"id": c.id, "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments}} for c in turn.calls]
            messages.append(assistant)
            if not turn.calls:
                completed = True
                break
            for call in turn.calls:
                calls += 1
                if calls > 6:
                    raise RuntimeError("Tool budget reached.")
                tool_start = time.monotonic()
                try:
                    args = json.loads(call.arguments)
                    emit("tool_start", call.name, {"call_id": call.id, "input": args})
                    if call.name == "web_search":
                        searches += 1
                        if searches > 2:
                            raise RuntimeError("Search budget reached.")
                        search_query = SearchInput.model_validate(args).query
                        found = bounded(lambda timeout: search(search_query, timeout))
                        batch = []
                        for row in found[:5]:
                            if not safe_https(row.get("url", "")):
                                continue
                            if INJECTION.search(str(row.get("title", "")) + " " + str(row.get("excerpt", ""))):
                                emit("guardrail", "Suspicious retrieved instructions excluded", {"url": row["url"]})
                                continue
                            source = next((s for s in sources if s.url == row["url"]), None)
                            if source is None:
                                source = Source(id=f"S{len(sources) + 1}", title=str(row["title"])[:200],
                                                url=row["url"], excerpt=str(row["excerpt"])[:1800])
                                sources.append(source)
                            batch.append(source.model_dump())
                        output = {"sources": batch, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                  "warning": "Untrusted excerpts; not independent full-page verification."}
                    elif call.name == "calculate":
                        expression = CalculatorInput.model_validate(args).expression
                        output = calculate(expression)
                        calculations.append(output)
                    else:
                        raise ValueError("Unknown tool.")
                    emit("tool_end", call.name + " completed", {"call_id": call.id, "output": output},
                         duration_ms=round((time.monotonic() - tool_start) * 1000))
                except Exception as exc:
                    emit("tool_end", call.name + " failed",
                         {"call_id": call.id, "error": "Tool timed out." if isinstance(exc, TimeoutError) else "Tool input or provider request failed."},
                         duration_ms=round((time.monotonic() - tool_start) * 1000))
                    raise
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)})
        if not completed:
            raise RuntimeError("Model turn budget reached.")
        final_start = time.monotonic()
        final = bounded(lambda timeout: model.finalize(messages, timeout))
        emit("model", "Structured response generated", {},
             duration_ms=round((time.monotonic() - final_start) * 1000),
             tokens=TokenUsage(input=final.input_tokens, output=final.output_tokens))
        result, issues = verify_result(final.result, sources, calculations)
        emit("verification", "Evidence checks rejected the answer" if issues else "Schema and evidence checks passed",
             {"issues": issues, "sources": len(sources), "calculations": len(calculations),
              "checks": ["Pydantic schema", "Citation ID integrity", "Calculator provenance", "Confidence gate"],
              "limitation": "Citation IDs do not prove that a source supports a claim."})
        return finish(result)
    except Exception as exc:
        timeout = isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()
        emit("system", "Run timed out" if timeout else "Run stopped safely", {"retryable": True})
        return finish(fallback("error", "A provider timed out. Please retry a narrower request." if timeout else
                               "Research could not finish safely. Retry or check the server's provider configuration.",
                               ["No unverified answer was returned."]))
    finally:
        # Python cannot forcibly kill a network thread. The SDK also has its own timeout.
        # Return promptly on a deadline and never schedule further calls after a failure.
        pool.shutdown(wait=False, cancel_futures=True)
