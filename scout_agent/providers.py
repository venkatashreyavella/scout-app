from __future__ import annotations

from .models import FinalTurn, ModelTurn, ResearchResult, Settings, ToolCall
from .tools import TOOL_DEFINITIONS, safe_https

class OpenAIModel:
    def __init__(self, settings: Settings):
        self.settings = settings

    def turn(self, messages: list[dict], timeout: float) -> ModelTurn:
        from openai import OpenAI
        with OpenAI(api_key=self.settings.api_key, timeout=timeout, max_retries=0) as client:
            response = client.chat.completions.create(
                model=self.settings.model, messages=messages, tools=TOOL_DEFINITIONS,
                parallel_tool_calls=False, temperature=0, max_completion_tokens=1000)
        choice = response.choices[0]
        if choice.finish_reason not in ("stop", "tool_calls") or choice.message.refusal:
            raise RuntimeError("The model did not complete the research turn.")
        calls = [ToolCall(c.id, c.function.name, c.function.arguments)
                 for c in choice.message.tool_calls or [] if c.type == "function"]
        return ModelTurn(choice.message.content or "", calls,
                         response.usage.prompt_tokens if response.usage else 0,
                         response.usage.completion_tokens if response.usage else 0)

    def finalize(self, messages: list[dict], timeout: float) -> FinalTurn:
        from openai import OpenAI
        instruction = """Return the structured ResearchResult. Every factual finding must cite retrieved source IDs.
Copy only expressions and values actually returned by calculate. Summarize only supported findings.
Never claim a suggested action was executed. If facts are missing or the task is outside scope,
use a non-complete status and low confidence. State assumptions and uncertainty explicitly."""
        with OpenAI(api_key=self.settings.api_key, timeout=timeout, max_retries=0) as client:
            response = client.chat.completions.parse(
                model=self.settings.model, messages=[*messages, {"role": "system", "content": instruction}],
                response_format=ResearchResult, temperature=0, max_completion_tokens=2400)
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("No structured response returned.")
        return FinalTurn(parsed, response.usage.prompt_tokens if response.usage else 0,
                         response.usage.completion_tokens if response.usage else 0)

class TavilySearch:
    def __init__(self, settings: Settings):
        self.settings = settings

    def __call__(self, query: str, timeout: float) -> list[dict]:
        import httpx
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.post("https://api.tavily.com/search",
                headers={"Authorization": "Bearer " + self.settings.tavily_key},
                json={"query": query, "search_depth": "basic", "max_results": 5,
                      "include_raw_content": False, "include_answer": False})
            response.raise_for_status()
            rows = response.json().get("results", [])
        if not isinstance(rows, list):
            raise ValueError("Unexpected search response.")
        results = []
        for row in rows[:5]:
            if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ("title", "url", "content")):
                continue
            if safe_https(row["url"]):
                results.append({"title": row["title"][:200], "url": row["url"], "excerpt": row["content"][:1800]})
        return results
