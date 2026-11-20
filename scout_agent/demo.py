from __future__ import annotations

import json
from .guardrails import fallback
from .models import FinalTurn, ModelTurn, ResearchResult, ToolCall

DEMO_QUERIES = [
    "Compare managed and self-hosted Postgres. Estimate a $25/month managed plan for one year.",
    "Research caching for a Next.js app. Calculate the latency reduction from 800 ms to 320 ms.",
    "Find technical documentation for a product called NebulaDB-DoesNotExist-9000.",
]
DEMO_SOURCES = [
    {"title": "PostgreSQL · Server administration", "url": "https://www.postgresql.org/docs/current/admin.html",
     "excerpt": "PostgreSQL server administration includes installation, configuration, maintenance, backups, and monitoring. Curated demo excerpt; not a live retrieval."},
    {"title": "Next.js · Caching", "url": "https://nextjs.org/docs/app/guides/caching",
     "excerpt": "Next.js provides caching for data and rendered output. Behavior depends on the API and framework version. Curated demo excerpt; not a live retrieval."},
]

def demo_search(query: str, timeout: float = 12) -> list[dict]:
    if query == DEMO_QUERIES[2]:
        return []
    return [dict(DEMO_SOURCES[1] if query == DEMO_QUERIES[1] else DEMO_SOURCES[0])]

class DemoModel:
    def __init__(self):
        self.step = 0

    @staticmethod
    def scenario(messages: list[dict]) -> int:
        query = next(m["content"] for m in messages if m["role"] == "user")
        return DEMO_QUERIES.index(query) if query in DEMO_QUERIES else -1

    def turn(self, messages: list[dict], timeout: float) -> ModelTurn:
        index = self.scenario(messages)
        if index < 0:
            return ModelTurn("This question is not a demo fixture.", [])
        self.step += 1
        if self.step == 1:
            return ModelTurn("", [ToolCall("demo-search", "web_search", json.dumps({"query": DEMO_QUERIES[index]}))])
        if self.step == 2 and index != 2:
            expression = "25 * 12" if index == 0 else "(800 - 320) / 800 * 100"
            return ModelTurn("", [ToolCall("demo-calculation", "calculate", json.dumps({"expression": expression}))])
        return ModelTurn("Evidence gathered.", [])

    def finalize(self, messages: list[dict], timeout: float) -> FinalTurn:
        index = self.scenario(messages)
        if index < 0:
            return FinalTurn(fallback("needs_clarification",
                "Demo mode supports the three example questions. Choose an example, or use live mode for your own technical research.",
                ["No live model or search provider was called."]))
        if index == 2:
            return FinalTurn(fallback("insufficient_evidence",
                "The demo search returned no documentation for this product. I cannot verify its features or invent a recommendation.",
                ["This is an intentionally empty fixture, not proof that a product does not exist."]))
        cost = index == 0
        return FinalTurn(ResearchResult(
            status="complete", title="Your database, with less maintenance" if cost else "Faster responses start with caching",
            summary=("A managed Postgres plan can reduce operational work. Using your assumed $25/month price, the annual base cost is $300. Compare storage, compute, and backup allowances before choosing a provider."
                     if cost else "Caching can reduce repeated work in a Next.js app. Your supplied latency values fall by 60%, from 800 ms to 320 ms. Validate the change against your actual workload."),
            confidence="medium",
            findings=[{"claim": "Operating PostgreSQL involves configuration, maintenance, backups, and monitoring." if cost else
                       "Next.js offers caching for data and rendered output; behavior depends on the API and version.", "source_ids": ["S1"]}],
            calculations=[{"expression": "25 * 12" if cost else "(800 - 320) / 800 * 100",
                           "value": 300.0 if cost else 60.0,
                           "explanation": "USD per year using your assumed price; taxes and usage extras excluded." if cost else
                           "Percent reduction relative to the supplied baseline; not an actual benchmark."}],
            actions=["Check backup retention and restore procedures.", "Price your storage and compute requirements."] if cost else
                    ["Choose a freshness policy for cached data.", "Measure p50 and p95 latency before and after changes."],
            limitations=["Demo uses a scripted model and curated source excerpts, not live research.",
                         "The price is a user-supplied assumption, not a verified quote." if cost else "The latency values are user-supplied examples."]))
