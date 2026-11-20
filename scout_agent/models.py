from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class AgentInput(StrictModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=2000)]
    mode: Literal["demo", "live"]

class Finding(StrictModel):
    claim: str = Field(max_length=1200)
    source_ids: list[str]

class Calculation(StrictModel):
    expression: str
    value: float = Field(allow_inf_nan=False)
    explanation: str

class ResearchResult(StrictModel):
    status: Literal["complete", "needs_clarification", "insufficient_evidence", "blocked", "error"]
    title: str = Field(min_length=1, max_length=140)
    summary: str = Field(min_length=1, max_length=2400)
    confidence: Literal["high", "medium", "low"]
    findings: list[Finding] = Field(max_length=8)
    calculations: list[Calculation] = Field(max_length=6)
    actions: list[str] = Field(max_length=6)
    limitations: list[str] = Field(max_length=8)

class Source(StrictModel):
    id: str
    title: str
    url: str
    excerpt: str

class TokenUsage(StrictModel):
    input: int = Field(ge=0)
    output: int = Field(ge=0)

class TraceEvent(StrictModel):
    id: str
    at: str
    kind: Literal["system", "guardrail", "model", "tool_start", "tool_end", "verification"]
    label: str
    data: dict
    duration_ms: int | None
    tokens: TokenUsage | None

class RunReport(StrictModel):
    run_id: str
    mode: Literal["demo", "live"]
    duration_ms: int
    result: ResearchResult
    sources: list[Source]
    trace: list[TraceEvent]

class SearchInput(StrictModel):
    query: str = Field(min_length=3, max_length=400)

class CalculatorInput(StrictModel):
    expression: str = Field(min_length=1, max_length=160)

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

@dataclass(frozen=True)
class ModelTurn:
    text: str
    calls: list[ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass(frozen=True)
class FinalTurn:
    result: ResearchResult | dict
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    tavily_key: str = field(default="", repr=False)
    access_token: str = field(default="", repr=False)
    model: str = "gpt-4o-mini"
