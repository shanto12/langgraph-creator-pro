#!/usr/bin/env python3
"""
LangGraph Creator Pro
- Single-file, conversational generator for production-ready LangGraph multi-agent systems.
- Discovers available models using your API keys (OpenAI / OpenRouter).
- Analyzes files (PDF, CSV, JSON, text) and integrates insights into the design.
- Outputs a complete project with Dockerfile, tests, docs, and a runnable graph.

Usage:
  python langgraph_creator.py "I want to build a research assistant" --files "/path/a.pdf,/path/b.csv"
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import base64
import shutil
import textwrap
import argparse
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from rich import print, box
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.tree import Tree
import requests
from jinja2 import Template
import pandas as pd
from pypdf import PdfReader
from pydantic import BaseModel, Field
from ruamel.yaml import YAML

# Try to import OpenAI client
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

console = Console()
yaml = YAML()

# --------------------------
# Data Models
# --------------------------

@dataclass
class FileSummary:
    path: str
    kind: str
    bytes: int
    summary: str


class UserSpec(BaseModel):
    goal: str = Field(..., description="User high-level request/intent.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Answers to clarifying questions.")
    file_summaries: List[Dict[str, Any]] = Field(default_factory=list)


class ModelInfo(BaseModel):
    provider: str
    model_id: str
    price_prompt: Optional[float] = None
    price_completion: Optional[float] = None
    context: Optional[int] = None
    tags: List[str] = Field(default_factory=list)


class SelectedModels(BaseModel):
    reasoner: str
    general: str
    vision: Optional[str] = None
    provider: str = "openai_or_openrouter"


class PlanSpec(BaseModel):
    project_name: str
    project_slug: str
    description: str
    agents: List[Dict[str, Any]]
    state_schema: Dict[str, str]
    steps: List[str]
    dependencies: List[str]
    tasks_by_agent: Dict[str, List[str]] = Field(default_factory=dict)
    notes: Optional[str] = None


# --------------------------
# Helpers
# --------------------------

def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\- ]+", "", name).strip().lower().replace(" ", "-")
    s = re.sub(r"-{2,}", "-", s)
    return s or "project"


def ensure_dir(p: str | pathlib.Path) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def read_text_file(path: str, limit: int = 20000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if len(content) > limit:
                return content[:limit] + "\n...[truncated]..."
            return content
    except Exception as e:
        return f"[Error reading text file: {e}]"


def summarize_pdf(path: str, limit_pages: int = 10, char_limit: int = 20000) -> str:
    try:
        reader = PdfReader(path)
        texts = []
        for i, page in enumerate(reader.pages[:limit_pages]):
            txt = page.extract_text() or ""
            texts.append(f"[Page {i+1}]\n{txt}")
        content = "\n\n".join(texts)
        if len(content) > char_limit:
            content = content[:char_limit] + "\n...[truncated]..."
        return content
    except Exception as e:
        return f"[Error reading PDF: {e}]"


def summarize_csv(path: str, rows: int = 10) -> str:
    try:
        df = pd.read_csv(path)
        head = df.head(rows).to_markdown(index=False)
        desc = df.describe(include="all").to_markdown()
        return f"CSV Head (first {rows} rows):\n{head}\n\nCSV Describe:\n{desc}"
    except Exception as e:
        return f"[Error reading CSV: {e}]"


def summarize_xlsx(path: str, rows: int = 10) -> str:
    try:
        df = pd.read_excel(path)
        head = df.head(rows).to_markdown(index=False)
        desc = df.describe(include="all").to_markdown()
        return f"XLSX Head (first {rows} rows):\n{head}\n\nXLSX Describe:\n{desc}"
    except Exception as e:
        return f"[Error reading XLSX: {e}]"


def summarize_json(path: str, char_limit: int = 20000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        pretty = json.dumps(data, indent=2)
        if len(pretty) > char_limit:
            pretty = pretty[:char_limit] + "\n...[truncated]..."
        return f"JSON Structure:\n{pretty}"
    except Exception as e:
        return f"[Error reading JSON: {e}]"


# --------------------------
# LLM Router & Model Discovery
# --------------------------

class LLMRouter:
    """
    Supports OpenAI or OpenRouter automatically.
    - Discovers models via provider listing.
    - Provides a simple chat() interface for prompts.
    """
    def __init__(self) -> None:
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.provider = self._detect_provider()

        self.client = None
        if self.provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package is required. Please install requirements.txt")
            self.client = OpenAI(api_key=self.openai_key)
        elif self.provider == "openrouter":
            # Use OpenAI client with different base_url for OpenRouter (OpenAI-compatible)
            if OpenAI is None:
                raise RuntimeError("openai package is required. Please install requirements.txt")
            self.client = OpenAI(api_key=self.openrouter_key, base_url=f"{self.openrouter_base}")
        else:
            raise RuntimeError("No supported provider keys found. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")

    def _detect_provider(self) -> str:
        if self.openrouter_key:
            return "openrouter"
        if self.openai_key:
            return "openai"
        return "none"

    def discover_models(self) -> List[ModelInfo]:
        if self.provider == "openai":
            try:
                result = self.client.models.list()
                models = []
                # We do not get pricing from OpenAI listing; capture ids
                for m in result.data:
                    mid = getattr(m, "id", None) or getattr(m, "model", None) or ""
                    if mid:
                        tags = []
                        if "o3" in mid or "reasoning" in mid:
                            tags.append("reasoning")
                        if "gpt-4o" in mid or "4o" in mid:
                            tags.append("vision")
                        models.append(ModelInfo(provider="openai", model_id=mid, tags=tags))
                return models
            except Exception as e:
                console.print(f"[yellow]OpenAI model discovery failed: {e}[/yellow]")
                return []
        elif self.provider == "openrouter":
            try:
                headers = {
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json"
                }
                resp = requests.get(f"{self.openrouter_base}/models", headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                models = []
                for m in data.get("data", []):
                    mid = m.get("id")
                    pricing = m.get("pricing", {})
                    ctx = m.get("context_length")
                    tags = m.get("tags") or []
                    models.append(ModelInfo(
                        provider="openrouter",
                        model_id=mid,
                        price_prompt=_try_float(pricing.get("prompt")) if pricing else None,
                        price_completion=_try_float(pricing.get("completion")) if pricing else None,
                        context=ctx,
                        tags=tags
                    ))
                return models
            except Exception as e:
                console.print(f"[yellow]OpenRouter model discovery failed: {e}[/yellow]")
                return []
        return []

    def pick_models(self, models: List[ModelInfo]) -> SelectedModels:
        """
        Heuristic:
        - Reasoner: prefer models tagged 'reasoning' or names containing 'o3', 'r1', 'opus', 'deepthink', etc.
        - Vision: prefer models tagged 'vision' or names containing '4o', 'vision'
        - General: cheapest with good context, fallback to 'gpt-4o-mini'/'gpt-4o' style
        """
        if not models:
            # conservative defaults likely to exist
            if self.provider == "openai":
                return SelectedModels(reasoner="gpt-4o", general="gpt-4o-mini", vision="gpt-4o", provider="openai")
            else:
                return SelectedModels(reasoner="openrouter/auto", general="openrouter/auto", vision=None, provider="openrouter")

        def score_reasoner(m: ModelInfo) -> int:
            name = m.model_id.lower()
            score = 0
            if "reason" in name or "o3" in name or "r1" in name or "opus" in name or "deepseek" in name:
                score += 10
            if "4o" in name:
                score += 2
            return score

        def score_vision(m: ModelInfo) -> int:
            name = m.model_id.lower()
            score = 0
            if "vision" in name or "4o" in name:
                score += 10
            if "image" in name:
                score += 5
            return score

        def score_general(m: ModelInfo) -> float:
            # Prefer lower price if available
            price = (m.price_prompt or 0.0) + (m.price_completion or 0.0)
            if price > 0:
                return 1.0 / price
            # fallback preference for known names
            name = m.model_id.lower()
            if "mini" in name or "small" in name:
                return 2.0
            if "4o" in name or "sonnet" in name or "haiku" in name:
                return 1.5
            return 1.0

        reasoner = max(models, key=score_reasoner).model_id
        vision_cands = [m for m in models if score_vision(m) > 0]
        vision = max(vision_cands, key=score_vision).model_id if vision_cands else None
        general = max(models, key=score_general).model_id

        return SelectedModels(reasoner=reasoner, general=general, vision=vision, provider=self.provider)

    def chat(self, messages: List[Dict[str, Any]], model: str, temperature: float = 0.2, max_tokens: int = 2000) -> str:
        """
        Uses OpenAI-compatible Chat Completions endpoint through openai client.
        """
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"LLM chat error: {e}") from e


def _try_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# --------------------------
# Conversational Orchestrator
# --------------------------

SYSTEM_PLANNER = """You are an expert AI systems architect specializing in LangGraph multi-agent systems.
- Conduct a brief, focused discovery to clarify ambiguous goals.
- Incorporate provided file summaries.
- Design a practical, production-ready multi-agent architecture.
- Choose cost-effective model roles (reasoning vs general).
- Output a JSON plan ONLY, following the schema strictly.

Return JSON with keys:
- project_name (string)
- project_slug (kebab-case)
- description (string)
- agents (list of {name, role, responsibilities})
- state_schema (dict of state_field_name -> shortTypeLike 'str'|'list[str]'|'dict'|'int')
- steps (list of step names in execution order)
- dependencies (list of python packages for the GENERATED project)
- tasks_by_agent (dict agentName -> list of task bullet points)
- notes (optional)
"""

FOLLOWUP_QUESTION_PROMPT = """You are an AI product manager. Generate up to 5 short clarifying questions (JSON list of strings) based on the user's goal and any file summaries. Focus on purpose, data, constraints, outputs, and budget/performance needs.
Return JSON ONLY as an array of strings.
"""


def ask_followups(router: LLMRouter, models: SelectedModels, spec: UserSpec) -> List[str]:
    msgs = [
        {"role": "system", "content": "You generate clarifying questions only as JSON."},
        {"role": "user", "content": json.dumps({
            "goal": spec.goal,
            "file_summaries": spec.file_summaries,
            "instruction": FOLLOWUP_QUESTION_PROMPT
        })}
    ]
    raw = router.chat(msgs, model=models.general, temperature=0.2, max_tokens=700)
    try:
        qs = json.loads(raw)
        if isinstance(qs, list):
            return [str(q) for q in qs][:5]
    except Exception:
        pass
    # fallback
    return [
        "Who are the primary users?",
        "What data sources or files should the system use?",
        "What are the desired outputs and format?",
        "Any constraints (budget, latency, compliance)?",
        "How will success be measured?"
    ]


def generate_plan(router: LLMRouter, models: SelectedModels, spec: UserSpec) -> PlanSpec:
    msgs = [
        {"role": "system", "content": SYSTEM_PLANNER},
        {"role": "user", "content": json.dumps(spec.model_dump(), indent=2)}
    ]
    raw = router.chat(msgs, model=models.reasoner, temperature=0.2, max_tokens=2500)
    # Expecting JSON
    try:
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Failed to parse plan JSON from LLM: {e}\nRaw:\n{raw}") from e

    # Minimal validation and defaulting
    if "project_slug" not in data:
        data["project_slug"] = slugify(data.get("project_name", "project"))
    if "dependencies" not in data or not isinstance(data["dependencies"], list):
        data["dependencies"] = []
    if "agents" not in data or not isinstance(data["agents"], list):
        data["agents"] = [{"name": "Planner", "role": "Planner", "responsibilities": ["Plan the work"]}]
    if "state_schema" not in data or not isinstance(data["state_schema"], dict):
        data["state_schema"] = {"input": "str", "final_answer": "str"}
    if "steps" not in data or not isinstance(data["steps"], list):
        data["steps"] = ["plan", "work", "synthesize"]

    return PlanSpec(**data)


# --------------------------
# File Analysis
# --------------------------

def analyze_files(file_paths: List[str]) -> List[FileSummary]:
    results: List[FileSummary] = []
    for p in file_paths:
        p = p.strip()
        if not p:
            continue
        try:
            st = os.stat(p)
        except Exception as e:
            results.append(FileSummary(path=p, kind="missing", bytes=0, summary=f"[Missing] {e}"))
            continue

        ext = pathlib.Path(p).suffix.lower()
        summary = ""
        kind = "unknown"

        if ext in [".txt", ".md", ".log"]:
            summary = read_text_file(p)
            kind = "text"
        elif ext in [".pdf"]:
            summary = summarize_pdf(p)
            kind = "pdf"
        elif ext in [".csv"]:
            summary = summarize_csv(p)
            kind = "csv"
        elif ext in [".xlsx", ".xls"]:
            summary = summarize_xlsx(p)
            kind = "xlsx"
        elif ext in [".json"]:
            summary = summarize_json(p)
            kind = "json"
        else:
            kind = f"binary({ext})"
            summary = f"[Binary or unsupported file type: {ext}. Will reference metadata only.]"

        results.append(FileSummary(path=p, kind=kind, bytes=st.st_size, summary=summary))
    return results


# --------------------------
# Project Generation Templates
# --------------------------

TEMPLATE_REQUIREMENTS = """\
langgraph>=0.2.35
langchain>=0.2.11
langchain-openai>=0.1.22
duckduckgo-search>=5.3.0
python-dotenv>=1.0.1
requests>=2.32.3
pydantic>=2.9.2
"""

TEMPLATE_ENV_EXAMPLE = """\
# One of the following is required:
OPENAI_API_KEY=
# or
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
"""

TEMPLATE_DOCKERFILE = """\
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "src/run.py", "Hello from Docker!"]
"""

TEMPLATE_MAKEFILE = """\
.PHONY: setup test run fmt

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
\tpytest -q

run:
\tpython src/run.py "Test the app end-to-end"

fmt:
\truff check --fix || true
"""

TEMPLATE_README = """\
# {{project_name}}

{{description}}

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Add your keys
python src/run.py "Hello, what can you do?"
```

## Project Structure
```
.
├── Dockerfile
├── Makefile
├── README.md
├── requirements.txt
├── .env.example
├── src
│   ├── llm_router.py
│   ├── run.py
│   ├── graph.py
│   ├── agents
│   │   ├── planner.py
│   │   ├── researcher.py
│   │   ├── coder.py
│   │   └── synthesizer.py
│   └── tools
│       ├── files.py
│       ├── search.py
│       └── web.py
└── tests
    └── test_smoke.py
```

## Environment
- OPENAI_API_KEY or OPENROUTER_API_KEY must be set (.env)

## Run
```bash
python src/run.py "Write an outline for the topic: renewable energy trends"
```

## Notes
- LLM routing and model selection are configured in `src/llm_router.py`.
- Agents are modular; add new ones or tools under `src/agents` and `src/tools`.
"""

TEMPLATE_LLM_ROUTER = """\
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# OpenAI-compatible client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

load_dotenv()

class LLMRouter:
    def __init__(self, preferred_reasoner: Optional[str] = None, preferred_general: Optional[str] = None, preferred_vision: Optional[str] = None):
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.provider = self._detect_provider()
        if self.provider == "none":
            raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY in environment.")
        self.client = self._make_client()
        self.reasoner = preferred_reasoner
        self.general = preferred_general
        self.vision = preferred_vision

    def _detect_provider(self) -> str:
        if self.openrouter_key:
            return "openrouter"
        if self.openai_key:
            return "openai"
        return "none"

    def _make_client(self):
        if OpenAI is None:
            raise RuntimeError("openai package required")
        if self.provider == "openai":
            return OpenAI(api_key=self.openai_key)
        else:
            return OpenAI(api_key=self.openrouter_key, base_url=f"{self.openrouter_base}")

    def chat(self, messages: List[Dict[str, Any]], model: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        model_to_use = model or self.general or self.reasoner
        if not model_to_use:
            model_to_use = "gpt-4o" if self.provider == "openai" else "openrouter/auto"
        resp = self.client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
"""

TEMPLATE_GRAPH = """\
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from .llm_router import LLMRouter
from .agents.planner import planner_node
from .agents.researcher import researcher_node
from .agents.coder import coder_node
from .agents.synthesizer import synthesizer_node

class AppState(TypedDict):
    input: str
    plan: List[str]
    research_notes: str
    code_snippets: List[str]
    final_answer: str

def build_graph(router: LLMRouter):
    graph = StateGraph(AppState)
    graph.add_node("plan", lambda s: planner_node(router, s))
    graph.add_node("research", lambda s: researcher_node(router, s))
    graph.add_node("code", lambda s: coder_node(router, s))
    graph.add_node("synthesize", lambda s: synthesizer_node(router, s))

    graph.add_edge("plan", "research")
    graph.add_edge("research", "code")
    graph.add_edge("code", "synthesize")
    graph.add_edge("synthesize", END)

    graph.set_entry_point("plan")
    return graph.compile()
"""

TEMPLATE_RUN = """\
import sys
import os
from dotenv import load_dotenv
from .llm_router import LLMRouter
from .graph import build_graph

load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/run.py \"Your task prompt\"")
        sys.exit(1)
    user_input = sys.argv[1]
    router = LLMRouter()
    app = build_graph(router)
    state = {
        "input": user_input,
        "plan": [],
        "research_notes": "",
        "code_snippets": [],
        "final_answer": ""
    }
    for event in app.stream(state):
        for name, update in event.items():
            print(f"[{name}] {update}")
    print("\n=== Final Answer ===")
    print(app.get_state()["final_answer"])

if __name__ == "__main__":
    main()
"""

TEMPLATE_AGENT_PLANNER = """\
from typing import List, Dict, Any
def planner_node(router, state):
    messages = [
        {"role": "system", "content": "You are a senior planner. Break the task into 3-6 concrete steps."},
        {"role": "user", "content": state["input"]}
    ]
    plan_text = router.chat(messages, temperature=0.2, max_tokens=600)
    steps = [s.strip("- ").strip() for s in plan_text.split("\n") if s.strip()]
    if not steps:
        steps = ["Understand the problem", "Research relevant info", "Draft answer", "Refine"]
    state["plan"] = steps
    return state
"""

TEMPLATE_AGENT_RESEARCHER = """\
from typing import List, Dict, Any
from ..tools.search import ddg_search
from ..tools.web import fetch_and_summarize

def researcher_node(router, state):
    notes = []
    for step in state.get("plan", [])[:5]:
        q = f"{step} background information"
        results = ddg_search(q, max_results=3)
        for r in results:
            url = r.get("href") or r.get("url")
            if not url:
                continue
            summary = fetch_and_summarize(url)
            notes.append(f"Source: {url}\n{summary}\n")
    state["research_notes"] = "\n\n".join(notes)[:15000]
    return state
"""

TEMPLATE_AGENT_CODER = """\
def coder_node(router, state):
    # If user wants code, generate snippets; otherwise create structured bullet points.
    messages = [
        {"role": "system", "content": "You write minimal, correct code snippets or structured solutions as requested."},
        {"role": "user", "content": f"Task: {state['input']}\nResearch:\n{state.get('research_notes','')[:4000]}\nCreate concise code snippets if relevant."}
    ]
    content = router.chat(messages, temperature=0.3, max_tokens=900)
    state["code_snippets"] = [content]
    return state
"""

TEMPLATE_AGENT_SYNTHESIZER = """\
def synthesizer_node(router, state):
    messages = [
        {"role": "system", "content": "Synthesize a final answer. Be concise, actionable, and include references to key sources when possible."},
        {"role": "user", "content": f"Task: {state['input']}\nPlan: {state.get('plan', [])}\nNotes:\n{state.get('research_notes','')[:4000]}\nCode:\n{state.get('code_snippets', [])}"}
    ]
    answer = router.chat(messages, temperature=0.2, max_tokens=900)
    state["final_answer"] = answer
    return state
"""

TEMPLATE_TOOL_FILES = """\
import os

def read_text(path: str, limit: int = 20000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            c = f.read()
            return c[:limit] + ("\n...[truncated]..." if len(c) > limit else "")
    except Exception as e:
        return f"[Error reading file {path}: {e}]"
"""

TEMPLATE_TOOL_SEARCH = """\
from duckduckgo_search import DDGS

def ddg_search(query: str, max_results: int = 3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            # Normalize
            normalized = []
            for r in results:
                normalized.append({
                    "title": r.get("title"),
                    "href": r.get("href") or r.get("url"),
                    "snippet": r.get("body")
                })
            return normalized
    except Exception as e:
        return []
"""

TEMPLATE_TOOL_WEB = """\
import requests
def fetch_and_summarize(url: str) -> str:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        text = r.text
        # naive summarization: take first ~800 chars
        return text[:800] + ("..." if len(text) > 800 else "")
    except Exception as e:
        return f"[Error fetching {url}: {e}]"
"""

TEMPLATE_TEST_SMOKE = """\
def test_import():
    import src.graph as g
    import src.llm_router as r
    assert hasattr(g, "build_graph")
    assert hasattr(r, "LLMRouter")
"""


# --------------------------
# Project Generator
# --------------------------

class ProjectGenerator:
    def __init__(self, out_dir: str, plan: PlanSpec, models: SelectedModels):
        self.out_dir = os.path.join(out_dir, plan.project_slug)
        self.plan = plan
        self.models = models

    def generate(self) -> None:
        # Layout
        ensure_dir(self.out_dir)
        ensure_dir(os.path.join(self.out_dir, "src"))
        ensure_dir(os.path.join(self.out_dir, "src", "agents"))
        ensure_dir(os.path.join(self.out_dir, "src", "tools"))
        ensure_dir(os.path.join(self.out_dir, "tests"))

        # Core files
        self._write("requirements.txt", TEMPLATE_REQUIREMENTS)
        self._write(".env.example", TEMPLATE_ENV_EXAMPLE)
        self._write("Dockerfile", TEMPLATE_DOCKERFILE)
        self._write("Makefile", TEMPLATE_MAKEFILE)

        # README (rendered with plan)
        readme = Template(TEMPLATE_README).render(
            project_name=self.plan.project_name,
            description=self.plan.description
        )
        self._write("README.md", readme)

        # src
        self._write("src/llm_router.py", TEMPLATE_LLM_ROUTER)
        self._write("src/graph.py", TEMPLATE_GRAPH)
        self._write("src/run.py", TEMPLATE_RUN)

        # agents
        self._write("src/agents/planner.py", TEMPLATE_AGENT_PLANNER)
        self._write("src/agents/researcher.py", TEMPLATE_AGENT_RESEARCHER)
        self._write("src/agents/coder.py", TEMPLATE_AGENT_CODER)
        self._write("src/agents/synthesizer.py", TEMPLATE_AGENT_SYNTHESIZER)

        # tools
        self._write("src/tools/files.py", TEMPLATE_TOOL_FILES)
        self._write("src/tools/search.py", TEMPLATE_TOOL_SEARCH)
        self._write("src/tools/web.py", TEMPLATE_TOOL_WEB)

        # tests
        self._write("tests/test_smoke.py", TEMPLATE_TEST_SMOKE)

        # pin chosen models into a simple config file for the app (optional)
        model_config = {
            "provider": self.models.provider,
            "reasoner": self.models.reasoner,
            "general": self.models.general,
            "vision": self.models.vision
        }
        self._write("model_selection.json", json.dumps(model_config, indent=2))

    def _write(self, rel: str, content: str) -> None:
        path = os.path.join(self.out_dir, rel)
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


# --------------------------
# Creator CLI
# --------------------------

def save_history(entry: Dict[str, Any], path: str = "creator_history.json") -> None:
    data = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="LangGraph Creator Pro")
    parser.add_argument("goal", type=str, help="High-level goal (quoted)")
    parser.add_argument("--files", type=str, default="", help="Comma-separated file paths")
    parser.add_argument("--out-dir", type=str, default="out", help="Output directory")
    parser.add_argument("--yes", action="store_true", help="Non-interactive; skip follow-ups")
    args = parser.parse_args()

    console.print(Panel.fit("[bold cyan]LangGraph Creator Pro[/bold cyan]\nBuild intelligent multi-agent systems with zero code.", box=box.ROUNDED))

    # LLM routing
    try:
        router = LLMRouter()
    except Exception as e:
        console.print(f"[red]Error initializing LLM provider: {e}[/red]")
        sys.exit(1)

    # Discover models
    console.print("[bold]Discovering available models...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        t = progress.add_task("Querying providers", total=None)
        discovered = router.discover_models()
        selected = router.pick_models(discovered)
        progress.remove_task(t)

    table = Table(title="Selected Models", box=box.MINIMAL)
    table.add_column("Role")
    table.add_column("Model ID")
    table.add_column("Provider")
    table.add_row("Reasoner", selected.reasoner, selected.provider)
    table.add_row("General", selected.general, selected.provider)
    table.add_row("Vision", selected.vision or "-", selected.provider)
    console.print(table)

    # Files
    files = [p.strip() for p in args.files.split(",")] if args.files else []
    files = [p for p in files if p]
    file_summaries: List[FileSummary] = []
    if files:
        console.print("[bold]Analyzing files...[/bold]")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            progress.add_task("Reading files", total=None)
            file_summaries = analyze_files(files)

        ft = Table(title="File Summaries", box=box.SIMPLE)
        ft.add_column("Path", overflow="fold")
        ft.add_column("Kind")
        ft.add_column("Bytes", justify="right")
        for fs in file_summaries:
            ft.add_row(fs.path, fs.kind, str(fs.bytes))
        console.print(ft)

    # Compose user spec
    spec = UserSpec(
        goal=args.goal,
        file_summaries=[fs.__dict__ for fs in file_summaries]
    )

    # Follow-ups
    if not args.yes:
        console.print("[bold]A few quick questions to tailor the system...[/bold]")
        questions = ask_followups(router, selected, spec)
        answers: Dict[str, Any] = {}
        for q in questions:
            ans = Prompt.ask(f"[green]?[/green] {q}", default="")
            if ans.strip():
                answers[q] = ans.strip()
        spec.details = answers

    # Generate plan
    console.print("[bold]Designing your multi-agent architecture...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task("Reasoning and planning", total=None)
        plan = generate_plan(router, selected, spec)

    # Preview
    console.print(Panel.fit(f"[bold]Project:[/bold] {plan.project_name}\n[bold]Slug:[/bold] {plan.project_slug}\n[bold]Description:[/bold] {plan.description}", title="Plan Preview"))

    at = Table(title="Agents", box=box.SIMPLE)
    at.add_column("Name")
    at.add_column("Role")
    at.add_column("Responsibilities", overflow="fold")
    for a in plan.agents:
        at.add_row(a.get("name", "?"), a.get("role", "?"), ", ".join(a.get("responsibilities", [])))
    console.print(at)

    st = Table(title="State Schema", box=box.SIMPLE)
    st.add_column("Field")
    st.add_column("Type")
    for k, v in plan.state_schema.items():
        st.add_row(k, v)
    console.print(st)

    steps_table = Table(title="Execution Steps", box=box.SIMPLE)
    steps_table.add_column("#", justify="right")
    steps_table.add_column("Step")
    for i, s in enumerate(plan.steps, 1):
        steps_table.add_row(str(i), s)
    console.print(steps_table)

    if not args.yes:
        if not Confirm.ask("Proceed to generate the project now?", default=True):
            console.print("[yellow]Cancelled.[/yellow]")
            sys.exit(0)

    # Generate project
    out_dir = args.out_dir
    ensure_dir(out_dir)
    generator = ProjectGenerator(out_dir=out_dir, plan=plan, models=selected)

    console.print(f"[bold]Generating project in:[/bold] {os.path.join(out_dir, plan.project_slug)}")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task("Writing files", total=None)
        generator.generate()

    console.print("[green]✅ Project generated successfully.[/green]")
    console.print(f"Next steps:\n  1) cd {os.path.join(out_dir, plan.project_slug)}\n  2) python -m venv .venv && source .venv/bin/activate\n  3) pip install -r requirements.txt\n  4) cp .env.example .env  # add your API keys\n  5) python src/run.py \"Your prompt here\"")

    # Save history (self-improving)
    save_history({
        "ts": time.time(),
        "goal": args.goal,
        "details": spec.details,
        "files": files,
        "chosen_models": selected.model_dump(),
        "plan": plan.model_dump(),
    })


if __name__ == "__main__":
    main()
