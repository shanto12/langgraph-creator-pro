#!/usr/bin/env python3
"""
LangGraph Creator Pro (Robust + Multi-Provider + Iterative Clarifications)

Key improvements:
- Robust JSON outputs: JSON mode when supported + LLM-based repair + fallback extract
- Iterative clarifications: ask -> assess completeness -> detect conflicts -> loop
- Multi-provider live discovery (OpenRouter, OpenAI, Anthropic, Google, Groq, Mistral, xAI)
- Capability-aware ranking by role (reasoner/vision/general) with vendor trust and cost
- Interactive model override accepts yes/no/y/n
- Filters out embeddings/audio/search/tts-only models, avoiding non-chat picks

Usage examples:
  python langgraph_creator.py "I want to build a SOC incident investigator and actioner, complete to end to end where input would be a json array of incidents"
  python langgraph_creator.py "..." --files "/path/a.pdf,/path/b.csv" --quality best
  python langgraph_creator.py "..." --yes --auto-models
  python langgraph_creator.py "..." --max-clarify-rounds 4
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import shutil
import argparse
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from rich import print, box
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from jinja2 import Template
import requests
import pandas as pd
from pypdf import PdfReader
from pydantic import BaseModel, Field

# OpenAI client (also used for OpenRouter/Groq/xAI via base_url)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

console = Console()

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
    goal: str
    details: Dict[str, Any] = Field(default_factory=dict)
    file_summaries: List[Dict[str, Any]] = Field(default_factory=list)

class ModelInfo(BaseModel):
    provider: str              # 'openai' | 'openrouter' | 'anthropic' | 'google' | 'groq' | 'mistral' | 'xai'
    model_id: str              # ID to call on that provider
    vendor: Optional[str] = None
    price_prompt: Optional[float] = None
    price_completion: Optional[float] = None
    context: Optional[int] = None
    tags: List[str] = Field(default_factory=list)

class SelectedModels(BaseModel):
    provider: str
    reasoner: str
    general: str
    vision: Optional[str] = None

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

TRUSTED_VENDORS = {
    "openai", "anthropic", "google", "mistral", "meta-llama", "qwen", "groq", "x-ai", "cohere", "perplexity"
}
# Exclude non-chat/specialized models; also filter noisy "search" previews
EXCLUDE_KEYWORDS = ["embedding", "embed", "tts", "whisper", "speech", "audio", "asr", "voice", "search"]

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
            return content if len(content) <= limit else content[:limit] + "\n...[truncated]..."
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
        return content if len(content) <= char_limit else content[:char_limit] + "\n...[truncated]..."
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
        return pretty if len(pretty) <= char_limit else pretty[:char_limit] + "\n...[truncated]..."
    except Exception as e:
        return f"[Error reading JSON: {e}]"

def _try_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _vendor_from_id(model_id: str) -> Optional[str]:
    if "/" in model_id:
        return model_id.split("/")[0]
    lower = model_id.lower()
    for v in TRUSTED_VENDORS:
        if v in lower:
            return v
    return None

def _has_excluded_keyword(model_id: str) -> bool:
    m = model_id.lower()
    return any(k in m for k in EXCLUDE_KEYWORDS)

def _size_score(name: str) -> float:
    # Heuristic: prefer larger models if indicated (e.g., "70b", "405b")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(b|t)\b", name.lower())
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "t":
        val *= 1000.0
    return min(10.0, max(0.0, (val / 70.0) * 10.0))

def _context_score(ctx: Optional[int]) -> float:
    if not ctx:
        return 0.0
    return min(10.0, max(0.0, (ctx / 200000.0) * 10.0))

def _price_penalty(model: ModelInfo, quality: str) -> float:
    if model.price_prompt is None and model.price_completion is None:
        return 0.0
    total = (model.price_prompt or 0.0) + (model.price_completion or 0.0)
    if quality == "budget":
        return total * 4.0
    if quality == "balanced":
        return total * 2.0
    return total * 1.0  # 'best'

def _vendor_trust_boost(vendor: Optional[str]) -> float:
    if not vendor:
        return 0.0
    weights = {
        "openai": 6.0, "anthropic": 6.0, "google": 5.5, "mistral": 4.5,
        "meta-llama": 4.5, "qwen": 4.0, "groq": 4.0, "x-ai": 4.0,
        "cohere": 3.5, "perplexity": 3.0
    }
    return weights.get(vendor, 0.5)

def _capability_boost_reasoner(model_id: str) -> float:
    n = model_id.lower()
    score = 0.0
    for kw, val in [
        ("o3", 8.0), ("r1", 8.0), ("reason", 7.0), ("opus", 6.5),
        ("sonnet", 5.0), ("pro", 4.5), ("deepseek", 5.5), ("qwen2.5", 4.5),
        ("gpt-5", 9.0), ("o4", 7.5)
    ]:
        if kw in n: score += val
    return score + _size_score(n)

def _capability_boost_vision(model_id: str) -> float:
    n = model_id.lower()
    score = 0.0
    for kw, val in [
        ("vision", 8.0), ("4o", 8.0), ("multimodal", 6.0), ("gemini-1.5", 7.0),
        ("sonnet", 4.5), ("haiku", 3.5)
    ]:
        if kw in n: score += val
    # Context unknown -> 0; vision signal mostly from name
    return score

def _capability_boost_general(model_id: str) -> float:
    n = model_id.lower()
    score = 0.0
    for kw, val in [("mini", 2.0), ("small", 2.0), ("haiku", 3.0), ("sonnet", 4.0), ("pro", 4.0), ("turbo", 3.5)]:
        if kw in n: score += val
    return score + _size_score(n)

def confirm_yn(prompt: string, default: bool = False) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        ans = Prompt.ask(f"{prompt} [{default_label}]", default=("y" if default else "n"))
        a = ans.strip().lower()
        if a in ("y", "yes"): return True
        if a in ("n", "no"): return False
        console.print("Please enter yes or no.")

# --------------------------
# LLM Router with Multi-Provider Discovery
# --------------------------

class LLMRouter:
    def __init__(self, quality: str = "balanced") -> None:
        self.quality = quality  # 'best' | 'balanced' | 'budget'

        # Keys
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.xai_key = os.environ.get("XAI_API_KEY")
        self.mistral_key = os.environ.get("MISTRAL_API_KEY")
        self.google_key = os.environ.get("GOOGLE_API_KEY")

        # Primary provider for chat/planning — OpenRouter first for breadth
        self.provider = self._choose_primary_provider()
        self.client = self._make_client_for_chat()

    def _choose_primary_provider(self) -> str:
        if self.openrouter_key: return "openrouter"
        if self.openai_key:     return "openai"
        if self.groq_key:       return "groq"
        if self.xai_key:        return "xai"
        if self.anthropic_key:  return "anthropic"
        raise RuntimeError("No supported provider keys found. Set one of: OPENROUTER_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, XAI_API_KEY, or ANTHROPIC_API_KEY.")

    def _make_client_for_chat(self):
        if self.provider in ("openai", "openrouter", "groq", "xai"):
            if OpenAI is None:
                raise RuntimeError("openai package is required. pip install -r requirements.txt")
            if self.provider == "openai":
                return OpenAI(api_key=self.openai_key)
            if self.provider == "openrouter":
                return OpenAI(api_key=self.openrouter_key, base_url=self.openrouter_base)
            if self.provider == "groq":
                return OpenAI(api_key=self.groq_key, base_url="https://api.groq.com/openai/v1")
            if self.provider == "xai":
                return OpenAI(api_key=self.xai_key, base_url="https://api.x.ai/v1")
        return None  # Anthropics handled via requests

    # ---- Discovery per provider ----

    def _discover_openai(self) -> List[ModelInfo]:
        if not self.openai_key or OpenAI is None: return []
        try:
            client = OpenAI(api_key=self.openai_key)
            result = client.models.list()
            out = []
            for m in result.data:
                mid = getattr(m, "id", None) or ""
                if not mid or _has_excluded_keyword(mid): continue
                out.append(ModelInfo(provider="openai", model_id=mid, vendor="openai"))
            return out
        except Exception:
            return []

    def _discover_openrouter(self) -> List[ModelInfo]:
        if not self.openrouter_key: return []
        try:
            headers = {"Authorization": f"Bearer {self.openrouter_key}", "Content-Type": "application/json"}
            r = requests.get(f"{self.openrouter_base}/models", headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            out = []
            for m in data.get("data", []):
                mid = m.get("id")
                if not mid or _has_excluded_keyword(mid): continue
                pricing = m.get("pricing", {})
                ctx = m.get("context_length")
                vendor = _vendor_from_id(mid)
                out.append(ModelInfo(
                    provider="openrouter",
                    model_id=mid,
                    vendor=vendor,
                    price_prompt=_try_float(pricing.get("prompt")) if pricing else None,
                    price_completion=_try_float(pricing.get("completion")) if pricing else None,
                    context=ctx,
                    tags=m.get("tags") or []
                ))
            return out
        except Exception:
            return []

    def _discover_anthropic(self) -> List[ModelInfo]:
        if not self.anthropic_key: return []
        try:
            headers = { "x-api-key": self.anthropic_key, "anthropic-version": "2023-06-01" }
            r = requests.get("https://api.anthropic.com/v1/models", headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            out = []
            for m in data.get("data", []):
                mid = m.get("id")
                if not mid or _has_excluded_keyword(mid): continue
                out.append(ModelInfo(provider="anthropic", model_id=mid, vendor="anthropic"))
            return out
        except Exception:
            return []

    def _discover_groq(self) -> List[ModelInfo]:
        if not self.groq_key: return []
        try:
            headers = {"Authorization": f"Bearer {self.groq_key}"}
            r = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            out = []
            for m in data.get("data", []):
                mid = m.get("id") or m.get("name")
                if not mid or _has_excluded_keyword(mid): continue
                out.append(ModelInfo(provider="groq", model_id=mid, vendor="groq"))
            return out
        except Exception:
            return []

    def _discover_xai(self) -> List[ModelInfo]:
        if not self.xai_key: return []
        try:
            headers = {"Authorization": f"Bearer {self.xai_key}"}
            r = requests.get("https://api.x.ai/v1/models", headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            out = []
            for m in data.get("data", []):
                mid = m.get("id") or m.get("name")
                if not mid or _has_excluded_keyword(mid): continue
                out.append(ModelInfo(provider="xai", model_id=mid, vendor="x-ai"))
            return out
        except Exception:
            return []

    def _discover_mistral(self) -> List[ModelInfo]:
        if not self.mistral_key: return []
        try:
            headers = {"Authorization": f"Bearer {self.mistral_key}"}
            r = requests.get("https://api.mistral.ai/v1/models", headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            out = []
            for m in data.get("data", []):
                mid = m.get("id")
                if not mid or _has_excluded_keyword(mid): continue
                out.append(ModelInfo(provider="mistral", model_id=mid, vendor="mistral"))
            return out
        except Exception:
            return []

    def _discover_google(self) -> List[ModelInfo]:
        if not self.google_key: return []
        try:
            r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={self.google_key}", timeout=30)
            r.raise_for_status()
            data = r.json()
            out = []
            for m in data.get("models", []):
                mid = m.get("name")
                if not mid or _has_excluded_keyword(mid): continue
                model_name = mid.split("/")[-1]
                out.append(ModelInfo(provider="google", model_id=model_name, vendor="google"))
            return out
        except Exception:
            return []

    def discover_models(self) -> List[ModelInfo]:
        models: List[ModelInfo] = []
        models += self._discover_openrouter()
        models += self._discover_openai()
        models += self._discover_anthropic()
        models += self._discover_groq()
        models += self._discover_xai()
        models += self._discover_mistral()
        models += self._discover_google()

        if self.provider == "openrouter":
            # Keep only models callable via OpenRouter (and trusted vendors if vendor detected)
            models = [m for m in models if m.provider == "openrouter" and (m.vendor in TRUSTED_VENDORS if m.vendor else True)]
        return models

    # ---- Ranking / selection ----

    def _rank_for_role(self, models: List[ModelInfo], role: str) -> List[ModelInfo]:
        def score(m: ModelInfo) -> float:
            base = 0.0
            if role == "reasoner":
                base += _capability_boost_reasoner(m.model_id)
            elif role == "vision":
                # Penalize previews more
                if "preview" in m.model_id.lower():
                    base -= 2.0
                base += _capability_boost_vision(m.model_id)
            else:
                base += _capability_boost_general(m.model_id)

            base += _vendor_trust_boost(m.vendor or _vendor_from_id(m.model_id))
            base += _context_score(m.context)
            base -= _price_penalty(m, self.quality)
            return base

        filtered = []
        for m in models:
            n = m.model_id.lower()
            if role == "vision":
                if any(k in n for k in ["audio", "tts", "whisper", "search"]):
                    continue
            filtered.append(m)
        ranked = sorted(filtered, key=score, reverse=True)
        return ranked

    def _pick_top(self, models: List[ModelInfo], role: str) -> Optional[ModelInfo]:
        ranked = self._rank_for_role(models, role)
        return ranked[0] if ranked else None

    def pick_models(self, models: List[ModelInfo]) -> SelectedModels:
        if not models:
            if self.provider == "openrouter":
                return SelectedModels(provider="openrouter", reasoner="openai/o3-pro", general="openai/gpt-4o-mini", vision="openai/gpt-4o")
            if self.provider == "openai":
                return SelectedModels(provider="openai", reasoner="gpt-4o", general="gpt-4o-mini", vision="gpt-4o")
            if self.provider == "groq":
                return SelectedModels(provider="groq", reasoner="llama-3.1-70b-versatile", general="llama-3.1-8b-instant", vision=None)
            if self.provider == "xai":
                return SelectedModels(provider="xai", reasoner="grok-2-latest", general="grok-2-mini", vision=None)
            if self.provider == "anthropic":
                return SelectedModels(provider="anthropic", reasoner="claude-3-5-sonnet-latest", general="claude-3-haiku-latest", vision="claude-3-5-sonnet-latest")

        reasoner_top = self._pick_top(models, "reasoner")
        general_top  = self._pick_top(models, "general")
        vision_top   = self._pick_top(models, "vision")
        return SelectedModels(
            provider=self.provider,
            reasoner=reasoner_top.model_id if reasoner_top else (general_top.model_id if general_top else models[0].model_id),
            general=general_top.model_id if general_top else (reasoner_top.model_id if reasoner_top else models[0].model_id),
            vision=vision_top.model_id if vision_top else None
        )

    # ---- Chat interface (with JSON mode support + fallback) ----

    def chat(self, messages: List[Dict[str, Any]], model: str, temperature: float = 0.2, max_tokens: int = 2000, json_mode: bool = False) -> str:
        try:
            if self.provider in ("openai", "openrouter", "groq", "xai"):
                if self.client is None:
                    raise RuntimeError("Client not initialized")
                params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    params["response_format"] = {"type": "json_object"}
                try:
                    resp = self.client.chat.completions.create(**params)
                except Exception:
                    if json_mode:
                        # Retry without forced JSON mode if provider/model doesn't support it
                        params.pop("response_format", None)
                        resp = self.client.chat.completions.create(**params)
                    else:
                        raise
                return resp.choices[0].message.content or ""
            elif self.provider == "anthropic":
                # Anthropics via HTTP (no formal JSON mode, so instruct the model)
                sys_prompt = ""
                converted_msgs = []
                for m in messages:
                    role = m.get("role")
                    content = m.get("content", "")
                    if role == "system":
                        sys_prompt += content + "\n"
                    elif role in ("user", "assistant"):
                        converted_msgs.append({"role": role, "content": content})
                headers = {
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                body = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": converted_msgs if converted_msgs else [{"role": "user", "content": "Hello"}],
                }
                if sys_prompt.strip():
                    body["system"] = sys_prompt.strip()
                r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=60)
                r.raise_for_status()
                data = r.json()
                parts = data.get("content", [])
                text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
                return text
            else:
                raise RuntimeError("Unsupported provider for chat")
        except Exception as e:
            raise RuntimeError(f"LLM chat error: {e}") from e

    def chat_json(self, messages: List[Dict[str, Any]], model: str, temperature: float = 0.2, max_tokens: int = 2500) -> Dict[str, Any]:
        # Try strict JSON mode if supported, else fallback
        raw = self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        data = try_parse_json(raw)
        if data is not None:
            return data
        # Fallback: try extract codeblock/brace content
        extracted = extract_json_block(raw)
        if extracted:
            data = try_parse_json(extracted)
            if data is not None:
                return data
        # Last chance: repair via LLM
        repaired = self.repair_json_via_llm(raw, model=model)
        data = try_parse_json(repaired)
        if data is not None:
            return data
        raise RuntimeError(f"Failed to parse JSON after repair. Raw:\n{raw}")

    def repair_json_via_llm(self, raw: str, model: str) -> str:
        prompt = [
            {"role": "system", "content": "You repair invalid JSON into valid, strict JSON that matches the user's intent. Return JSON ONLY, no markdown."},
            {"role": "user", "content": f"Fix this into valid JSON (do not invent fields, just close strings/arrays/objects and remove comments):\n\n{raw}"}
        ]
        fixed = self.chat(prompt, model=model, temperature=0.0, max_tokens=2000, json_mode=True)
        # If still not valid, return raw to fail upstream
        return fixed if fixed else raw

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
- state_schema (dict of state_field_name -> 'str'|'list[str]'|'dict'|'int')
- steps (list of step names in execution order)
- dependencies (list of python packages for the GENERATED project)
- tasks_by_agent (dict agentName -> list of task bullet points)
- notes (optional)
"""

FOLLOWUP_QUESTION_PROMPT = """You are an AI product manager. Generate up to 5 short clarifying questions (JSON list of strings) based on the user's goal and any file summaries. Focus on purpose, data, constraints, outputs, and budget/performance needs.
Return JSON ONLY as an array of strings.
"""

ASSESS_REQUIREMENTS_PROMPT = """You are an AI analyst. Assess if the current requirements are sufficient to generate a production-ready system.
Return JSON ONLY with:
{
  "satisfied": boolean,
  "reasons": [string],
  "followups": [string]   // additional questions to ask if not satisfied
}
"""

DETECT_CONFLICTS_PROMPT = """You are an AI reviewer. Detect conflicts or inconsistencies in the user's requirements.
Return JSON ONLY with:
{
  "conflicts": [string],
  "suggestions": [string] // how to resolve conflicts
}
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
    try:
        qs = router.chat_json(msgs, model=models.general, temperature=0.2, max_tokens=700)
        if isinstance(qs, list):
            return [str(q) for q in qs][:5]
        if isinstance(qs, dict) and "questions" in qs and isinstance(qs["questions"], list):
            return [str(q) for q in qs["questions"]][:5]
    except Exception:
        pass
    return [
        "Who are the primary users?",
        "What data sources or files should the system use?",
        "What are the desired outputs and format?",
        "Any constraints (budget, latency, compliance)?",
        "How will success be measured?"
    ]

def assess_requirements(router: LLMRouter, models: SelectedModels, spec: UserSpec) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": ASSESS_REQUIREMENTS_PROMPT},
        {"role": "user", "content": json.dumps(spec.model_dump(), indent=2)}
    ]
    try:
        return router.chat_json(msgs, model=models.general, temperature=0.0, max_tokens=600)
    except Exception:
        return {"satisfied": True, "reasons": [], "followups": []}

def detect_conflicts(router: LLMRouter, models: SelectedModels, spec: UserSpec) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": DETECT_CONFLICTS_PROMPT},
        {"role": "user", "content": json.dumps(spec.model_dump(), indent=2)}
    ]
    try:
        return router.chat_json(msgs, model=models.general, temperature=0.0, max_tokens=700)
    except Exception:
        return {"conflicts": [], "suggestions": []}

def clarification_loop(router: LLMRouter, models: SelectedModels, spec: UserSpec, max_rounds: int = 3) -> UserSpec:
    """Iteratively ask follow-ups until satisfied or until max rounds."""
    for round_idx in range(max_rounds):
        questions = ask_followups(router, models, spec)
        if not questions:
            break
        console.print(f"[bold]Clarification round {round_idx+1}[/bold]")
        for q in questions:
            ans = Prompt.ask(f"[green]?[/green] {q}", default="")
            if ans.strip():
                spec.details[q] = ans.strip()

        # Check conflicts
        review = detect_conflicts(router, models, spec)
        conflicts = review.get("conflicts") or []
        if conflicts:
            console.print(Panel.fit("\n".join(conflicts), title="Potential Conflicts Detected", style="yellow"))
            if confirm_yn("Would you like to add clarifications to resolve these?", default=True):
                for conf in conflicts:
                    add = Prompt.ask(f"Resolution/clarification for: {conf}", default="")
                    if add.strip():
                        # Record under a special key so planner can see resolutions
                        spec.details.setdefault("conflict_resolutions", []).append({"conflict": conf, "resolution": add.strip()})

        # Assess completeness
        assessment = assess_requirements(router, models, spec)
        if bool(assessment.get("satisfied", False)):
            break
        followups = assessment.get("followups") or []
        if not followups:
            break
        console.print("[bold]Additional follow-ups needed:[/bold]")
        for q in followups:
            ans = Prompt.ask(f"[green]?[/green] {q}", default="")
            if ans.strip():
                spec.details[q] = ans.strip()
    return spec

def generate_plan(router: LLMRouter, models: SelectedModels, spec: UserSpec) -> PlanSpec:
    msgs = [
        {"role": "system", "content": SYSTEM_PLANNER},
        {"role": "user", "content": json.dumps(spec.model_dump(), indent=2)}
    ]
    # Use strict JSON path + repair
    data = router.chat_json(msgs, model=models.reasoner, temperature=0.2, max_tokens=3000)

    if "project_slug" not in data:
        data["project_slug"] = slugify(data.get("project_name", "project"))
    data.setdefault("dependencies", [])
    data.setdefault("agents", [{"name": "Planner", "role": "Planner", "responsibilities": ["Plan the work"]}])
    data.setdefault("state_schema", {"input": "str", "final_answer": "str"})
    data.setdefault("steps", ["plan", "work", "synthesize"])

    # Normalize responsibilities to string list if model returned a single string
    for a in data.get("agents", []):
        resp = a.get("responsibilities")
        if isinstance(resp, str):
            a["responsibilities"] = [resp]
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
        if ext in [".txt", ".md", ".log"]:
            summary, kind = read_text_file(p), "text"
        elif ext == ".pdf":
            summary, kind = summarize_pdf(p), "pdf"
        elif ext == ".csv":
            summary, kind = summarize_csv(p), "csv"
        elif ext in [".xlsx", ".xls"]:
            summary, kind = summarize_xlsx(p), "xlsx"
        elif ext == ".json":
            summary, kind = summarize_json(p), "json"
        else:
            summary, kind = f"[Binary or unsupported file type: {ext}. Metadata only.]", f"binary({ext})"
        results.append(FileSummary(path=p, kind=kind, bytes=st.st_size, summary=summary))
    return results

# --------------------------
# Generation Templates (project scaffold)
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
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
test:
	pytest -q
run:
	python src/run.py "Test the app end-to-end"
fmt:
	ruff check --fix || true
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
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

load_dotenv()

def _load_model_config():
    try:
        with open("model_selection.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

class LLMRouter:
    def __init__(self, preferred_reasoner: Optional[str] = None, preferred_general: Optional[str] = None, preferred_vision: Optional[str] = None):
        cfg = _load_model_config()
        provider_from_cfg = cfg.get("provider")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        if provider_from_cfg == "openrouter" and self.openrouter_key:
            self.provider = "openrouter"
        elif self.openai_key:
            self.provider = "openai"
        elif self.openrouter_key:
            self.provider = "openrouter"
        else:
            raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY")

        if OpenAI is None:
            raise RuntimeError("openai package required")

        if self.provider == "openai":
            self.client = OpenAI(api_key=self.openai_key)
        else:
            self.client = OpenAI(api_key=self.openrouter_key, base_url=f"{self.openrouter_base}")

        self.reasoner = preferred_reasoner or cfg.get("reasoner")
        self.general  = preferred_general  or cfg.get("general")
        self.vision   = preferred_vision   or cfg.get("vision")

    def chat(self, messages: List[Dict[str, Any]], model: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        model_to_use = model or self.general or self.reasoner
        if not model_to_use:
            model_to_use = "gpt-4o" if self.provider == "openai" else "openai/gpt-4o"
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
from dotenv import load_dotenv
from .llm_router import LLMRouter
from .graph import build_graph

load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/run.py \"Your task prompt\"")
        raise SystemExit(1)
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
            normalized = []
            for r in results:
                normalized.append({
                    "title": r.get("title"),
                    "href": r.get("href") or r.get("url"),
                    "snippet": r.get("body")
                })
            return normalized
    except Exception:
        return []
"""

TEMPLATE_TOOL_WEB = """\
import requests
def fetch_and_summarize(url: str) -> str:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        text = r.text
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
        ensure_dir(self.out_dir)
        ensure_dir(os.path.join(self.out_dir, "src", "agents"))
        ensure_dir(os.path.join(self.out_dir, "src", "tools"))
        ensure_dir(os.path.join(self.out_dir, "tests"))

        self._write("requirements.txt", TEMPLATE_REQUIREMENTS)
        self._write(".env.example", TEMPLATE_ENV_EXAMPLE)
        self._write("Dockerfile", TEMPLATE_DOCKERFILE)
        self._write("Makefile", TEMPLATE_MAKEFILE)

        readme = Template(TEMPLATE_README).render(project_name=self.plan.project_name, description=self.plan.description)
        self._write("README.md", readme)

        self._write("src/llm_router.py", TEMPLATE_LLM_ROUTER)
        self._write("src/graph.py", TEMPLATE_GRAPH)
        self._write("src/run.py", TEMPLATE_RUN)

        self._write("src/agents/planner.py", TEMPLATE_AGENT_PLANNER)
        self._write("src/agents/researcher.py", TEMPLATE_AGENT_RESEARCHER)
        self._write("src/agents/coder.py", TEMPLATE_AGENT_CODER)
        self._write("src/agents/synthesizer.py", TEMPLATE_AGENT_SYNTHESIZER)

        self._write("src/tools/files.py", TEMPLATE_TOOL_FILES)
        self._write("src/tools/search.py", TEMPLATE_TOOL_SEARCH)
        self._write("src/tools/web.py", TEMPLATE_TOOL_WEB)

        self._write("tests/test_smoke.py", TEMPLATE_TEST_SMOKE)

        model_config = {"provider": self.models.provider, "reasoner": self.models.reasoner, "general": self.models.general, "vision": self.models.vision}
        self._write("model_selection.json", json.dumps(model_config, indent=2))

    def _write(self, rel: str, content: str) -> None:
        path = os.path.join(self.out_dir, rel)
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

# --------------------------
# CLI + Flow
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

def _render_responsibilities(v: Any) -> str:
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    if isinstance(v, str):
        return v
    return str(v)

def interactive_model_choice(role: str, ranked: List[ModelInfo], default_model_id: str) -> str:
    table = Table(title=f"Top candidates for {role}", box=box.MINIMAL)
    table.add_column("#", justify="right")
    table.add_column("Model ID")
    table.add_column("Provider")
    table.add_column("Vendor")
    table.add_column("Prompt $")
    table.add_column("Completion $")
    table.add_column("Context")
    top = ranked[:8]
    for i, m in enumerate(top, 1):
        table.add_row(
            str(i),
            m.model_id,
            m.provider,
            m.vendor or "-",
            f"{m.price_prompt:.6f}" if m.price_prompt is not None else "-",
            f"{m.price_completion:.6f}" if m.price_completion is not None else "-",
            str(m.context or "-")
        )
    console.print(table)
    choice = Prompt.ask(f"Choose {role} model [1-{len(top)}] or press Enter to accept default", default="")
    if choice.strip().isdigit():
        idx = int(choice.strip())
        if 1 <= idx <= len(top):
            return top[idx - 1].model_id
    return default_model_id

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None

def extract_json_block(text: str) -> Optional[str]:
    # Prefer fenced code block ```json ... ```
    fence = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence[0]
    fence_any = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if fence_any:
        return fence_any[0]
    # Fallback: first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return None

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="LangGraph Creator Pro")
    parser.add_argument("goal", type=str, help="High-level goal (quoted)")
    parser.add_argument("--files", type=str, default="", help="Comma-separated file paths")
    parser.add_argument("--out-dir", type=str, default="out", help="Output directory")
    parser.add_argument("--yes", action="store_true", help="Non-interactive; skip follow-ups")
    parser.add_argument("--auto-models", action="store_true", help="Skip interactive model choice; accept defaults")
    parser.add_argument("--quality", type=str, choices=["best", "balanced", "budget"], default="balanced", help="Model selection preference")
    parser.add_argument("--max-clarify-rounds", type=int, default=3, help="Max clarification rounds before planning")
    args = parser.parse_args()

    console.print(Panel.fit("[bold cyan]LangGraph Creator Pro[/bold cyan]\nBuild intelligent multi-agent systems with zero code.", box=box.ROUNDED))

    # Router (with quality preference)
    try:
        router = LLMRouter(quality=args.quality)
    except Exception as e:
        console.print(f"[red]Error initializing providers: {e}[/red]")
        sys.exit(1)

    # Discover models
    console.print("[bold]Discovering available models (live)...[/bold]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task("Querying providers", total=None)
        discovered = router.discover_models()

    # Rank and pick
    reasoner_ranked = router._rank_for_role(discovered, "reasoner")
    general_ranked  = router._rank_for_role(discovered, "general")
    vision_ranked   = router._rank_for_role(discovered, "vision")

    selected = router.pick_models(discovered)

    # Display selections
    table = Table(title="Selected Models", box=box.MINIMAL)
    table.add_column("Role")
    table.add_column("Model ID")
    table.add_column("Provider")
    table.add_row("Reasoner", selected.reasoner, selected.provider)
    table.add_row("General", selected.general, selected.provider)
    table.add_row("Vision", selected.vision or "-", selected.provider)
    console.print(table)

    # Interactive override
    if not args.auto_models:
        if confirm_yn("Would you like to choose different models?", default=False):
            selected.reasoner = interactive_model_choice("reasoner", reasoner_ranked, selected.reasoner)
            selected.general  = interactive_model_choice("general", general_ranked, selected.general)
            if vision_ranked:
                selected.vision = interactive_model_choice("vision", vision_ranked, selected.vision or vision_ranked[0].model_id)

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

    # Compose spec
    spec = UserSpec(goal=args.goal, file_summaries=[fs.__dict__ for fs in file_summaries])

    # Clarifications loop (unless --yes)
    if not args.yes:
        console.print("[bold]A few quick questions to tailor the system...[/bold]")
        spec = clarification_loop(router, selected, spec, max_rounds=args.max_clarify_rounds)

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
        resp = _render_responsibilities(a.get("responsibilities", []))
        at.add_row(a.get("name", "?"), a.get("role", "?"), resp)
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
        if not confirm_yn("Proceed to generate the project now?", default=True):
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

    save_history({
        "ts": time.time(),
        "goal": args.goal,
        "details": spec.details,
        "files": files,
        "chosen_models": selected.model_dump(),
        "provider": router.provider
    })

if __name__ == "__main__":
    main()
