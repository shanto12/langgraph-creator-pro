# LangGraph Creator Pro

A single-file Python assistant that converses with users, discovers current models, analyzes files, and generates complete, production-ready LangGraph multi-agent systems with zero code required from the user.

## Highlights
- Conversational: Asks intelligent follow-ups and clarifies requirements
- Model-Aware: Discovers available models in your account (OpenAI / OpenRouter)
- File-Savvy: Analyzes PDFs, CSVs, JSON, text, and integrates insights into generated systems
- Production Output: Generates LangGraph multi-agent app with Dockerfile, tests, docs
- Cost-Aware: Chooses reasoning vs. general models based on your needs

## Prerequisites
- Python 3.10+
- One of:
  - OPENAI_API_KEY
  - OPENROUTER_API_KEY (and OPENROUTER_BASE_URL, default provided)

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Simple start
python langgraph_creator.py "I want to build a research assistant"

# With files
python langgraph_creator.py "Summarize and analyze this data" --files "/path/to/data.csv,/path/to/report.pdf"

# Non-interactive mode (skip follow-ups)
python langgraph_creator.py "Build an internal knowledge assistant" --yes

# Customize output directory
python langgraph_creator.py "Create a grant-writing assistant" --out-dir "./out"
```

## Generated Project
The script creates a full-stack LangGraph multi-agent project (in ./out/<slug>/) with:
- Python package in `src/`
- Graph definition and agents
- LLM router configured to your keys
- Dockerfile, requirements, tests, Makefile, README
- .env.example with needed keys

## Keys
- Set `OPENAI_API_KEY` for OpenAI
- Or set `OPENROUTER_API_KEY` (and optionally `OPENROUTER_BASE_URL`)

```bash
export OPENAI_API_KEY="sk-..."
# or
export OPENROUTER_API_KEY="or-..."
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

## Notes
- By default, the creator prioritizes a strong reasoning model for planning, and a cheaper model for general tasks.
- You can edit the generated project's `src/llm_router.py` to add more providers or harden selection rules.
```
