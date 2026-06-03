"""
Autonomous Research Agent (ReAct Pattern)
=========================================
Takes a research question, searches the web, reads sources,
and produces a structured Markdown summary — autonomously.

Setup:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    export TAVILY_API_KEY=tvly-...
    python agent.py "What are the latest advances in RAG systems in 2025?"

Architecture:
    ReAct loop → Thought → Action (search/read/summarise) → Observation → repeat
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import OpenAI

client = OpenAI()
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
MODEL = "gpt-4o"
MAX_ITERATIONS = 12

SYSTEM_PROMPT = """You are an autonomous research agent. You research topics thoroughly
by searching the web and reading sources, then synthesise findings into a structured report.

Use this EXACT format for every step:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON input for the tool]

When you have enough information to write the final report:
Thought: I have enough information to write the final report.
Action: finish
Action Input: {"report": "your full markdown report here"}

Available tools:
- search: Search the web. Input: {"query": "your search query"}
- read_url: Read the content of a URL. Input: {"url": "https://..."}
- finish: End the research and return the report. Input: {"report": "..."}
"""


@dataclass
class AgentState:
    task: str
    history: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    iteration: int = 0


# ── Tools ────────────────────────────────────────────────────────────────────

def search(query: str) -> str:
    """Search the web using Tavily."""
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return "\n\n".join(
        f"**{r['title']}**\nURL: {r['url']}\n{r.get('content', '')[:600]}"
        for r in results
    )


def read_url(url: str) -> str:
    """Fetch the text content of a URL (simplified)."""
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        # Very basic HTML stripping
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:3000]
    except Exception as e:
        return f"Could not read URL: {e}"


TOOLS = {"search": search, "read_url": read_url}


# ── Agent loop ───────────────────────────────────────────────────────────────

def parse_action(text: str) -> tuple[str, Any] | None:
    """Parse Action and Action Input from LLM output."""
    action_match = re.search(r"Action:\s*(.+)", text)
    input_match = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)
    if not action_match:
        return None
    action = action_match.group(1).strip()
    try:
        action_input = json.loads(input_match.group(1)) if input_match else {}
    except json.JSONDecodeError:
        action_input = {}
    return action, action_input


def run_agent(task: str) -> str:
    state = AgentState(task=task)
    state.history.append({"role": "user", "content": f"Research task: {task}"})

    print(f"\n{'='*60}")
    print(f"Research Task: {task}")
    print("="*60)

    for i in range(MAX_ITERATIONS):
        state.iteration = i + 1

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + state.history,
            temperature=0.3,
        )
        assistant_text = response.choices[0].message.content
        state.history.append({"role": "assistant", "content": assistant_text})

        print(f"\n[Step {state.iteration}]\n{assistant_text[:500]}...")

        parsed = parse_action(assistant_text)
        if not parsed:
            break

        action, action_input = parsed

        if action == "finish":
            report = action_input.get("report", "No report generated.")
            print(f"\n{'='*60}\nFINAL REPORT\n{'='*60}\n{report}")
            return report

        if action in TOOLS:
            try:
                tool_result = TOOLS[action](**action_input)
                observation = f"Observation: {tool_result}"
            except Exception as e:
                observation = f"Observation: Tool error — {e}"
        else:
            observation = f"Observation: Unknown tool '{action}'"

        print(f"\n{observation[:300]}...")
        state.history.append({"role": "user", "content": observation})

    return "Max iterations reached without completing the research."


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is RAG in AI and how does it work?"
    result = run_agent(task)
    print("\n\nDone.")
