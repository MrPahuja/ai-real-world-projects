"""
GitHub MCP Server
=================
Exposes GitHub operations as MCP tools and resources.

Tools:
  - read_file(owner, repo, path)  — read a file's contents
  - list_issues(owner, repo)      — list open issues
  - create_issue(owner, repo, title, body) — create an issue
  - search_code(query, repo)      — search code in a repo

Resources:
  - repo://owner/repo/readme      — the repository README
  - repo://owner/repo/structure   — directory tree

Setup:
    pip install -r requirements.txt
    export GITHUB_TOKEN=ghp_...
    python server.py

Compatible with any MCP host (Claude Desktop, etc.)
"""

import asyncio
import json
import os
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    TextContent,
    Tool,
)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
BASE_URL = "https://api.github.com"

server = Server("github-mcp-server")


def github_headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def github_get(path: str) -> Any:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}{path}", headers=github_headers())
        resp.raise_for_status()
        return resp.json()


# ── Tools ────────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(tools=[
        Tool(
            name="read_file",
            description="Read the contents of a file from a GitHub repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "path": {"type": "string", "description": "File path in the repo"},
                    "ref": {"type": "string", "description": "Branch, tag, or commit (default: main)", "default": "main"},
                },
                "required": ["owner", "repo", "path"],
            },
        ),
        Tool(
            name="list_issues",
            description="List open issues in a GitHub repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
                    "limit": {"type": "integer", "default": 20, "maximum": 100},
                },
                "required": ["owner", "repo"],
            },
        ),
        Tool(
            name="create_issue",
            description="Create a new issue in a GitHub repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["owner", "repo", "title"],
            },
        ),
    ])


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    try:
        if name == "read_file":
            ref = arguments.get("ref", "main")
            data = await github_get(
                f"/repos/{arguments['owner']}/{arguments['repo']}/contents/{arguments['path']}?ref={ref}"
            )
            import base64
            content = base64.b64decode(data["content"]).decode("utf-8")
            return CallToolResult(content=[TextContent(type="text", text=content)])

        elif name == "list_issues":
            state = arguments.get("state", "open")
            limit = min(arguments.get("limit", 20), 100)
            data = await github_get(
                f"/repos/{arguments['owner']}/{arguments['repo']}/issues?state={state}&per_page={limit}"
            )
            issues_text = "\n".join(
                f"#{i['number']}: {i['title']} ({i['state']}) — {i['html_url']}"
                for i in data
            )
            return CallToolResult(content=[TextContent(type="text", text=issues_text or "No issues found.")])

        elif name == "create_issue":
            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    f"{BASE_URL}/repos/{arguments['owner']}/{arguments['repo']}/issues",
                    headers=github_headers(),
                    json={
                        "title": arguments["title"],
                        "body": arguments.get("body", ""),
                        "labels": arguments.get("labels", []),
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            return CallToolResult(
                content=[TextContent(type="text", text=f"Created issue #{data['number']}: {data['html_url']}")]
            )

        else:
            return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")])

    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Error: {e}")], isError=True)


# ── Resources ────────────────────────────────────────────────────────────────

@server.list_resources()
async def list_resources() -> ListResourcesResult:
    return ListResourcesResult(resources=[
        Resource(
            uri="repo://help",
            name="How to use GitHub resources",
            description="Instructions for accessing repo resources",
            mimeType="text/plain",
        )
    ])


@server.read_resource()
async def read_resource(uri: str) -> ReadResourceResult:
    if uri == "repo://help":
        return ReadResourceResult(
            contents=[TextContent(type="text", text="Use tool read_file to access repo files.")]
        )
    raise ValueError(f"Unknown resource: {uri}")


# ── Entry point ──────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
