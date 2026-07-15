"""
Intelligraph-mini — local-first MCP graph intelligence server.

Usage:
  python -m intelligraph_mini.server --repo-dir /path/to/project
  intelligraph-mini --repo-dir /path/to/project

First run builds graphify + CRG indexes (~60s). Subsequent runs load cached (~2s).

MCP tools:
  search(query)         — RRF hybrid search (FTS5 + semantic embeddings)
  node(name, depth=2)   — multi-hop subgraph + source snippets + rationale
  path(from, to)        — shortest path between symbols
  impact(name)          — blast-radius over CALLS/IMPORTS_FROM edges
  local_files(paths)    — read source files from disk
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict, deque

from mcp.server import Server
from mcp import types
from mcp.server.stdio import stdio_server

_VERSION = "0.1.0"
REPO_DIR = None
_JUNK_PATH_PATTERNS = [
    "/build/", "/bundle/", "/devtools/", "/dist/", "/out/",
    ".min.js", ".chunk.js", ".bundle.js", ".pack.js",
    "/generated/", "/codegen/", "/__generated__/",
    ".ngfactory.ts", "redux-dev-tools", "build-resources",
]


def _is_junk_path(fp):
    if not fp:
        return True
    lower = fp.lower() if isinstance(fp, str) else ""
    return any(p in lower for p in _JUNK_PATH_PATTERNS)


def _build_indexes(repo_dir):
    """Build graphify + CRG indexes on the repo."""
    print(f"[intelligraph-mini] Building indexes for {repo_dir}...", file=sys.stderr)

    # 1. Graphify
    gf_out = os.path.join(repo_dir, "graphify-out", "graph.json")
    if not os.path.exists(gf_out):
        print("[intelligraph-mini] Running graphify update...", file=sys.stderr)
        env = {**os.environ, "GRAPHIFY_MAX_WORKERS": "4"}
        try:
            subprocess.run(["graphify", "update", "."], cwd=repo_dir, timeout=300, env=env)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[intelligraph-mini] graphify build failed: {e}", file=sys.stderr)
    else:
        print("[intelligraph-mini] graphify-out/graph.json found, skipping", file=sys.stderr)

    # 2. CRG ignore file
    ignore_path = os.path.join(repo_dir, ".code-review-graphignore")
    try:
        with open(ignore_path, "w", encoding="utf-8") as f:
            f.write(
                "build-resources/**\nbundle/**\ndevtools/**\nredux-dev-tools/**\n"
                "out/**\n.nuxt/**\n.cache/**\n*.chunk.js\n*.bundle.js\n*.pack.js\n"
                "*.dev.js\n*.umd.js\n**/generated/**\n**/codegen/**\n**/__generated__/**\n"
                "*.ngfactory.ts\n*.ngstyle.ts\n*.shim.ngstyle.ts\nwebpack/**\n.vite/**\n"
            )
    except Exception:
        pass

    # 3. CRG build
    crg_db = os.path.join(repo_dir, ".code-review-graph", "graph.db")
    if not os.path.exists(crg_db):
        print("[intelligraph-mini] Running code-review-graph build...", file=sys.stderr)
        env = {**os.environ, "CRG_PARSE_WORKERS": "4"}
        try:
            subprocess.run(["code-review-graph", "build"], cwd=repo_dir, timeout=300, env=env)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[intelligraph-mini] CRG build failed: {e}", file=sys.stderr)
    else:
        print("[intelligraph-mini] .code-review-graph/graph.db found, skipping", file=sys.stderr)

    # 4. Populate node_snippets table
    crg_db = os.path.join(repo_dir, ".code-review-graph", "graph.db")
    if os.path.exists(crg_db):
        print("[intelligraph-mini] Populating source snippets...", file=sys.stderr)
        try:
            conn = sqlite3.connect(crg_db)
            conn.execute("CREATE TABLE IF NOT EXISTS node_snippets (node_name TEXT PRIMARY KEY, snippet TEXT)")
            nodes = conn.execute(
                "SELECT name, file_path, line_start, line_end FROM nodes "
                "WHERE line_start IS NOT NULL AND file_path IS NOT NULL AND name IS NOT NULL"
            ).fetchall()
            file_groups = defaultdict(list)
            for n in nodes:
                file_groups[n[1]].append(n)
            stored = 0
            for fp, node_list in file_groups.items():
                full_path = fp if os.path.isabs(fp) else os.path.join(repo_dir, fp)
                if not os.path.isfile(full_path):
                    continue
                try:
                    with open(full_path, "r", errors="replace") as f:
                        lines = f.readlines()
                except Exception:
                    continue
                for n in node_list:
                    start = max(0, (n[2] or 1) - 1)
                    end = min(len(lines), n[3] or start + 20)
                    snippet = "".join(lines[start:end])[:500]
                    if snippet.strip():
                        conn.execute("INSERT OR REPLACE INTO node_snippets VALUES (?, ?)", (n[0], snippet))
                        stored += 1
            conn.commit()
            conn.close()
            print(f"[intelligraph-mini] Stored {stored} snippets", file=sys.stderr)
        except Exception as e:
            print(f"[intelligraph-mini] Snippet storage failed: {e}", file=sys.stderr)


def _get_provider():
    """Load CRGProvider for the repo."""
    from .intelligence import CRGProvider
    crg_db = os.path.join(REPO_DIR, ".code-review-graph", "graph.db")
    gf_json = os.path.join(REPO_DIR, "graphify-out", "graph.json")
    graphify_data = {}
    if os.path.exists(gf_json):
        with open(gf_json) as f:
            graphify_data = json.load(f)
    proj = {"crg_db_path": crg_db, "graphify_data": graphify_data, "id": 0}
    provider = CRGProvider(proj)
    if not provider.is_available():
        return None
    return provider


def _read_local_file(path, max_bytes=15000):
    clean = path.replace("\\", "/").lstrip("/")
    full = os.path.normpath(os.path.join(REPO_DIR, clean))
    if not full.startswith(os.path.normpath(REPO_DIR)):
        return f"ERROR: path outside repo"
    if not os.path.isfile(full):
        return f"ERROR: file not found: {path}"
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes + 1)
        if len(content) > max_bytes:
            content = content[:max_bytes] + f"\n... (truncated at {max_bytes} bytes)"
        return content
    except Exception as e:
        return f"ERROR: {e}"


def _build_tools():
    return [
        types.Tool(name="search", description=(
            "Search the codebase using RRF hybrid search (keyword FTS5 + semantic embeddings). "
            "Finds symbols by meaning, not just exact text. "
            "Example: 'add entity' finds 'upsertEntity'."
        ), inputSchema={"type": "object", "properties": {
            "query": {"type": "string"}
        }, "required": ["query"]}),
        types.Tool(name="node", description=(
            "Get a symbol's details, multi-hop subgraph (2-hop), source code snippets, and rationale notes. "
            "Use after search to understand what a symbol connects to."
        ), inputSchema={"type": "object", "properties": {
            "name": {"type": "string"}, "depth": {"type": "integer", "default": 2}
        }, "required": ["name"]}),
        types.Tool(name="path", description=(
            "Trace the shortest path between two symbols in the codebase graph."
        ), inputSchema={"type": "object", "properties": {
            "from": {"type": "string"}, "to": {"type": "string"}
        }, "required": ["from", "to"]}),
        types.Tool(name="impact", description=(
            "Analyze blast-radius of changing a symbol. Traverses CALLS/IMPORTS_FROM edges."
        ), inputSchema={"type": "object", "properties": {
            "name": {"type": "string"}
        }, "required": ["name"]}),
        types.Tool(name="local_files", description=(
            "Read full source code from the local repository on disk."
        ), inputSchema={"type": "object", "properties": {
            "paths": {"type": "array", "items": {"type": "string"}},
            "max_bytes": {"type": "integer", "default": 15000}
        }, "required": ["paths"]}),
    ]


def _dispatch(name, args):
    provider = _get_provider()
    if not provider:
        return "No CRG database found. Run with --rebuild to build indexes."

    if name == "search":
        query = args.get("query", "")
        results = provider.hybrid_search(query, max_results=10, embedding_weight=0.4)
        if not results:
            return f"No results for '{query}'."
        lines = [f"## Search: '{query}' ({len(results)} results)"]
        for r in results:
            lines.append(f"- {r.get('name','?')} ({r.get('kind','')}) — `{r.get('file_path','')}` [score={r.get('score',0)}, {r.get('mode','')}]")
        return "\n".join(lines)

    if name == "node":
        sym = args.get("name", "")
        depth = min(3, max(1, args.get("depth", 2)))
        results = provider.hybrid_search(sym, max_results=5, embedding_weight=0.4)
        target = results[0]["name"] if results else sym
        trav = provider.traverse(target, max_hops=depth, max_nodes=20, max_tokens=300)
        snippets = provider.get_snippets([target] + [n["name"] for n in trav.get("nodes", [])[:3]], max_chars=500)
        rationale = provider.get_rationale(target)

        lines = [f"## {target}"]
        if trav.get("nodes"):
            lines.append(f"### Subgraph ({depth} hops, {len(trav['nodes'])} nodes)")
            for sn in trav["nodes"][:15]:
                indent = "  " * sn.get("depth", 0)
                lines.append(f"{indent}{'→ ' if sn.get('depth',0) > 0 else ''}{sn.get('name','?')} — `{sn.get('file','')}`")
        if snippets:
            lines.append("\n### Source Code")
            for sn, sd in list(snippets.items())[:3]:
                snip = sd.get("snippet", "")
                if snip:
                    lines.append(f"```{sn}\n{snip[:300]}\n```\n")
        if rationale:
            lines.append("### Notes")
            for rn in rationale[:5]:
                lines.append(f"- {rn.get('text','')}")
        return "\n".join(lines)

    if name == "path":
        src, dst = args.get("from", ""), args.get("to", "")
        return _find_path(provider, src, dst)

    if name == "impact":
        sym = args.get("name", "")
        results = provider.impact(sym, max_depth=2)
        if not results:
            return f"No impact data for '{sym}'."
        lines = [f"## Impact: '{sym}' ({len(results)} files)"]
        for r in results[:15]:
            lines.append(f"- `{r.get('file_path','')}` (depth={r.get('depth',0)}, score={r.get('score',0)})")
        return "\n".join(lines)

    if name == "local_files":
        paths = args.get("paths", [])
        max_bytes = args.get("max_bytes", 15000)
        lines = []
        for p in paths:
            content = _read_local_file(p, max_bytes)
            ext = os.path.splitext(p)[1].lstrip(".")
            if content.startswith("ERROR"):
                lines.append(f"## {p}\n{content}")
            else:
                lines.append(f"## {p}\n```{ext}\n{content}\n```")
        return "\n".join(lines)

    return f"Unknown tool: {name}"


def _find_path(provider, src_name, dst_name):
    """BFS shortest path using graphify links."""
    gf = provider.proj.get("graphify_data", {})
    nodes = gf.get("nodes", [])
    links = gf.get("links", [])
    adj = {}
    node_lookup = {}
    for n in nodes:
        nid = n.get("id") or n.get("label")
        if nid:
            adj.setdefault(nid, [])
            node_lookup[nid] = n
            node_lookup[nid.lower()] = n
    for l in links:
        s = l.get("source") or l.get("from")
        t = l.get("target") or l.get("to")
        if s and t:
            adj.setdefault(s, []).append(t)
            adj.setdefault(t, []).append(s)
    src_id = dst_id = None
    src_lower, dst_lower = src_name.lower(), dst_name.lower()
    for nid in node_lookup:
        if nid.lower() == src_lower:
            src_id = [k for k in node_lookup if k.lower() == src_lower][0]; break
    if not src_id:
        for nid, n in node_lookup.items():
            if src_lower in (n.get("label") or "").lower() or src_lower in nid.lower():
                src_id = nid; break
    for nid in node_lookup:
        if nid.lower() == dst_lower:
            dst_id = [k for k in node_lookup if k.lower() == dst_lower][0]; break
    if not dst_id:
        for nid, n in node_lookup.items():
            if dst_lower in (n.get("label") or "").lower() or dst_lower in nid.lower():
                dst_id = nid; break
    if not src_id or not dst_id:
        return "Node not found."
    visited = {src_id}
    queue = deque([(src_id, [(src_id, None)])])
    while queue:
        cur, path = queue.popleft()
        if cur == dst_id:
            lines = [f"## Path: {src_name} → {dst_name} ({len(path)-1} hops)"]
            for i, (nid, _) in enumerate(path):
                n = node_lookup.get(nid, {})
                fp = n.get("source_file", "")
                prefix = "  " if i > 0 else ""
                lines.append(f"{prefix}{'→ ' if i > 0 else ''}{n.get('label', nid)} — `{fp}`")
            return "\n".join(lines)
        for nb in adj.get(cur, []):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [(nb, None)]))
    return f"No path found between '{src_name}' and '{dst_name}'."


server = Server("intelligraph-mini")


@server.list_tools()
async def list_tools():
    return _build_tools()


@server.call_tool()
async def call_tool(name, arguments):
    try:
        text = _dispatch(name, arguments)
    except Exception as e:
        text = f"Error: {str(e)[:500]}"
    return [types.TextContent(type="text", text=text)]


def main():
    global REPO_DIR
    parser = argparse.ArgumentParser(description="Intelligraph-mini MCP server")
    parser.add_argument("--repo-dir", required=True, help="Path to your project")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild indexes")
    args = parser.parse_args()
    REPO_DIR = os.path.abspath(args.repo_dir)
    if not os.path.isdir(REPO_DIR):
        print(f"ERROR: {REPO_DIR} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.rebuild or not os.path.exists(os.path.join(REPO_DIR, ".code-review-graph", "graph.db")):
        _build_indexes(REPO_DIR)

    print(f"[intelligraph-mini] v{_VERSION} ready (repo={REPO_DIR})", file=sys.stderr)

    import asyncio
    async def _run():
        async with stdio_server() as (r, w):
            await server.run(r, w, server.create_initialization_options())
    asyncio.run(_run())


if __name__ == "__main__":
    main()
