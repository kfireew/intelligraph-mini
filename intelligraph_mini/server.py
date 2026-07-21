"""
Intelligraph-mini — local-first MCP graph intelligence server.

Usage:
  python -m intelligraph_mini.server                    # uses current working directory
  python -m intelligraph_mini.server --repo-dir /path   # explicit repo
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
            "Returns relevant symbols WITH signatures, source snippets, and confidence levels "
            "(HIGH/MEDIUM/LOW). Use this FIRST. Usually sufficient for 'what/where is X' "
            "questions — no file read needed when confidence is HIGH. "
            "Example: 'add entity' finds 'upsertEntity'."
        ), inputSchema={"type": "object", "properties": {
            "query": {"type": "string"}
        }, "required": ["query"]}),
        types.Tool(name="node", description=(
            "Get a symbol's details, multi-hop subgraph (2-hop), 500-char source snippets for "
            "top 5 neighbors, and rationale notes. Use AFTER search. Snippets are usually "
            "sufficient — only read full files if you need implementation details beyond the "
            "snippet. Each result includes role annotations (hub/leaf) to gauge importance."
        ), inputSchema={"type": "object", "properties": {
            "name": {"type": "string"}, "depth": {"type": "integer", "default": 2}
        }, "required": ["name"]}),
        types.Tool(name="path", description=(
            "Trace the shortest path between two symbols in the codebase graph."
        ), inputSchema={"type": "object", "properties": {
            "from": {"type": "string"}, "to": {"type": "string"}
        }, "required": ["from", "to"]}),
        types.Tool(name="impact", description=(
            "Analyze blast-radius of changing a symbol. Traverses CALLS/IMPORTS_FROM edges. "
            "Returns affected files with depth + score. Use to plan refactors or assess risk."
        ), inputSchema={"type": "object", "properties": {
            "name": {"type": "string"}
        }, "required": ["name"]}),
        types.Tool(name="local_files", description=(
            "Read raw source files from disk. EXPENSIVE (~1000-4000 tokens per file). "
            "Use ONLY when search/node snippets are insufficient or search confidence is LOW. "
            "If a file was already covered by a prior search/node result, this tool will note "
            "what you already have before returning the content. Prefer 'node' for focused context."
        ), inputSchema={"type": "object", "properties": {
            "paths": {"type": "array", "items": {"type": "string"}},
            "max_bytes": {"type": "integer", "default": 15000}
        }, "required": ["paths"]}),
    ]


# ── Session tracking: which files search/node already described ──
# Maps file_path -> {tool, call_id, snippet_chars, had_signature, had_relationships}
_SESSION_SEEN = {}
_SESSION_STATS = {"search": 0, "node": 0, "path": 0, "impact": 0, "local_files": 0, "est_tokens": 0}
_SESSION_CALL_COUNTER = [0]


def _track_seen(file_path, tool, call_id, snippet_chars=0, had_signature=False, had_relationships=False):
    if not file_path:
        return
    _SESSION_SEEN[file_path] = {
        "tool": tool, "call_id": call_id, "snippet_chars": snippet_chars,
        "had_signature": had_signature, "had_relationships": had_relationships,
    }


def _log_call(tool, result_count, est_tokens):
    _SESSION_STATS[tool] = _SESSION_STATS.get(tool, 0) + 1
    _SESSION_STATS["est_tokens"] += est_tokens
    _SESSION_CALL_COUNTER[0] += 1
    cid = _SESSION_CALL_COUNTER[0]
    stats_summary = ", ".join(f"{k}={v}" for k, v in _SESSION_STATS.items() if k != "est_tokens")
    print(f"[intelligraph-mini] {tool}#{cid} -> {result_count} results, ~{est_tokens} tokens | session: {stats_summary}, total_tokens~{_SESSION_STATS['est_tokens']}", file=sys.stderr)


_SUFFICIENCY_TEXT = {
    "HIGH": (
        "Search confidence: HIGH\n"
        "These results include signatures, representative source snippets, and graph relationships. "
        "They are typically sufficient for architectural, navigation, and high-level code understanding. "
        "Open source files only if you need implementation details that are not present here."
    ),
    "MEDIUM": (
        "Search confidence: MEDIUM\n"
        "These results include partial signatures, snippets, and some graph relationships. "
        "They are typically sufficient for navigation and identifying relevant symbols, "
        "but implementation details may be incomplete. Open source files if you need fuller "
        "context around a specific function or its callers."
    ),
    "LOW": (
        "Search confidence: LOW\n"
        "These results are best-effort matches based on weak semantic similarity or fuzzy keyword hits. "
        "They identify candidate symbols but may not directly answer the question. "
        "Open source files to confirm relevance, or call node() on a specific symbol to explore its neighborhood."
    ),
    "NONE": (
        "Search confidence: NONE\n"
        "No symbols matched the query. Try rephrasing with a more specific symbol name, "
        "or open source files directly if you know the file path."
    ),
}


def _dispatch(name, args):
    provider = _get_provider()
    if not provider:
        return "No CRG database found. Run with --rebuild to build indexes."

    if name == "search":
        query = args.get("query", "")
        results = provider.hybrid_search(query, max_results=10, embedding_weight=0.4)
        if not results:
            _log_call("search", 0, 0)
            return _SUFFICIENCY_TEXT["NONE"] + f"\n\nNo results for '{query}'."
        # Fetch snippets for top 5 results so LLM gets context without reading files
        top_names = [r.get("name", "") for r in results[:5] if r.get("name")]
        snippets = provider.get_snippets(top_names, max_chars=250) if top_names else {}
        call_id = _SESSION_CALL_COUNTER[0] + 1
        # Sufficiency recommendation derived from top result's confidence
        top_conf = results[0].get("confidence", "MEDIUM")
        lines = [_SUFFICIENCY_TEXT.get(top_conf, _SUFFICIENCY_TEXT["MEDIUM"]), "",
                 f"## Search: '{query}' ({len(results)} results)"]
        for i, r in enumerate(results[:10], 1):
            sym = r.get("name", "?")
            kind = r.get("kind", "")
            fp = r.get("file_path", "")
            conf = r.get("confidence", "MEDIUM")
            reason = r.get("confidence_reason", "")
            sig = r.get("signature", "")
            snip_data = snippets.get(sym, {})
            snippet = (snip_data.get("snippet", "") or "").strip()[:200]
            lines.append(f"\n{i}. {sym} — `{fp}` [{kind}]")
            lines.append(f"   confidence: {conf} ({reason})")
            if sig:
                lines.append(f"   signature: {sig[:150]}")
            if snippet:
                lines.append(f"   snippet: {snippet}")
            _track_seen(fp, "search", call_id,
                        snippet_chars=len(snippet),
                        had_signature=bool(sig),
                        had_relationships=False)
        est_tokens = len(lines) * 25
        _log_call("search", len(results), est_tokens)
        return "\n".join(lines)

    if name == "node":
        sym = args.get("name", "")
        depth = min(3, max(1, args.get("depth", 2)))
        results = provider.hybrid_search(sym, max_results=5, embedding_weight=0.4)
        target = results[0]["name"] if results else sym
        trav = provider.traverse(target, max_hops=depth, max_nodes=20, max_tokens=300)
        # 500-char snippets for top 5 nodes
        node_names = [target] + [n["name"] for n in trav.get("nodes", [])[:5] if n["name"]]
        snippets = provider.get_snippets(node_names, max_chars=500)
        rationale = provider.get_rationale(target)
        call_id = _SESSION_CALL_COUNTER[0] + 1

        lines = [f"## {target}"]
        if trav.get("nodes"):
            lines.append(f"\n### Subgraph ({depth} hops, {len(trav['nodes'])} nodes)")
            for sn in trav["nodes"][:15]:
                indent = "  " * sn.get("depth", 0)
                arrow = "→ " if sn.get("depth", 0) > 0 else ""
                degree = sn.get("degree", 0)
                role = "hub" if degree >= 8 else ("connector" if degree >= 3 else "leaf")
                lines.append(f"{indent}{arrow}{sn.get('name','?')} — `{sn.get('file','')}` [{role}: degree {degree}]")
                _track_seen(sn.get("file", ""), "node", call_id,
                            snippet_chars=500,
                            had_signature=False,
                            had_relationships=True)
        if snippets:
            lines.append("\n### Source Code")
            for sn, sd in list(snippets.items())[:5]:
                snip = (sd.get("snippet", "") or "").strip()[:500]
                if snip:
                    lines.append(f"\n```python\n# {sn}\n{snip}\n```")
        if rationale:
            lines.append("\n### Notes")
            for rn in rationale[:5]:
                lines.append(f"- {rn.get('text','')}")
        est_tokens = len(lines) * 25
        _log_call("node", len(trav.get("nodes", [])), est_tokens)
        return "\n".join(lines)

    if name == "path":
        src, dst = args.get("from", ""), args.get("to", "")
        _log_call("path", 1, 100)
        return _find_path(provider, src, dst)

    if name == "impact":
        sym = args.get("name", "")
        results = provider.impact(sym, max_depth=2)
        _log_call("impact", len(results), len(results) * 10)
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
        total_bytes = 0
        for p in paths:
            # Source-aware: inform LLM what it already has from prior search/node calls
            seen = _SESSION_SEEN.get(p)
            info_prefix = ""
            if seen:
                already = []
                if seen.get("had_signature"):
                    already.append("function signature")
                if seen.get("snippet_chars", 0) > 0:
                    already.append(f"{seen['snippet_chars']}-char snippet")
                if seen.get("had_relationships"):
                    already.append("caller/callee relationships")
                if already:
                    info_prefix = (
                        f"[INFO] `{p}` was previously returned by {seen['tool']} result #{seen['call_id']}.\n"
                        f"Already provided: {', '.join(already)}.\n"
                        f"Reading the raw file will retrieve the complete implementation.\n\n"
                    )
            content = _read_local_file(p, max_bytes)
            total_bytes += len(content)
            ext = os.path.splitext(p)[1].lstrip(".")
            if content.startswith("ERROR"):
                lines.append(f"## {p}\n{content}")
            else:
                lines.append(f"{info_prefix}## {p}\n```{ext}\n{content}\n```")
        _log_call("local_files", len(paths), total_bytes // 4)
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
    parser.add_argument("--repo-dir", default=None, help="Path to your project (default: current working directory)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild indexes")
    args = parser.parse_args()
    REPO_DIR = os.path.abspath(args.repo_dir if args.repo_dir else os.getcwd())
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
