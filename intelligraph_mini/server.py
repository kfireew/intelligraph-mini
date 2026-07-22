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
    proj = {"crg_db_path": crg_db, "graphify_data": graphify_data, "id": 0, "repo_dir": REPO_DIR}
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
            "Search the codebase for symbols, files, or concepts. "
            "Returns name, kind, file path with line ranges (file:start-end), and confidence [H/M/L]. "
            "Use built-in Read with offset=line_start, limit=line_end-line_start to get source. "
            "Use this FIRST — replaces grep and glob."
        ), inputSchema={"type": "object", "properties": {
            "query": {"type": "string"}
        }, "required": ["query"]}),
        types.Tool(name="node", description=(
            "Get a symbol's connections (callers, callees, imports) with file:line ranges. "
            "Use AFTER search. Then use built-in Read with those line ranges to get implementation details. "
            "Replaces reading whole files — read only the specific line ranges shown."
        ), inputSchema={"type": "object", "properties": {
            "name": {"type": "string"}, "depth": {"type": "integer", "default": 2}
        }, "required": ["name"]}),
        types.Tool(name="path", description=(
            "Trace the shortest path between two symbols in the codebase graph."
        ), inputSchema={"type": "object", "properties": {
            "from": {"type": "string"}, "to": {"type": "string"}
        }, "required": ["from", "to"]}),
        types.Tool(name="impact", description=(
            "Complete blast radius of changing a symbol. Exhaustive traversal of ALL edge types. "
            "Returns every affected file with symbols to check. Use before refactoring. "
            "Files not listed do not depend on the target."
        ), inputSchema={"type": "object", "properties": {
            "name": {"type": "string"}
        }, "required": ["name"]}),
        types.Tool(name="local_files", description=(
            "Read full source files from disk. EXPENSIVE. "
            "Prefer built-in Read with line ranges from search/node results instead. "
            "Use this only when you need a whole file that search/node didn't cover."
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


# ── Session search dedup cache ───────────────────────────────────
_SESSION_SEARCHES = {}


def _dispatch(name, args):
    provider = _get_provider()
    if not provider:
        return "No CRG database found. Run with --rebuild to build indexes."

    if name == "search":
        query = args.get("query", "")
        results = provider.hybrid_search(query, max_results=10, embedding_weight=0.4)
        if not results:
            _log_call("search", 0, 0)
            return f"No symbols found matching '{query}'."

        # Dedup: same query in same session → cached one-liner
        cache_key = query.lower().strip()
        if cache_key in _SESSION_SEARCHES:
            prev = _SESSION_SEARCHES[cache_key]
            return f"[CACHED] Same as search#{prev['call_id']}. Files: {', '.join(prev['files'])}"

        call_id = _SESSION_CALL_COUNTER[0] + 1
        top_conf = results[0].get("confidence", "MEDIUM")
        conf_tag = {"HIGH": "H", "MEDIUM": "M", "LOW": "L"}.get(top_conf, "M")

        lines = [f'## "{query}" — {len(results)} results [{conf_tag}]']
        files_list = []

        for i, r in enumerate(results[:10], 1):
            rname = r.get("name", "?")
            kind = r.get("kind", "?")
            fp = r.get("file_path", "?")
            ls = r.get("line_start", 0)
            le = r.get("line_end", 0)
            r_conf = r.get("confidence", "MEDIUM")
            r_tag = {"HIGH": "H", "MEDIUM": "M", "LOW": "L"}.get(r_conf, "M")

            if ls and le and le > ls:
                loc = f"{fp}:{ls}-{le}"
            elif ls:
                loc = f"{fp}:{ls}"
            else:
                loc = fp
            files_list.append(loc)
            lines.append(f"{i}. {rname} ({kind}) {loc} [{r_tag}]")
            _track_seen(fp, "search", call_id, had_signature=bool(r.get("signature")))

        _SESSION_SEARCHES[cache_key] = {"call_id": call_id, "files": files_list}
        est_tokens = sum(len(l) for l in lines) // 4
        _log_call("search", len(results), est_tokens)
        return "\n".join(lines)

    if name == "node":
        sym = args.get("name", "")
        depth = min(3, max(1, args.get("depth", 2)))
        # Find the node via search, then traverse for connections
        results = provider.hybrid_search(sym, max_results=5, embedding_weight=0.4)
        target = results[0]["name"] if results else sym
        trav = provider.traverse(target, max_hops=depth, max_nodes=30, max_tokens=400)

        # Get line_start/line_end for target node
        target_snips = provider.get_snippets([target], max_chars=1)
        target_info = target_snips.get(target, {})
        target_ls = target_info.get("line_start", 0)
        target_le = target_info.get("line_end", 0)
        target_fp = target_info.get("file_path", "")

        call_id = _SESSION_CALL_COUNTER[0] + 1
        loc = f"{target_fp}:{target_ls}-{target_le}" if target_ls and target_le else target_fp
        lines = [f"## {target} {loc}", f"degree={len(trav.get('nodes', []))}"]

        # Connections — from subgraph traversal, one line each with file:line range
        trav_nodes = trav.get("nodes", [])
        if trav_nodes:
            # Get line ranges for all subgraph nodes
            node_names = [n["name"] for n in trav_nodes if n.get("name")]
            all_snips = provider.get_snippets(node_names, max_chars=1) if node_names else {}

            lines.append("")
            lines.append(f"### Connections ({len(trav_nodes)})")
            for n in trav_nodes:
                nname = n.get("name", "?")
                nfile = n.get("file", "")
                ndepth = n.get("depth", 0)
                nsnip = all_snips.get(nname, {})
                nls = nsnip.get("line_start", 0)
                nle = nsnip.get("line_end", 0)
                nloc = f"{nfile}:{nls}-{nle}" if nls and nle else (f"{nfile}:{nls}" if nls else nfile)
                prefix = "<-" if ndepth > 0 else "->"
                lines.append(f"  {prefix} {nname} {nloc}")
                _track_seen(nfile, "node", call_id, had_relationships=True)

        est_tokens = sum(len(l) for l in lines) // 4
        _log_call("node", len(trav_nodes), est_tokens)
        return "\n".join(lines)

    if name == "path":
        src, dst = args.get("from", ""), args.get("to", "")
        _log_call("path", 1, 100)
        return _find_path(provider, src, dst)

    if name == "impact":
        sym = args.get("name", "")
        results = provider.impact(sym)
        _log_call("impact", len(results), len(results) * 20)
        if not results:
            return f"No impact data for '{sym}'."
        lines = [f"## Impact: '{sym}' ({len(results)} files — exhaustive)"]
        lines.append("Files not listed here do not depend on the target in the code graph.")
        lines.append("")
        for r in results:
            fp = r.get("file_path", "?")
            depth = r.get("depth", 0)
            symbols = r.get("symbols", [])
            edge_types = r.get("edge_types", [])
            sources = r.get("sources", [])
            depth_label = "definition" if depth == 0 else f"depth {depth}"
            src_label = "/".join(sources) if sources else "crg"
            lines.append(f"- {fp} ({depth_label}, {src_label})")
            if symbols:
                lines.append(f"  symbols: {', '.join(symbols[:5])}")
            if edge_types:
                lines.append(f"  edges: {', '.join(edge_types[:5])}")
        return "\n".join(lines)

    if name == "local_files":
        paths = args.get("paths", [])
        max_bytes = args.get("max_bytes", 15000)
        lines = []
        total_bytes = 0
        for p in paths:
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
