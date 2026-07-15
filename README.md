# Intelligraph-mini

Local-first MCP graph intelligence server — same RRF hybrid search, multi-hop traversal, source snippets, and rationale nodes as the full Intelligraph platform, but without Docker, web UI, SSO, or chat. Just tools for your AI agent.

> **Need the full platform?** [Intelligraph](https://github.com/kfireew/Intelligraph) adds Docker, React web UI, chat completions, SSO/PKCE, closed-network deployment, and tuning controls.

## Quick start

```bash
pip install intelligraph-mini

# In your project directory:
intelligraph-mini --repo-dir .

# Or with MCP config (.mcp.json):
{
  "mcpServers": {
    "intelligraph-mini": {
      "command": "intelligraph-mini",
      "args": ["--repo-dir", "."]
    }
  }
}
```

First run builds graphify + CRG indexes (~60s). Subsequent runs load cached (~2s). The bundled `all-MiniLM-L6-v2` model (87MB) works fully offline — no API calls, no network.

## Tools

| Tool | Description |
|------|-------------|
| `search(query)` | RRF hybrid search (FTS5 + semantic embeddings). Finds symbols by meaning. |
| `node(name, depth=2)` | Multi-hop subgraph + source code snippets + rationale notes. |
| `path(from, to)` | Shortest path between two symbols in the call graph. |
| `impact(name)` | Blast-radius analysis over CALLS/IMPORTS_FROM edges. |
| `local_files(paths)` | Read source files from disk. |

## How it works

1. **Build** (first run): `graphify update .` + `code-review-graph build` → `graphify-out/graph.json` + `.code-review-graph/graph.db`
2. **Snippets**: reads source files, stores ~500 char snippets per node in `node_snippets` table
3. **Search**: RRF (Reciprocal Rank Fusion, k=30) blends FTS5 keyword ranking with embedding cosine similarity. Adaptive 50% cutoff returns only genuinely relevant files.
4. **Traversal**: BFS with token budget over cached adjacency (scales to 140k edges)
5. **Rationale**: surfaces `#NOTE`/`#WHY` nodes from graphify's rationale extraction

## Requirements

- Python 3.10+
- `graphifyy` and `code-review-graph` CLIs on PATH (installed automatically as dependencies)

## License

MIT
