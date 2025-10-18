# MCP Servers

Personal MCP servers collection using mise + Python 3.13 + FastMCP.

## Project Structure

```
mcp/
├── domains.py           # Domain registration checker
├── domains.fastmcp.json # Domain checker config
├── tests/               # Unit tests
│   ├── test_domains.py
│   └── fixtures/
├── .mcp.json           # Claude Code configuration
└── pyproject.toml      # Python dependencies
```

## Quick Start

```bash
# Install dependencies
mise run install

# Run tests
python -m pytest tests/ -v

# Available tasks
mise tasks
```

## Available Servers

### Domain Checker

Check domain registration status via WHOIS with DNS fallback.

**Tool**: `check_domains`
- Batch check up to 50 domains
- Real-time progress reporting
- WHOIS query with DNS fallback
- Supports 50+ TLDs including compound (.com.cn) and Chinese TLDs

**Example usage in Claude Code**:
```
Check if these domains are available: example.com, test.io, demo.ai
```

## Creating a New Server

1. Create `myserver.py`:
   ```python
   from fastmcp import FastMCP

   mcp = FastMCP("My Server")

   @mcp.tool
   def my_tool(param: str) -> str:
       """Tool description"""
       return f"Result: {param}"
   ```

2. Create `myserver.fastmcp.json`:
   ```json
   {
     "source": {"path": "myserver.py", "entrypoint": "mcp"},
     "environment": {"type": "uv", "project": "."},
     "deployment": {"transport": "stdio"}
   }
   ```

3. Add to `.mcp.json`:
   ```json
   {
     "mcpServers": {
       "My Server": {
         "type": "stdio",
         "command": "sh",
         "args": ["-c", "uv run --with fastmcp fastmcp run myserver.fastmcp.json"]
       }
     }
   }
   ```

**Note**: Project name is `mcp-servers` (not `mcp`) to avoid package conflicts.

## Links

- [FastMCP](https://gofastmcp.com)
- [mise](https://mise.jdx.dev)
