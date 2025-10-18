# MCP Servers

Personal MCP servers using mise + Python 3.13 + FastMCP.

## Quick Start

```bash
# Install
mise run install

# Run server locally
fastmcp dev

# Available tasks
mise tasks
```

## Create a Server

```python
from fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool
def my_tool(param: str) -> str:
    """Tool description"""
    return f"Result: {param}"
```

## Claude Code Integration

### Default Server

`fastmcp.json` is auto-detected. Already configured in `.mcp.json` as "Echo" server.

### Add More Servers

1. Create `myserver.py` and `myserver.fastmcp.json`:
   ```json
   {
     "source": {"path": "myserver.py", "entrypoint": "mcp"},
     "environment": {"type": "uv", "project": "."},
     "deployment": {"transport": "stdio"}
   }
   ```

2. Add to `.mcp.json`:
   ```json
   {
     "My Server": {
       "type": "stdio",
       "command": "sh",
       "args": ["-c", "uv run --with fastmcp fastmcp run myserver.fastmcp.json"]
     }
   }
   ```

**Note**: Project name is `mcp-servers` (not `mcp`) to avoid package conflicts.

## Links

- [FastMCP](https://gofastmcp.com)
- [mise](https://mise.jdx.dev)
