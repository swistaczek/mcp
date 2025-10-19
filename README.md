# MCP Servers

Personal MCP servers collection using mise + Python 3.13 + FastMCP.

## Quick Start

```bash
mise run install        # Install dependencies
python -m pytest tests/ # Run tests
mise tasks             # List available tasks
```

## Available Servers

### Domain Checker (`domains.py`)

Batch check domain registration status via WHOIS with DNS fallback.
- **Tool**: `check_domains` - Check up to 50 domains
- **Features**: Real-time progress, 50+ TLDs including .com.cn and Chinese TLDs
- **Usage**: "Check if example.com and test.io are available"

## Development

To create a new server, see [FastMCP documentation](https://gofastmcp.com). Project uses `mcp-servers` package name to avoid conflicts.

## Links

- [FastMCP](https://gofastmcp.com)
- [mise](https://mise.jdx.dev)
