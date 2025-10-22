# MCP Servers

Personal MCP servers collection using mise + Python 3.13 + FastMCP.

## Quick Start

```bash
mise run install        # Install dependencies
python -m pytest tests/ # Run tests
mise tasks             # List available tasks
```

## Configuration

API keys can be configured in two ways:

1. **Claude Code settings** (recommended for Claude Code users):
   - Add to `.claude/settings.local.json` (gitignored):
     ```json
     {
       "env": {
         "GEMINI_API_KEY": "your-key-here"
       }
     }
     ```
   - API keys are automatically available to all MCP servers

2. **Environment file** (for other MCP clients):
   - Copy `.env.example` to `.env` and add your keys
   - Note: `.env` is gitignored to protect secrets

## Available Servers

### Domain Checker (`domains.py`)

Batch check domain registration status via WHOIS with DNS fallback.
- **Tool**: `check_domains` - Check up to 50 domains
- **Features**: Real-time progress, 50+ TLDs including .com.cn and Chinese TLDs
- **Usage**: "Check if example.com and test.io are available"

### Gemini Alt Tag Generator (`gemini_alt.py`)

Generate accessible alt tags for images using Google's Gemini LLM.
- **Tool**: `generate_alt_tags` - Process up to 20 images with optional context
- **Features**:
  - Smart image optimization (adaptive resizing for text-heavy images)
  - Batch processing (up to 10 images per request)
  - Context from file or text (saves tokens for large documents)
- **Setup**: Set `GEMINI_API_KEY` in `.env` file (get from [AI Studio](https://aistudio.google.com/))
- **Usage**:
  - `generate_alt_tags(images=["img.png"], context="./docs/guide.md")`
  - Accepts image paths, URLs, or base64 data

## Development

To create a new server, see [FastMCP documentation](https://gofastmcp.com). Project uses `mcp-servers` package name to avoid conflicts.

## Links

- [FastMCP](https://gofastmcp.com)
- [mise](https://mise.jdx.dev)
