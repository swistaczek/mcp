# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal MCP servers collection built with FastMCP framework. The project uses Python 3.13+, mise for task management, and uv for dependency management.

## Development Commands

```bash
# Setup and Dependencies
mise run install              # Install project dependencies via uv

# Testing
python -m pytest tests/ -v    # Run full test suite
python -m pytest tests/test_domains.py::TestTLDExtraction -v  # Run specific test class
python -m pytest tests/ -k "chinese" -v  # Run tests matching pattern

# Development
mise tasks                    # List all available mise tasks
mise run dev                  # Run server in development mode
```

## Architecture

### MCP Server Pattern
Each server follows the FastMCP pattern:
1. Import FastMCP and create instance: `mcp = FastMCP("Server Name")`
2. Define tools using `@mcp.tool` decorator with Pydantic field validation
3. Use async functions for I/O operations (WHOIS queries, DNS lookups)
4. Return `ToolResult` with both human-readable text and structured JSON

### Domain Checker Architecture
- **Primary method**: WHOIS protocol queries (port 43, 10s timeout)
- **Fallback**: DNS record lookup when WHOIS fails
- **Batch processing**: Up to 50 domains per request
- **TLD support**: 40+ TLDs including compound (.com.cn) and Chinese IDN domains
- **Pattern matching**: TLD-specific "not found" patterns in `NOT_FOUND_PATTERNS` dict

## Testing Strategy

Tests are organized by functionality:
- `TestTLDExtraction` - TLD parsing logic
- `TestChineseCharacterDetection` - Unicode character detection
- `TestNotFoundPatterns` - WHOIS response parsing
- `TestWHOISServerMapping` - Server configuration validation
- Integration tests marked with `@pytest.mark.skip` (require mocking)

Test fixtures in `tests/fixtures/whois_responses.json` contain sample WHOIS responses.

## FastMCP Documentation with Context7

When working with FastMCP, use Context7 MCP tool for up-to-date documentation:
```
# Official FastMCP documentation (most comprehensive)
mcp__Context7__get-library-docs with:
  - context7CompatibleLibraryID="/llmstxt/gofastmcp_llms-full_txt"
  - topic="tools server" (or any specific topic)
  - tokens=2500 (adjust based on needed detail)

# Alternative: GitHub source (if specific implementation needed)
mcp__Context7__get-library-docs with:
  - context7CompatibleLibraryID="/jlowin/fastmcp"
  - topic="your_topic"
```

**Recommended documentation source:** `/llmstxt/gofastmcp_llms-full_txt`
- 12,289 code snippets (official FastMCP documentation)
- Trust Score: 8.0

**Common FastMCP documentation topics:**
- "getting started" - Quickstart and installation
- "decorators context" - Tool/prompt/resource decorators with Context
- "tools server" - Server tool management and patterns
- "testing" - Writing tests for MCP servers
- "deployment" - Deployment configurations
- "client" - Client usage patterns

**Key FastMCP patterns from docs:**
```python
# Tool with Context injection
@mcp.tool
async def process_file(file_uri: str, ctx: Context) -> str:
    ctx.info("Processing file")  # Use context for logging
    return "Result"

# Resource definition
@mcp.resource("resource://{city}/weather")
def get_weather(city: str) -> str:
    return f"Weather for {city}"
```

## Adding New Servers

1. Create `servername.py` with FastMCP instance
2. Implement tools using `@mcp.tool` decorator
3. Add to `[tool.setuptools] py-modules` in pyproject.toml
4. Create corresponding tests in `tests/test_servername.py`
5. Update `.mcp.json` for Claude Code integration

## CI/CD Pipeline

GitHub Actions runs on push to main and PRs:
- Python 3.13 on Ubuntu latest
- Installs project with `pip install -e .`
- Runs pytest with verbose output
- Claude Code integration for PR reviews (claude-code-review.yml)