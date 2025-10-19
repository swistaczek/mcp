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
# First resolve the library ID (returns multiple options, choose highest trust score)
mcp__Context7__resolve-library-id with libraryName="fastmcp"
# Recommended: /jlowin/fastmcp (Trust Score: 9.3, 1218 code snippets)

# Then fetch documentation on specific topics
mcp__Context7__get-library-docs with:
  - context7CompatibleLibraryID="/jlowin/fastmcp"
  - topic="tools" (or any specific topic)
  - tokens=2000 (adjust based on needed detail)
```

Common FastMCP documentation topics:
- "tools" - Creating and managing MCP tools
- "context" - Using Context for logging and progress
- "testing" - Writing tests for MCP servers
- "server" - Server setup and configuration
- "client" - Client usage patterns
- "deployment" - Deployment configurations

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