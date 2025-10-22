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

### Tablica Rejestracyjna PL (`tablica.py`)

Polish license plate reporting integration with tablica-rejestracyjna.pl.
- **Tools**:
  - `fetch_comments` - Get all comments/reports for a license plate
  - `submit_complaint` - Submit new violation reports with images
- **Features**: HEIC to JPEG conversion, automatic image optimization, LLM-guided workflow
- **Usage**: "Fetch comments for plate WW12345" or "Submit complaint for plate with image"

### EXIF Metadata Extractor (`exif_extractor.py`)

Extract EXIF metadata from images with GPS reverse geocoding.
- **Tool**: `analyze_image_metadata` - Extract metadata from up to 50 images
- **Features**:
  - GPS extraction (coordinates, altitude, speed, direction, timestamp)
  - Reverse geocoding to street addresses (via Nominatim)
  - Supports PNG, JPEG, HEIC formats
- **Usage**: "Extract metadata from photo.heic" or "Get GPS location from these images"

### Plate Recognition (`plate_recognition.py`)

Analyze traffic photos to identify license plates and violations using Gemini Vision.
- **Tool**: `recognize_plates` - Identify plates and determine violating vehicle
- **Features**: Multi-vehicle support, violation detection, pedestrian-focused analysis
- **Setup**: Set `GEMINI_API_KEY` in `.env` file
- **Usage**: "Analyze violation.jpg and identify the violating vehicle"

## Development

To create a new server, see [FastMCP documentation](https://gofastmcp.com). Project uses `mcp-servers` package name to avoid conflicts.

## Links

- [FastMCP](https://gofastmcp.com)
- [mise](https://mise.jdx.dev)
