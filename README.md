# MCP Servers

Personal MCP servers collection — Python 3.13 + [FastMCP](https://gofastmcp.com), managed with [mise](https://mise.jdx.dev) and [uv](https://docs.astral.sh/uv/).

## Requirements

- **mise** (`brew install mise`) — provisions Python 3.13 and uv automatically from `.mise.toml`.
- That's it. mise picks up `uv` from `[tools]` in `.mise.toml`; you don't need to install `python` or `uv` yourself.

## Quick Start

```bash
git clone https://github.com/swistaczek/mcp.git
cd mcp
mise install              # provisions Python 3.13 + uv from .mise.toml
mise run install          # uv sync — installs all project deps from uv.lock
mise run test             # 141 tests, ~3s, no network
```

To register the servers with Claude Code (user-scope, one-shot):

```bash
mise run claude-install                   # registers key-less servers
GEMINI_API_KEY=sk-... mise run claude-install-gemini   # registers Gemini servers
```

Then restart Claude Code and the servers appear under their slug names (`czynaczas`, `gazetki`, `tablica`, `exif_extractor`, `domains`, plus `gemini_image_descriptions` and `plate_recognition` if you ran the second task). Remove with `mise run claude-uninstall`.

## Configuration

### API keys

`gemini_image_descriptions` and `plate_recognition` need `GEMINI_API_KEY` (free at [AI Studio](https://aistudio.google.com/)). Two ways to provide it:

1. **Claude Code settings** — add to `.claude/settings.local.json` (gitignored):
   ```json
   { "env": { "GEMINI_API_KEY": "your-key-here" } }
   ```
2. **Environment** — `cp .env.example .env`, edit, and use `mise run claude-install-gemini` which forwards the var into the registration via `claude mcp add -e`.

### Two install paths

Servers can be registered two ways and both work:

- **Direct python (recommended)** — what `mise run claude-install` does. Points Claude Code at `.venv/bin/python <server>.py`. Fast cold-start, no fastmcp.json indirection, no `uv` shell-out.
- **fastmcp.json** — what `.mcp.json` uses (read by Claude Code on session start when this repo's directory is its cwd). Goes through `uv run --project . fastmcp run <server>.fastmcp.json`. Requires `uv` on `PATH` and the project's cwd to be this repo.

If you bind the repo as a workspace in Claude Code, `.mcp.json` is the easy path. For user-scope (always-on) registration outside this repo, use `mise run claude-install`.

## Available Servers

### Czy Na Czas (`czynaczas.py`)

Polish public-transport realtime data via czynaczas.pl — vehicle positions, delays, ETAs.
- **Cities**: `poznan`, `warsaw`, `krakow`, `wroclaw`, `lodz`
- **Tools**:
  - `list_supported_cities` — list the five cities
  - `find_stops(city, query)` — diacritic-insensitive multi-token search across the 3000+ stop list per city ("Dabrowskiego" matches "Dąbrowskiego")
  - `get_trip(city, trip_id)` — route polyline + scheduled stops
  - `get_vehicle(city, vehicle_id)` — fleet metadata (model, depot, A/C, low-floor)
  - `get_departures(city, stop_id, destination_filter?, line_filter?)` — headline tool. Snapshots all live vehicles via Socket.IO, filters by destination headsign + line, and computes ETA from each candidate's scheduled arrival plus realtime delay.
- **Usage**: "Kiedy najbliższy tramwaj 16 z Polna na Ogrody?" → `find_stops` → `get_departures(stop_id, destination_filter="Ogrody")`.

### Gazetki (`gazetki.py`)

Current promotional flyer PDFs from Polish supermarket chains — Lidl and Biedronka.
- **Tools**:
  - `list_flyers(chain)` — `lidl` / `biedronka` / `all`
  - `download_flyer(chain, flyer_id)` — streams native PDFs (Lidl) or assembles from per-page images (Biedronka)
  - `get_current_flyers(chain?)` — convenience: headline current flyer per chain
- **Cache**: `~/.cache/mcp-gazetki/<chain>/<flyer_id>/` (override with `GAZETKI_CACHE_DIR`).

### Tablica Rejestracyjna PL (`tablica.py`)

Polish license-plate violation reporting via tablica-rejestracyjna.pl.
- **Tools**: `fetch_comments`, `submit_complaint` (with image upload + HEIC→JPEG)

### EXIF Metadata Extractor (`exif_extractor.py`)

Extract EXIF + GPS from images and reverse-geocode to street addresses.
- **Tool**: `analyze_image_metadata` (up to 50 images, PNG/JPEG/HEIC, Nominatim geocoding)

### Plate Recognition (`plate_recognition.py`)

Identify license plates and traffic violations in photos via Gemini Vision.
- **Tool**: `recognize_plates` (multi-vehicle, pedestrian-perspective reasoning)
- **Setup**: requires `GEMINI_API_KEY`

### Image Descriptions (`gemini_image_descriptions.py`)

Generate alt text and accessible descriptions for images and GIFs via Gemini.
- **Tool**: `generate_image_descriptions` (batch up to 20 images, GIF support via FFmpeg)
- **Setup**: requires `GEMINI_API_KEY`

### Domain Checker (`domains.py`)

Batch domain registration check via WHOIS with DNS fallback and optional OVH browser-verified availability.
- **Tool**: `check_domains` (up to 50 domains, 50+ TLDs incl. .com.cn and Chinese IDN)

## Development

```bash
mise tasks                # list all available tasks
mise run dev <server>.fastmcp.json   # run a server in stdio mode
mise run test             # quick (no integration)
mise run test-all         # full (network + GEMINI_API_KEY needed)
```

To create a new server: copy `czynaczas.py` and `czynaczas.fastmcp.json` as a template, add the module name to `[tool.setuptools] py-modules` in `pyproject.toml`, and re-run `mise run install`. See [FastMCP docs](https://gofastmcp.com) for the full pattern.

## Links

- [FastMCP](https://gofastmcp.com) · [mise](https://mise.jdx.dev) · [uv](https://docs.astral.sh/uv/)
