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

Then restart Claude Code and the servers appear under their slug names (`czynaczas`, `gazetki`, `tablica`, `exif_extractor`, `domains`, `krs`, `poznan_events`, plus `gemini_image_descriptions` and `plate_recognition` if you ran the second task). Remove with `mise run claude-uninstall`.

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

### KRS Poland (`krs.py`)

Polish National Court Register (KRS) Open API — fetch the current or full extract for any registered entity.
- **Tools**:
  - `get_krs_current_extract(krs, rejestr?)` — odpis aktualny: name, NIP, REGON, formaPrawna, address, board, capital
  - `get_krs_full_extract(krs, rejestr?)` — odpis pełny: full historical record (past names, board changes, all entries)
- **Inputs**: KRS is 10 digits starting with `0` (e.g. `0001236495`); `rejestr` is `P` (companies) or `S` (associations/foundations), defaults to `P`. A 204 response hints at trying the other register.
- **Source**: `api-krs.ms.gov.pl/api/krs` (no key required). Port of [krs-poland-mcp-server](https://github.com/pkolawa/krs-poland-mcp-server).

### Poznań Events (`poznan_events.py`)

City of Poznań events calendar (Co? Gdzie? Kiedy? — `poznan.pl/mim/events/`).
- **Tools**:
  - `list_events(category?, page?, pages?)` — paginated event list (20 per page); optional category id filter; up to 5 pages per call
  - `get_event(event_ref)` — full detail (title, ISO date, time, place, categories, short + long description, hero image). Accepts numeric id, slug+id, path, or full URL
  - `list_event_categories` — catalog of category names with numeric ids (Sport=214, Muzyka=217, Sztuka=218, Teatr, Film, Książki, Dziecko, …)

### Label Printer (`label_printer.py`)

Prints courier labels (InPost, DPD, DHL…) on a direct-thermal label printer attached to this machine. Works as an MCP server *and* a standalone CLI.
- **Tools**:
  - `list_thermal_printers` — detects connected label printers, whether each has a CUPS queue, and its page-description language
  - `print_label(file_path, printer?, label_size?, copies?, darkness?, speed?, threshold?, bold?, dry_run?)` — renders and prints a PDF or image; creates the CUPS queue on first use
  - `print_qr_code(data, printer?, module_dots?, error_correction?, caption?)` — prints a QR code for any URL or text
  - `get_printer_status(printer?)` — queue state and pending jobs
  - `cancel_print_jobs(printer?, job_id?)` — clear the queue
- **CLI**: `uv run python label_printer.py detect | status | install | render | print | qr | cancel` (add `--json` for machine-readable output)
- **Two transports**: CUPS-backed label printers (Zebra EPL2/ZPL, Dymo) go through `lp`; cheap ESC/POS receipt printers, which no bundled CUPS driver can drive, are handled directly over libusb by `escpos_printer.py`. Both are listed together and picked automatically.
- **Rendering**: thermal heads are 1-bit, so the label is rasterised at 4× resolution, box-filtered down to the exact dot grid, and hard-thresholded to pure black/white. Letting the print system dither an anti-aliased page instead produces visibly faint output with barcodes that scan poorly.
- **Auto-detection**: with a single thermal printer attached, `printer` can be omitted everywhere. Zebra, Dymo, TSC, Godex, Citizen, Bixolon, Sato, Argox, Intermec, Honeywell, Brother QL, Toshiba TEC and Seiko devices are recognised.
- **Requires**: `pdftoppm` (`brew install poppler`) for PDF input, plus the CUPS command-line tools. Verified on macOS with a Zebra TLP2844 (USB, EPL2, 203 dpi) and 4×6" stock.
- **Usage**: "Print this InPost label" → `print_label(file_path="~/Downloads/label.pdf")`. Darker output → raise `threshold` (up to 254) or set `bold=1`.

### Domain Checker (`domains.py`)

Batch domain registration check via WHOIS with DNS fallback and optional OVH browser-verified availability.
- **Tool**: `check_domains` (up to 50 domains, 50+ TLDs incl. .com.cn and Chinese IDN)

### Google Ads Planner (`google_ads_planner.py`)

Keyword research, demand forecasting and budget planning against the Google Ads API (v25, REST).
- **Tools**:
  - `google_ads_keyword_metrics(keywords, geo_target_ids?, language_id?, network?, start_year_month?, end_year_month?)` — "how much demand is there for these exact terms?" Historical average monthly searches, competition band + index, top-of-page bid range, and the month-by-month volume series. Up to 10,000 keywords per call.
  - `google_ads_keyword_ideas(seed_keywords? | seed_url? | seed_site?, geo_target_ids?, language_id?, network?, limit?)` — "what else should we be targeting?" Expands one seed kind (max **20** seed keywords, or a single URL/site) into related keywords, each scored with an `opportunity_score` (volume ÷ competition), plus ready-made `top_by_volume`, `top_opportunities` and `easy_targets` (LOW/MEDIUM competition) buckets.
  - `google_ads_forecast_budget(keywords, match_type?, max_cpc_bid, daily_budget?, start_date?, end_date?, …)` — "what would this cost and what would we get?" Forward-looking forecast for a hypothetical campaign: clicks, cost, average CPC, conversions, average CPA.
  - `google_ads_budget_curve(keywords, match_type?, bids?, …)` — "where does spending more stop paying off?" Sweeps the forecast across a bid ladder (default `0.25 … 8.00`, max 10 points), computes marginal clicks / marginal cost / marginal cost-per-click between consecutive points, and picks the `best_efficiency_bid`.
  - `google_ads_seasonality(keywords, months_back?, geo_target_ids?, language_id?)` — "when in the year should we spend?" Up to 48 months of history normalised into a monthly index (1.0 = average month), with peak/trough months per keyword and in aggregate, plus year-over-year change.
- **Defaults**: geo `2840` (USA), language `1000` (English), network `GOOGLE_SEARCH`. UK is `2826`, Canada `2124`; Spanish `1003`, French `1002`.
- **Setup**: see [Google Ads Planner setup](#google-ads-planner-setup) below — credentials come from 1Password, not from files.
- **Caveats**:
  - **A Basic-access-tier developer token is required.** Test and Explorer tier tokens *cannot* call the keyword planning endpoints at all — they fail regardless of the account or the query. Apply for Basic access in your manager account's API Center.
  - **Keyword planning is rate-limited to 1 QPS per customer id.** The server serialises and paces every request itself, so a `google_ads_budget_curve` over six bids takes ~7s of wall clock by design. Don't fan these tools out in parallel.
  - **The v25 forecast returns no impressions and no CTR** — those fields were removed in v24. You get clicks, cost, average CPC, conversions and average CPA only; impression share and click-through rate cannot be derived from it.
  - **Absent numbers mean "no data", not zero.** Bid and competition fields are frequently missing for long-tail keywords and come back as `null` — don't read them as $0.00 or as zero competition.
- **Source**: `googleads.googleapis.com` REST endpoints under `customers/{id}:generateKeywordHistoricalMetrics` / `:generateKeywordIdeas` / `:generateKeywordForecastMetrics`.


## Google Ads Planner setup

The server reads four environment variables. **Nothing secret is stored in this repo** —
`.mcp.json` holds only 1Password *references*, and `op run` resolves them at launch.

| Variable | What it is |
| --- | --- |
| `GOOGLE_ADS_DEVELOPER_TOKEN` | API token from your manager account's API Center |
| `GOOGLE_ADS_SERVICE_ACCOUNT_KEY_B64` | base64 of the GCP service-account JSON key |
| `GOOGLE_ADS_LOGIN_CUSTOMER_ID` | manager (MCC) account id, digits only |
| `GOOGLE_ADS_CUSTOMER_ID` | the client account actually queried, digits only |
| `GOOGLE_ADS_API_VERSION` | optional, defaults to `v25` |

### 1. Get a developer token (Basic tier)

Only a **manager (MCC)** account can issue one. In the MCC: **Tools & Settings → Setup →
API Center** (<https://ads.google.com/aw/apicenter>). Apply for **Basic access** — new
tokens start on Explorer tier, which cannot call keyword planning at all, and Test tier
only works against test accounts. Approval is manual, typically 1–3 business days.

### 2. Create the service account

Auth is a GCP service account with **no impersonation** (no domain-wide delegation, no
Workspace admin needed). Create a JSON key, then add the service account's email as a
user on the Ads account under **Admin → Access and security**. Encode the key:

```bash
base64 -i service-account.json | tr -d '\n'
```

### 3. Store the credentials in 1Password

Create one API Credential item — the reference implementation lives in the `Startupkit`
vault as **Google Ads MCP**:

| Field | Contents |
| --- | --- |
| `credential` | developer token |
| `ads_manager_account_id` | MCC id |
| `company_account_id` | client account id |
| `Service Account` → `SERVICE_ACCOUNT_KEY_B64` | the base64 blob from step 2 |

> **Item titles must not contain `(`, `)` or other punctuation** — `op://` secret
> references reject them with `invalid character in secret reference`. Stick to letters,
> digits, spaces and hyphens.

Verify each reference resolves before wiring it up:

```bash
op read "op://Startupkit/Google Ads MCP/credential"
```

### 4. Launch

`.mcp.json` already wraps the server in `op run`, so Claude Code picks it up with no
further setup as long as you have an active `op` session (`op signin`):

```json
"args": ["-c", "op run --no-masking -- uv run --with fastmcp fastmcp run google_ads_planner.fastmcp.json"],
"env": {
  "GOOGLE_ADS_DEVELOPER_TOKEN": "op://Startupkit/Google Ads MCP/credential"
}
```

`--no-masking` is **required**, not optional. `op run` otherwise scans stdout and
replaces anything resembling a secret with `<concealed>` — and for a stdio MCP server
stdout *is* the JSON-RPC channel, so masking can corrupt the protocol stream.

To point at a different vault or item, edit the `op://` references in `.mcp.json`; they
are not secrets and are safe to commit.

### Running outside Claude Code

Same mechanism, driven from a dotenv file of references:

```bash
cat > .env.google-ads <<'EOF'
GOOGLE_ADS_DEVELOPER_TOKEN=op://Startupkit/Google Ads MCP/credential
GOOGLE_ADS_SERVICE_ACCOUNT_KEY_B64=op://Startupkit/Google Ads MCP/Service Account/SERVICE_ACCOUNT_KEY_B64
GOOGLE_ADS_LOGIN_CUSTOMER_ID=op://Startupkit/Google Ads MCP/ads_manager_account_id
GOOGLE_ADS_CUSTOMER_ID=op://Startupkit/Google Ads MCP/company_account_id
EOF

op run --no-masking --env-file=.env.google-ads -- mise run dev google_ads_planner.fastmcp.json
```

`.env*` is gitignored. If you genuinely cannot use 1Password, plain values in `.env`
work too — but never put them in `.mcp.json`, which *is* committed.


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
