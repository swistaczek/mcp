# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal MCP servers collection built with FastMCP framework. The project uses Python 3.13+, mise for task management, and uv for dependency management.

## Development Commands

```bash
# Setup and Dependencies
mise run install              # Install project dependencies via uv

# Testing
uv run pytest tests/ -v       # Run full test suite (73 passed, 14 skipped expected)
uv run pytest tests/test_domains.py::TestTLDExtraction -v  # Run specific test class
uv run pytest tests/ -k "chinese" -v  # Run tests matching pattern
uv run pytest tests/ -m "not integration" -v  # Skip integration tests

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
Multi-layer domain availability verification with optional OVH browser-based validation.

**Verification Layers:**
1. **Primary**: WHOIS protocol queries (port 43, 10s timeout)
2. **Fallback**: DNS record lookup when WHOIS fails (potential false positives)
3. **Optional OVH Verification**: Browser automation to catch false positives

**Key Features:**
- **Batch processing**: Up to 50 domains per request
- **TLD support**: 40+ TLDs including compound (.com.cn) and Chinese IDN domains
- **Pattern matching**: TLD-specific "not found" patterns in `NOT_FOUND_PATTERNS` dict
- **False positive detection**: OVH verification catches DNS-based false positives
- **Graceful degradation**: OVH verification failures don't break core functionality

**Tool:**
`check_domains(domains, verify_available?)` - Check domain registration status
- `domains: list[str]` - Domain names to check (1-50)
- `verify_available: bool = False` - Enable OVH verification for available results

**OVH Verification Module** (`ovh_verifier.py`):
- Browser-based domain availability check using OVH's web interface
- Bypasses OVH API bot protection via Playwright automation
- Provides pricing information for available domains
- Catches false positives: domains reported available by DNS but actually registered
- **Detects aftermarket/premium domains**: Distinguishes between standard registration and secondary market

**Aftermarket Detection:**
OVH displays domain type labels that indicate aftermarket/resale domains:
- **"Premium"** - Premium domains sold at higher prices
- **"Sprzedaż przez stronę trzecią"** - Third-party sales (external marketplace)

The verifier parses these labels and marks such domains as NOT available for standard registration.

**Verification Flow:**
```python
1. WHOIS query for each domain
2. If WHOIS fails → DNS fallback (⚠️ potential false positives)
3. If verify_available=True and domain appears available:
   → OVH browser verification checks availability
   → Detects aftermarket labels (Premium, third-party)
   → Aftermarket domains marked as NOT available
   → False positives flagged if OVH says registered
```

**Response Structure (with OVH and Aftermarket):**
```json
{
  "results": {
    "catch.dev": {
      "registered": false,
      "available": false,
      "method": "dns",
      "reason": "AFTERMARKET: Domain on secondary market (Premium) - 1 384,70 zł",
      "aftermarket": true,
      "aftermarket_type": "Premium",
      "aftermarket_price": "1 384,70 zł",
      "ovh_verification": {
        "ovh_available": false,
        "ovh_verified": true,
        "ovh_price": "1 384,70 zł",
        "ovh_price_type": "premium",
        "ovh_is_aftermarket": true,
        "ovh_aftermarket_type": "Premium",
        "ovh_error": null
      }
    },
    "erne.dev": {
      "registered": false,
      "available": true,
      "method": "dns",
      "reason": "OVH CONFIRMED available (23,39 zł)",
      "ovh_confirmed": true,
      "standard_price": "23,39 zł",
      "ovh_verification": {
        "ovh_available": true,
        "ovh_verified": true,
        "ovh_price": "23,39 zł",
        "ovh_price_type": "standard",
        "ovh_is_aftermarket": false,
        "ovh_aftermarket_type": null,
        "ovh_error": null
      }
    }
  },
  "available_domains": ["erne.dev"],
  "aftermarket_domains": ["catch.dev"],
  "registered_domains": [],
  "failed_domains": [],
  "ovh_verification": {
    "enabled": true,
    "confirmed_available": ["erne.dev"],
    "aftermarket": ["catch.dev"],
    "false_positives": [],
    "verification_failed": [],
    "duration_seconds": 18.36
  }
}
```

**Domain Categories:**
- `available_domains` - Standard registration at normal prices
- `aftermarket_domains` - Secondary market (Premium/third-party) with high prices
- `registered_domains` - Already registered, not for sale
- `failed_domains` - Check failed (unsupported TLD, network error)

**Dependencies:**
- `playwright` - Browser automation (optional, for OVH verification)
- `socket` - DNS resolution
- `asyncio` - Async WHOIS queries

**Known Limitations:**
- DNS fallback has high false positive rate for .dev/.app TLDs (domains can be registered without DNS records)
- OVH verification is slower (~2-3s per domain) but more accurate
- OVH verification requires Playwright installation: `pip install playwright && python -m playwright install chromium`
- Aftermarket prices can be 10-100x standard registration prices

### Tablica Rejestracyjna PL Architecture
Integration with Polish license plate reporting website (tablica-rejestracyjna.pl) for traffic violation reporting.

**Key Features:**
- **Fetch Comments**: Retrieve all comments/reports for a license plate
- **Submit Complaints**: Post new complaints with images and descriptions
- **Image Processing**: Automatic HEIC to JPEG conversion and downscaling
- **LLM Integration**: Tool prompts LLM to analyze images first, then generate Polish descriptions

**Tools:**
1. `fetch_comments(plate_number)` - Get existing reports for a plate
   - Parses HTML to extract comment text, ratings, timestamps
   - Returns structured data with all comments
   - Validates Polish plate format (e.g., WW12345, KR1234)

2. `submit_complaint(plate_number, violation_description, image_path, location?)` - Submit new report
   - **LLM Workflow**: Tool instructs calling LLM to:
     1. Analyze the violation image from pedestrian perspective
     2. Generate contextual Polish description
     3. Pass description to the tool
   - Converts HEIC images to JPEG automatically (requires pillow-heif)
   - Downscales large images (max 1920px, 85% quality)
   - Submits via multipart/form-data POST request
   - Anonymous submission (no authentication required)

**Image Processing Pipeline:**
```python
1. Detect format (HEIC, PNG, JPG, etc.)
2. Load with PIL (pillow-heif for HEIC)
3. Downscale if width or height > 1920px (maintains aspect ratio)
4. Convert to RGB if needed (RGBA → white background)
5. Compress as JPEG (quality 85)
6. Return bytes for upload
```

**Dependencies:**
- `aiohttp` - Async HTTP requests
- `beautifulsoup4` + `lxml` - HTML parsing
- `pillow-heif` - HEIC format support (optional but recommended)
- `Pillow` - Image processing
- `aioresponses` - HTTP mocking for async tests

**Usage Example:**
```python
# LLM workflow for submitting a complaint:
# 1. User provides: plate="WW12345", image="/path/to/violation.heic"
# 2. LLM analyzes image and generates Polish description
# 3. LLM calls: submit_complaint(
#      plate_number="WW12345",
#      violation_description="Samochód zaparkowany na chodniku...",
#      image_path="/path/to/violation.heic"
#    )
```

### EXIF Metadata Extractor Architecture
Extracts EXIF metadata from images (PNG, JPEG, HEIC) with focus on GPS data and automatic reverse geocoding to street addresses.

**Key Features:**
- **Universal Format Support**: PNG, JPEG, HEIC (via pillow-heif)
- **GPS Extraction**: Coordinates, altitude, speed, direction, timestamp, accuracy
- **Reverse Geocoding**: Automatic address lookup via Nominatim (OpenStreetMap)
- **Batch Processing**: Up to 50 images per request with progress reporting
- **No API Key Required**: Uses free Nominatim service

**Tool:**
`analyze_image_metadata(images, include_address, zoom_level, batch_size)` - Extract EXIF and geocode GPS coordinates
- `images: list[str]` - Image paths (1-50)
- `include_address: bool = True` - Enable reverse geocoding
- `zoom_level: int = 18` - Detail level (18=street, 16=area, 14=city)
- `batch_size: int = 10` - Concurrent processing limit

**Prioritized EXIF Fields:**
- **GPS Data**: Latitude/longitude (decimal degrees), altitude (meters), speed (km/h), direction (degrees), timestamp (UTC), accuracy (meters)
- **Timestamps**: DateTime, DateTimeOriginal, DateTimeDigitized
- **Camera Info**: Make, Model, Software, LensModel (when available)
- **Image Metadata**: Format, size, mode, orientation, resolution

**GPS Parsing:**
```python
# Extract GPS IFD (tag 0x8825) from EXIF
# Convert DMS (degrees/minutes/seconds) to decimal degrees
# Apply hemisphere references (N/S for latitude, E/W for longitude)
# Parse additional metadata (altitude, speed, direction, timestamp)
```

**Reverse Geocoding:**
- Service: Nominatim (OpenStreetMap API)
- Rate limit: 1 request/second (automatically enforced)
- Returns: Street, city, state, postal code, country, display name
- Handles failures gracefully (returns metadata without address)

**Output Structure:**
```json
{
  "image_path": {
    "format": "HEIF",
    "size": [4032, 3024],
    "exif": {
      "make": "Apple",
      "model": "iPhone 13 Pro",
      "datetime": "2025-07-27 09:55:45"
    },
    "gps": {
      "latitude": 52.408447,
      "longitude": 16.867817,
      "altitude_meters": 88.46,
      "speed_kmh": 0.18,
      "direction_degrees": 196.24,
      "timestamp_utc": "2025-07-27T07:55:44+00:00",
      "accuracy_meters": 3.54
    },
    "address": {
      "road": "Konstancji Łubieńskiej",
      "city": "Poznań",
      "country": "Polska",
      "postcode": "60-378"
    }
  }
}
```

**Dependencies:**
- `Pillow` - Image loading and EXIF extraction
- `pillow-heif` - HEIC format support
- `aiohttp` - Async HTTP for geocoding API

**Use Cases:**
- Photo geolocation analysis
- Travel photo mapping
- Privacy auditing (check for GPS data before sharing)
- Photo organization by location
- Forensic metadata extraction

### Plate Recognition Architecture
Analyzes traffic violation photos using Gemini's vision API to extract license plates and identify which vehicle is most likely committing a violation.

**Key Features:**
- **Plate Extraction**: Identifies all visible license plates in an image
- **Violation Detection**: Determines which vehicle is committing a traffic violation
- **Contextual Reasoning**: Provides one-sentence explanation from pedestrian perspective
- **Multi-Vehicle Support**: Handles images with multiple cars
- **Simple Single-Tool Design**: One tool for complete analysis

**Tool:**
`recognize_plates(image_path, model?)` - Analyze traffic photo for plates and violations
- `image_path: str` - Path to the violation photo
- `model: str = "gemini-flash-latest"` - Optional Gemini model override

**Violation Detection Focus:**
- Parking on sidewalks
- Blocking pedestrian crosswalks
- Illegal parking in restricted zones
- Blocking pedestrian access
- Other pedestrian-affecting violations

**Image Processing:**
```python
1. Load image from file path
2. Optimize for Gemini API (max 1920px, JPEG quality 85)
3. Convert RGBA/LA/L/P modes to RGB with white background
4. Send to Gemini with specialized prompt
5. Parse JSON response with plate numbers and reasoning
```

**Output Structure:**
```json
{
  "plates": ["WW12345", "KR1234"],
  "violation_vehicle": "WW12345",
  "reasoning": "Vehicle WW12345 is parked on the sidewalk, blocking pedestrian access.",
  "metadata": {
    "image_path": "/path/to/photo.jpg",
    "model": "gemini-flash-latest",
    "plates_count": 2
  }
}
```

**Gemini Prompt Strategy:**
- Analyze from pedestrian perspective
- Focus on violations affecting pedestrians
- Extract plates as they appear (with spaces/dashes)
- Return structured JSON with plates array, violation_vehicle, and reasoning
- Handle cases with no plates or no violations

**Dependencies:**
- `google-generativeai` - Gemini API client
- `Pillow` - Image loading and optimization
- `fastmcp` - FastMCP framework

**Use Cases:**
- Automated traffic violation reporting
- Batch analysis of violation photos
- Integration with complaint submission systems (e.g., Tablica MCP)
- Pedestrian advocacy and documentation

**Integration with Tablica MCP:**
Can be used together with Tablica MCP server for complete workflow:
1. Use `recognize_plates` to analyze photo and identify violating vehicle
2. Use `submit_complaint` from Tablica MCP to report the violation

### Poznań Events Architecture
Scraper for the City of Poznań events calendar (`poznan.pl/mim/events/`) — the official municipal "Co? Gdzie? Kiedy?" listing.

**Data Source:**
- Listing XHR: `GET /mim/events/events.html?co=list&lang=pl&category={ID}&p={PAGE}` — returns the same `.event-box` HTML fragment as the homepage. `category` is empty for "all" or a numeric id (e.g. `214` for Sport). `p` is zero-indexed; each page = 20 events. High `p` returns 404 with empty body.
- Detail pages: `/mim/events/{slug},{numeric_id}.html` — clean structured selectors (`.event-print-date`, `.event-print-place`, `.event-print-categories`, `.event-print-describe`) plus `og:title` / `og:description` meta tags.

**Tools:**
1. `list_events(category?, page?, pages?)` — paginated event listing
   - `category: str?` — numeric id (empty / null for all)
   - `page: int = 0` — zero-indexed start page
   - `pages: int = 1` — consecutive pages to fetch in one call (capped at 5)
   - Returns: `{count, events: [{event_id, title, date (ISO), date_raw, time (HH:MM), place, categories: [{id, name}], thumbnail_url, detail_url}], pages: [{page, count}]}`
2. `get_event(event_ref)` — single-event detail
   - `event_ref: str` — numeric id, `slug,id`, `/mim/events/...` path, or full URL
   - Returns: title, ISO date, time, place, categories, `short_description` (from `og:description`), `description` (full prose with UI controls stripped), `image_url`
3. `list_event_categories` — category catalog
   - Returns: `[{id, slug, name}]` sorted alphabetically by name

**Parser Strategy:**
- List card: relies on `.event-box`, `.description-event-title-link[href]` for id/url, two `<time>` tags by position (date first, time second), `.description-event-place`, `.description-event-category-link[data-id]`, `.image-events img[data-src]`.
- Detail page: prefers `og:title` / `og:description` for canonical text; reads `.event-print-date` / `.event-print-place` for clean fields; the `.event-print-describe` block is decomposed (banners, share buttons, QR-code UI) before text extraction.
- Date conversion: Polish `DD.MM.YYYY` → ISO `YYYY-MM-DD`. Time padded to `HH:MM`.
- Event references accept numeric id, `slug,id[.html]`, absolute path, or full URL.

**Dependencies:**
- `aiohttp` — async HTTP
- `beautifulsoup4` + `lxml` — HTML parsing

**Use Cases:**
- "What's happening in Poznań this weekend?" — `list_events(category=None, pages=2)`
- "Sport events in Poznań" — `list_events(category="214")`
- "Tell me about event 179433" — `get_event("179433")`

### Label Printer Architecture
Prints courier labels (InPost, DPD, DHL…) on direct-thermal label printers attached to the local machine. Ships as both a CLI (`python label_printer.py …`) and an MCP server from one module — `main()` sits behind an `if __name__ == "__main__"` guard so `fastmcp run` can import the module for its `mcp` object without argparse firing.

**Verified against:** Zebra TLP2844 (USB, EPL2, 203 dpi) with 4×6" direct-thermal stock and real InPost labels.

**Why the label is rendered here instead of by CUPS:**
A thermal head is strictly 1-bit — a dot is burnt or it is not. Letting CUPS rasterise a PDF at 203 dpi anti-aliases thin strokes into grey, which `rastertolabel` then dithers into sparse dots; the print comes out visibly faint and barcodes scan poorly. Instead:

1. `pdftoppm -gray -r 812` renders at 4× the device resolution
2. A **box filter** downscales to the exact dot grid — each output dot is the true area average of the pixels it covers
3. A hard threshold (default 200) maps that average to pure black or white, which *thickens* strokes rather than thinning them
4. The result is written as a 1-bit PDF whose MediaBox equals the label exactly (4×6 in → 288×432 pt → 812×1218 dots), so CUPS has no reason to rescale

Empirically this was the difference between "readable but pale" and solid black; `Darkness=30` alone was not enough.

**macOS raw-queue restriction:**
`lpadmin -m raw` fails on macOS 26 with `Raw queues are no longer supported on macOS`, which rules out hand-rolled EPL2 `GW` bitmaps over a raw queue. The server creates a **driver-backed** queue from the bundled sample drivers instead — `drv:///sample.drv/zebraep2.ppd` for EPL2-era Zebras, `zebra.ppd` for ZPL, `dymo.ppd` for Dymo. `lpadmin` warns that drivers are deprecated but still works.

**Printer discovery:**
`lpinfo -v` lists reachable devices (including ones with no queue yet), `lpstat -v`/`-p` list existing queues and their state; the two are merged on device URI so an installed printer appears once. Device URIs are matched against `THERMAL_MAKES` (Zebra, Dymo, TSC, Godex, Citizen, Bixolon, Sato, Argox, Intermec, Honeywell, Brother QL, Toshiba TEC, Seiko) so a thermal printer is told apart from an office laser. With exactly one detected printer the `printer` argument can be omitted everywhere.

Zebra language detection keys off the model number: `TLP2844`/`LP2844`/`GK420`… are EPL2, but a `-Z` suffix means ZPL firmware on the same model number, so the regex excludes it.

**Option filtering:**
CUPS silently ignores an unsupported `-o`, so options are checked against `lpoptions -p <queue> -l` before use. `Darkness` and `zePrintRate` exist on Zebra PPDs but not on Dymo's — they are simply omitted there rather than assumed.

**Tuned defaults** (established by printing real InPost labels and comparing):
| Setting | Value | Reason |
|---|---|---|
| `threshold` | 200 | Anti-aliased stroke edges survive as solid black |
| `margin_pt` | 8 | InPost artwork is 297×435 pt — wider than the 4.09" head; a full-bleed fit clips the printed frame |
| `darkness` | 30 | Max burn energy; direct-thermal stock needs the top of the range |
| `speed` | 1 in/s | Slowest travel = longest dwell per dot = darkest print |
| `dpi` | 203 | TLP2844 head resolution |

**Tools:**
- `list_thermal_printers` — detected printers, whether each has a queue, its language and state
- `print_label(file_path, printer?, label_size?, copies?, darkness?, speed?, threshold?, bold?, dry_run?)` — renders and submits; creates the CUPS queue on first use
- `print_qr_code(data, printer?, module_dots?, error_correction?, caption?)` — encodes any value; routes to whichever transport is present
- `get_printer_status(printer?)` — queue state + pending jobs
- `cancel_print_jobs(printer?, job_id?)` — clear the queue

**CLI:** `detect`, `status`, `install`, `render` (write the 1-bit PDF without printing — useful for inspecting output), `print`, `qr`, `cancel`. `--json` works on either side of the verb (the subparser copy uses `default=argparse.SUPPRESS` so it doesn't clobber the top-level value).

**Label sizes:** presets (`4x6`, `4x4`, `4x3`, `2x1`, `a6`) or explicit dimensions (`100x150mm`, `4x6in`, `288x432pt`); a bare `WxH` is read as inches, matching how stock is sold. Zebra PPDs name media `w<width_pt>h<height_pt>`; when no listed size matches, a `Custom.<w>x<h>` fallback is used.

**Orientation:** artwork whose orientation disagrees with the stock is rotated a quarter turn before fitting, so a landscape label still fills a portrait 4×6.

**Second transport: ESC/POS over libusb** (`escpos_printer.py`)
Cheap receipt printers (Winbond `0416:5011` and relatives, sold as Xprinter/Gprinter/Zjiang or unbranded) expose a USB printer-class interface but speak **ESC/POS**, which no bundled CUPS driver understands — CUPS lists the device and then has nothing to send it. These are driven directly over libusb instead. Discovery is by USB vendor/product id, since the devices identify themselves only as `Generic Bulk Device` with a placeholder serial, leaving the name-matching heuristic nothing to grip.

Three constraints were established by trial against real hardware and are enforced in the module rather than left to callers:

1. **Never exceed the head width.** An oversized raster does not error — it wedges the firmware until the printer is power-cycled. `raster_payload` raises instead, and `qr_image` shrinks the symbol to fit.
2. **Send a page in exactly one write.** Banding the raster looks safer and is not: bulk USB already provides flow control (a full buffer NAKs and the host controller retries in hardware, inside the single transfer), whereas separate transfers leave gaps this firmware reads as the end of the raster, after which it errors and wedges. A 27 KB label sent as 24 paced bands died at band 5; the same bytes in one `GS v 0` command printed completely.
3. **Rasterise QR codes.** The native `GS ( k` QR command is orders of magnitude cheaper on the wire (98 bytes vs 12 KB) but this firmware silently ignores it and prints nothing, so QR codes go through `segno` → bitmap → raster like any other image.

The interface is also claimed explicitly (`claim_interface`): CUPS enumerates the same device and its usb backend probes periodically, which a short write survives by luck but a multi-second raster transfer does not.

Note that `EscPosDevice` and the CUPS path share only the rendering idea — supersample, box-filter, threshold — not the code. That split is deliberate: the renderer produces a 1-bit bitmap, and the transport decides whether to wrap it in `GS v 0` or hand it to `lp`.

**Verified against:** Winbond `0416:5011`, 58 mm continuous roll, 48 mm/384-dot printable width at 203 dpi (measured by printing a millimetre ruler, since the device reports nothing).

**Known limitations:**
- ESC/POS printers give no usable feedback: the IN endpoint is protocol 1 (unidirectional), so status queries go unanswered — paper-out and head-up cannot be detected
- A wedged ESC/POS printer cannot be recovered over USB; `reset()` times out and only a power cycle helps
- Base TLP2844 has a tear bar only — cutter and peeler are optional factory modules, and the CUPS EPL2 PPD exposes no cutter option (that would need an EPL2 `OC` command over a raw path macOS blocks)
- 300 dpi thermal models need `DEFAULT_DPI` adjusted; nothing auto-detects head resolution
- Requires `pdftoppm` (poppler) for PDF input; `bilevel=False` falls back to the CUPS raster path

**Dependencies:**
- `Pillow` — rasterised-page fitting, thresholding, 1-bit PDF output
- `pdftoppm` (poppler, external) — PDF rasterisation
- CUPS command-line tools (`lp`, `lpstat`, `lpinfo`, `lpoptions`, `lpadmin`, `cancel`)

**Use Cases:**
- "Print this InPost label" — `print_label(file_path="~/Downloads/label.pdf")`
- "Print 3 copies darker" — `print_label(file_path=…, copies=3, threshold=235, bold=1)`
- "Why didn't my label come out?" — `get_printer_status()`

### Image Descriptions Architecture
Generates accessible descriptions for images and GIFs using Gemini LLM.

**Key Features:**
- **Two description types**: Concise alt text vs detailed accessible descriptions
- **SHA256 checksums**: Raw input file checksums for client-side caching
- Adaptive image resizing (text-heavy detection)
- GIF support via FFmpeg conversion to MP4 + Gemini File API
- Batch processing with configurable sizes
- Context-aware descriptions

**Tool:**
`generate_image_descriptions(images, type?, context?, batch_size?, model?)` - Generate descriptions
- `images: list[str]` - Image/GIF paths (1-20)
- `type: str = "alt"` - Description type:
  - `"alt"`: Concise alt text (50-125 chars) - suitable for HTML alt attributes
  - `"description"`: Adaptive descriptions - auto-detects tutorial content for verbose step-by-step narration
- `context: str?` - Document context for more relevant descriptions
- `batch_size: int = 5` - Images per Gemini request (1-10)
- `model: str?` - Gemini model override (default: gemini-flash-latest)

**Description Types:**
- **"alt" (default)**: Concise, focused descriptions for HTML alt attributes (50-125 chars)
- **"description"**: Adaptive descriptions that auto-detect content type:
  - **For UI tutorials/screencasts**: Detailed step-by-step narration with no length limit
    - "Click the 'Save' button", "Select 'Export' from the dropdown"
    - Describes each action: what, where, and result
    - Includes text typed, values entered, transitions
  - **For non-tutorial content**: Detailed but concise (150-300 chars)
    - Spatial layout, colors, textures
    - Visible text, actions, expressions
    - Mood, context, and purpose

**Output Structure:**
```json
{
  "descriptions": {
    "image1.png": "Generated description text"
  },
  "checksums": {
    "image1.png": "sha256_hex_string_64_chars"
  },
  "stats": {
    "total_images": 1,
    "successful": 1,
    "failed": 0,
    "duration_seconds": 2.5
  },
  "metadata": {
    "model": "gemini-flash-latest",
    "description_type": "alt",
    "context_provided": false,
    "checksum_algorithm": "sha256"
  }
}
```

**Dependencies:**
- `google-generativeai`, `google-genai`, `Pillow`
- FFmpeg (optional, for GIF support): `brew install ffmpeg`

## Testing Strategy

Tests are organized by functionality:

**Domain Checker Tests** (`tests/test_domains.py`):
- `TestTLDExtraction` - TLD parsing logic
- `TestChineseCharacterDetection` - Unicode character detection
- `TestNotFoundPatterns` - WHOIS response parsing
- `TestWHOISServerMapping` - Server configuration validation
- `TestOVHVerificationIntegration` - OVH verification response structures, false positive detection, summary formatting
- Integration tests marked with `@pytest.mark.skip` (require mocking)

**Tablica Tests** (`tests/test_tablica.py`):
- `TestPlateValidation` - Polish license plate format validation
- `TestImageOptimization` - Image downscaling and conversion
- `TestHTMLParsing` - Comment extraction from HTML
- `TestHEICSupport` - HEIC format detection
- `TestImageFormats` - PNG, JPEG, grayscale handling
- Integration tests marked with `@pytest.mark.skip` (require HTTP mocking)

**Gemini Image Description Tests** (`tests/test_gemini_alt.py`):
- `TestImageOptimization` - Adaptive image resizing
- `TestContextLoading` - Document context handling
- `TestPromptGeneration` - Prompt creation for alt/description modes
- `TestGenerateImageDescriptionsTool` - Main tool functionality
- `TestGifSupport` - GIF detection (magic bytes, extension)
- `TestGifConversion` - FFmpeg GIF→MP4 conversion
- `TestGifUploadAndGeneration` - Mocked Gemini File API
- `TestGifIntegration` - Full GIF pipeline (requires FFmpeg + GEMINI_API_KEY)
- Integration tests (require GEMINI_API_KEY)

**EXIF Extractor Tests** (`tests/test_exif_extractor.py`):
- `TestConvertToDegrees` - GPS coordinate conversion (DMS to decimal)
- `TestExtractExifData` - EXIF extraction from PNG, JPEG, HEIC
- `TestParseGpsData` - GPS metadata parsing and coordinate extraction
- `TestReverseGeocode` - Nominatim geocoding with mocked responses
- `TestAnalyzeImageMetadataTool` - Main tool functionality (single/batch)
- Integration tests marked with `@pytest.mark.integration` (hit real Nominatim API)

**Poznań Events Tests** (`tests/test_poznan_events.py`):
- `TestPolishDateConversion` / `TestTimeNormalization` — date/time format helpers
- `TestExtractEventId` — id extraction from canonical URLs
- `TestNormalizeCategory` / `TestResolveEventUrl` — input coercion edge cases
- `TestListParser` — fixture-driven (real homepage snapshot, 20 events parsed)
- `TestCategoriesParser` — homepage category nav extraction
- `TestDetailParser` — fixture-driven event detail page parsing
- `TestLiveEndpoint` (marked `@pytest.mark.integration`) — live calls against poznan.pl

**Label Printer Tests** (`tests/test_label_printer.py`):
- `TestClassify` / `TestModelFromUri` — thermal-printer identification from device URIs, including the `-Z` ZPL-variant exclusion and rejection of office printers
- `TestCupsParsers` — `lpinfo -v`, `lpstat -v`, `lpstat -p`, `lpoptions -l` output parsing from captured real output
- `TestDiscovery` — device/queue merge, single-printer auto-selection, ambiguous and missing-printer errors
- `TestQueueName` — CUPS-legal name sanitisation, queue reuse, `lpadmin` failure reporting
- `TestLabelSize` / `TestPageSizeOption` — size parsing (presets, mm/in/pt) and PPD keyword selection with `Custom.` fallback
- `TestFitGeometry` — scaling, centring, margin, and quarter-turn rotation
- `TestBilevelConversion` — thresholding (not dithering), margin border, rotation onto portrait stock, bold dilation
- `TestRenderLabel` — end-to-end rendering of the bundled InPost fixture to the 812×1218 device grid
- `TestPrintFile` — `lp` command assembly, option filtering against the PPD, job-id parsing, failure handling
- `TestCli` — exit codes and `--json` output
- Integration tests marked `@pytest.mark.integration` — require an attached printer

CUPS commands are faked from captured output; `pdftoppm` deliberately is **not** — the test double delegates any non-CUPS command to the real `subprocess`, so the rendering half of the pipeline stays honest. Tests needing poppler skip cleanly when it is absent.

**ESC/POS Tests** (`tests/test_escpos_printer.py`):
- `TestDeviceTable` — USB id table and dots→mm conversion
- `TestFindPrinters` — known/unknown device filtering, unreadable serial
- `TestRasterPayload` — `GS v 0` header encoding, **single command per image**, inverted bit polarity, byte padding, oversized/overtall/greyscale rejection
- `TestFitToHead` — box-filter scaling and thresholding
- `TestQrImage` — fits the paper width, module size clamped rather than overflowing, quiet zone, input validation
- `TestTextPayload` — alignment codes, CP852 Polish characters, graceful degradation
- `TestDevice` — interface claim, one-write-per-page, halt-clearing retry, power-cycle hints, cleanup on close
- Integration test marked `@pytest.mark.integration` — needs an attached ESC/POS printer

The USB layer is faked with `MagicMock`; `test_page_goes_out_as_one_transfer` pins the single-write invariant, which is the one behaviour that cannot be relaxed without wedging real hardware.

**Plate Recognition Tests** (`tests/test_plate_recognition.py`):
- `TestImageOptimization` - Image downscaling and RGB conversion
- `TestPromptGeneration` - Prompt creation validation
- `TestJSONParsing` - Response parsing (with/without violations)
- `TestMarkdownCodeBlockRemoval` - Cleaning Gemini markdown responses
- `TestRecognizePlatesTool` - Tool functionality (requires mocking)
- Integration tests marked with `@pytest.mark.integration` (require GEMINI_API_KEY)

Test fixtures:
- `tests/fixtures/whois_responses.json` - Sample WHOIS responses
- `tests/fixtures/*.png` - Test images for alt tag generation
- `tests/fixtures/example.gif` - Animated GIF for GIF description testing
- `tests/fixtures/IMG_5134.heic` - iPhone photo with GPS data (Poznań, Poland)
- `tests/fixtures/IMG_2852.heic` - iPhone photo without GPS data
- `tests/fixtures/poznan_events_list.html` - Snapshot of the Poznań events homepage (20 cards + category nav)
- `tests/fixtures/inpost_label_4x6.pdf` - InPost's public sample courier label (297×435 pt) used to exercise the rendering pipeline; contains no personal data
- `tests/fixtures/poznan_event_detail.html` - Snapshot of a single Poznań event detail page

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

**GitHub CLI (gh) debugging:**
```bash
# View GitHub Actions run logs (detailed output)
gh run view <RUN_ID> --log

# Extract test failures from logs
gh run view <RUN_ID> --log | grep -A 50 "FAILED\|ERROR\|test session starts" | tail -100

# Watch running workflow
gh run watch <RUN_ID>

# List recent runs
gh run list --limit 10

# Rerun failed jobs
gh run rerun <RUN_ID>
```