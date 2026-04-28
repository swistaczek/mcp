"""FastMCP Server for Polish supermarket promotional flyers (gazetki).

V1 supports two chains: Lidl and Biedronka. Provides tools to list
currently-available flyers and to download them as PDFs.

- Lidl: server-rendered listing + JSON API → direct PDF download.
- Biedronka: server-rendered listing → press page → leaflet JSON API →
  per-page PNG images that we assemble into a PDF locally with Pillow.

Caching: PDFs are cached on disk under ~/.cache/mcp-gazetki/{chain}/{flyer_id}.pdf
with a sibling meta.json. Override path via env var GAZETKI_CACHE_DIR.
TTL: serve from cache while today <= valid_to OR fetched_at within 24h.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import io
import json
import os
import re
import time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from fastmcp import Context, FastMCP
from fastmcp.tools.tool import ToolResult
from PIL import Image
from pydantic import Field

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

mcp = FastMCP("Gazetki")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)

LIDL_LIST_URL = "https://www.lidl.pl/c/nasze-gazetki/s10008614"
LIDL_API_URL = "https://endpoints.leaflets.schwarz/v4/flyer"
LIDL_API_ORIGIN = "https://lidl.leaflets.schwarz"

BIEDRONKA_BASE = "https://www.biedronka.pl"
BIEDRONKA_LIST_URL = f"{BIEDRONKA_BASE}/pl/gazetki"
BIEDRONKA_LEAFLET_API = "https://leaflet-api.prod.biedronka.cloud/api/leaflets"

CACHE_TTL_SECONDS = 24 * 3600
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=60)
DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=180)
# Some upstream sites (notably lidl.pl) return CSP headers far larger than
# aiohttp's 8190-byte default. Bump generously.
MAX_FIELD_SIZE = 64 * 1024
MAX_LINE_SIZE = 64 * 1024


def _new_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        max_field_size=MAX_FIELD_SIZE,
        max_line_size=MAX_LINE_SIZE,
    )


# ---------------------------------------------------------------------------
# Throttle (per-host, monotonic)
# ---------------------------------------------------------------------------

_LAST_REQUEST: dict[str, float] = {}
_THROTTLE_LOCK = asyncio.Lock()
_MIN_INTERVAL_SECONDS = 1.0


async def _throttle(host: str, min_interval: float = _MIN_INTERVAL_SECONDS) -> None:
    """Ensure at least `min_interval` seconds between requests to the same host."""
    async with _THROTTLE_LOCK:
        last = _LAST_REQUEST.get(host, 0.0)
        now = time.monotonic()
        wait = (last + min_interval) - now
        if wait > 0:
            await asyncio.sleep(wait)
        _LAST_REQUEST[host] = time.monotonic()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    override = os.environ.get("GAZETKI_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "mcp-gazetki"


def _flyer_cache_paths(chain: str, flyer_id: str) -> tuple[Path, Path]:
    base = _cache_dir() / chain
    base.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", flyer_id)
    return base / f"{safe_id}.pdf", base / f"{safe_id}.meta.json"


def _read_meta(meta_path: Path) -> Optional[dict]:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_cache_fresh(meta: dict) -> bool:
    """Cache fresh if today <= valid_to (when known) OR fetched_at within 24h."""
    valid_to = meta.get("valid_to")
    if valid_to:
        try:
            vt = date.fromisoformat(valid_to)
            if date.today() <= vt:
                return True
        except ValueError:
            pass
    fetched_at = meta.get("fetched_at")
    if fetched_at:
        try:
            ts = datetime.fromisoformat(fetched_at)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age < CACHE_TTL_SECONDS:
                return True
        except ValueError:
            pass
    return False


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pdf_page_count(path: Path) -> Optional[int]:
    if not PYPDF_AVAILABLE:
        return None
    try:
        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lidl
# ---------------------------------------------------------------------------

# `<a class="flyer" href="..." data-track-id="..." data-track-name="...">`
_LIDL_FLYER_RE = re.compile(
    r'<a\b[^>]*\bclass="flyer"[^>]*?>',
    re.IGNORECASE | re.DOTALL,
)
_LIDL_HREF_RE = re.compile(r'\bhref="([^"]+)"', re.IGNORECASE)
_LIDL_DATA_ID_RE = re.compile(r'\bdata-track-id="([^"]+)"', re.IGNORECASE)
_LIDL_DATA_NAME_RE = re.compile(r'\bdata-track-name="([^"]+)"', re.IGNORECASE)
# slug = path segment after /gazetki/
_LIDL_SLUG_RE = re.compile(r"/gazetki/([^/]+)/")


def _parse_lidl_list(html_text: str) -> list[dict]:
    """Return list of {flyer_id, slug, name, detail_url} parsed from Lidl listing HTML.

    Matches anchors that have BOTH class="flyer" AND data-track-id (both orderings).
    """
    flyers: list[dict] = []
    seen: set[str] = set()
    # Match <a ...> tags with attributes in any order
    anchor_re = re.compile(r"<a\b[^>]+>", re.IGNORECASE)
    for match in anchor_re.finditer(html_text):
        tag = match.group(0)
        if 'class="flyer"' not in tag:
            continue
        href_m = _LIDL_HREF_RE.search(tag)
        id_m = _LIDL_DATA_ID_RE.search(tag)
        name_m = _LIDL_DATA_NAME_RE.search(tag)
        if not (href_m and id_m):
            continue
        href = href_m.group(1)
        flyer_id = id_m.group(1)
        if flyer_id in seen:
            continue
        seen.add(flyer_id)
        slug_m = _LIDL_SLUG_RE.search(href)
        slug = slug_m.group(1) if slug_m else None
        if not slug:
            continue
        name = html.unescape(name_m.group(1)) if name_m else ""
        flyers.append(
            {
                "flyer_id": flyer_id,
                "slug": slug,
                "name": name,
                "detail_url": href,
            }
        )
    return flyers


async def _fetch_lidl_list(
    session: aiohttp.ClientSession, ctx: Optional[Context] = None
) -> list[dict]:
    host = urlparse(LIDL_LIST_URL).hostname or ""
    await _throttle(host)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
    }
    async with session.get(LIDL_LIST_URL, headers=headers, timeout=DEFAULT_TIMEOUT) as r:
        r.raise_for_status()
        text = await r.text()
    return _parse_lidl_list(text)


def _parse_lidl_detail(payload: dict) -> dict:
    """Pull useful fields out of the Lidl flyer JSON envelope."""
    flyer = payload.get("flyer") or {}
    return {
        "flyer_id": flyer.get("id"),
        "title": flyer.get("title"),
        "name": flyer.get("name"),
        "valid_from": flyer.get("startDate"),
        "valid_to": flyer.get("endDate"),
        "pdf_url": flyer.get("pdfUrl"),
        "thumbnail_url": flyer.get("thumbnailUrl"),
        "file_size": flyer.get("fileSize"),
    }


async def _fetch_lidl_detail(
    session: aiohttp.ClientSession, slug: str, ctx: Optional[Context] = None
) -> dict:
    host = urlparse(LIDL_API_URL).hostname or ""
    await _throttle(host)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Origin": LIDL_API_ORIGIN,
        "Referer": f"{LIDL_API_ORIGIN}/",
    }
    params = {"flyer_identifier": slug, "region_id": "0", "region_code": "0"}
    async with session.get(LIDL_API_URL, headers=headers, params=params,
                            timeout=DEFAULT_TIMEOUT) as r:
        r.raise_for_status()
        data = await r.json()
    return _parse_lidl_detail(data)


async def _download_lidl_pdf(
    session: aiohttp.ClientSession, pdf_url: str, ctx: Optional[Context] = None
) -> bytes:
    host = urlparse(pdf_url).hostname or ""
    await _throttle(host)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,*/*;q=0.8",
        "Referer": LIDL_API_ORIGIN + "/",
    }
    chunks: list[bytes] = []
    async with session.get(pdf_url, headers=headers, timeout=DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        async for chunk in r.content.iter_chunked(64 * 1024):
            chunks.append(chunk)
            if ctx is not None and len(chunks) % 50 == 0:
                await ctx.report_progress(
                    sum(len(c) for c in chunks), None, "Downloading PDF"
                )
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Biedronka
# ---------------------------------------------------------------------------

# Each flyer card: <div class="adwrapper ad_gazetka" data-slot-id="2026_T18A_..." ...>
#   <a class="page-slot-columns" href="https://www.biedronka.pl/pl/press,id,XYZ,title,SLUG[#page=1]">
_BIEDRONKA_LINK_RE = re.compile(
    r'press(?:adult)?,id,(?P<id>[a-z0-9]+),title,(?P<slug>[a-z0-9-]+)',
    re.IGNORECASE,
)
_BIEDRONKA_UUID_RE = re.compile(
    r'galleryLeaflet\.init\("([0-9a-f-]{36})"\)',
    re.IGNORECASE,
)
# data-slot-id like 2026_T18A_..., 2026_T18_..., 2026-T18A-...
_BIEDRONKA_WEEK_RE = re.compile(
    r'(?P<year>20\d{2})[_-]T(?P<week>\d{1,2})[A-Za-z]?',
)


def _parse_biedronka_list(html_text: str) -> list[dict]:
    """Parse Biedronka gazetki listing HTML.

    Each flyer is a `<div class="adwrapper ad_gazetka">` whose first child is
    an `<a class="page-slot-columns" href="...press,id,XYZ,title,SLUG">`.
    """
    soup = BeautifulSoup(html_text, "lxml")
    flyers: list[dict] = []
    seen: set[str] = set()
    for wrapper in soup.find_all("div", class_="ad_gazetka"):
        anchor = wrapper.find("a", class_="page-slot-columns")
        if not anchor or not anchor.get("href"):
            continue
        href = anchor["href"]
        # Reject external links (e.g., glovoapp.com) — gazetka cards always
        # link to www.biedronka.pl/pl/press[adult]
        m = _BIEDRONKA_LINK_RE.search(href)
        if not m:
            continue
        press_id = m.group("id")
        slug = m.group("slug")
        if press_id in seen:
            continue
        seen.add(press_id)

        slot_id = wrapper.get("data-slot-id") or ""
        valid_from_hint = None
        wm = _BIEDRONKA_WEEK_RE.search(slot_id)
        if wm:
            valid_from_hint = f"week {wm.group('week')}/{wm.group('year')}"

        # Thumbnail = first image inside the anchor
        thumb = None
        img = anchor.find("img")
        if img and img.get("src"):
            thumb = img["src"]

        # Strip any trailing fragment from detail URL (#page=N)
        detail_url = href.split("#", 1)[0]

        # Title heuristic: from the slug (kebab → spaces, capitalize words)
        title = slug.replace("-", " ").strip()

        flyers.append(
            {
                "flyer_id": press_id,
                "slug": slug,
                "title": title,
                "name": title,
                "detail_url": detail_url,
                "thumbnail_url": thumb,
                "slot_id": slot_id,
                "valid_from_hint": valid_from_hint,
                "is_adult": "pressadult" in href,
            }
        )
    return flyers


async def _fetch_biedronka_list(
    session: aiohttp.ClientSession, ctx: Optional[Context] = None
) -> list[dict]:
    host = urlparse(BIEDRONKA_LIST_URL).hostname or ""
    await _throttle(host)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
    }
    async with session.get(BIEDRONKA_LIST_URL, headers=headers,
                            timeout=DEFAULT_TIMEOUT) as r:
        r.raise_for_status()
        text = await r.text()
    return _parse_biedronka_list(text)


def _extract_biedronka_uuid(html_text: str) -> Optional[str]:
    m = _BIEDRONKA_UUID_RE.search(html_text)
    return m.group(1) if m else None


async def _fetch_biedronka_uuid_from_press(
    session: aiohttp.ClientSession, detail_url: str, ctx: Optional[Context] = None
) -> Optional[str]:
    host = urlparse(detail_url).hostname or ""
    await _throttle(host)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
    }
    async with session.get(detail_url, headers=headers, timeout=DEFAULT_TIMEOUT) as r:
        r.raise_for_status()
        text = await r.text()
    return _extract_biedronka_uuid(text)


def _parse_biedronka_leaflet(payload: dict) -> list[str]:
    """Return ordered list of one image URL per page (mobile preferred)."""
    pages = payload.get("images_mobile") or []
    pages = sorted(pages, key=lambda p: p.get("page", 0))
    urls: list[str] = []
    for p in pages:
        img = p.get("image")
        if isinstance(img, str) and img:
            urls.append(img)
    if urls:
        return urls
    # Fallback: pick the highest-resolution (last) image from each desktop page.
    desktop_pages = sorted(
        payload.get("images_desktop") or [], key=lambda p: p.get("page", 0)
    )
    for p in desktop_pages:
        imgs = [u for u in (p.get("images") or []) if isinstance(u, str) and u]
        if imgs:
            urls.append(imgs[-1])
    return urls


async def _fetch_biedronka_leaflet_api(
    session: aiohttp.ClientSession, leaflet_uuid: str, ctx: Optional[Context] = None
) -> list[str]:
    url = f"{BIEDRONKA_LEAFLET_API}/{leaflet_uuid}"
    host = urlparse(url).hostname or ""
    await _throttle(host)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Origin": BIEDRONKA_BASE,
        "Referer": f"{BIEDRONKA_BASE}/",
    }
    async with session.get(url, headers=headers, params={"ctx": "web"},
                            timeout=DEFAULT_TIMEOUT) as r:
        r.raise_for_status()
        data = await r.json()
    return _parse_biedronka_leaflet(data)


def _assemble_biedronka_pdf(pages_bytes: list[bytes]) -> bytes:
    """Combine downloaded page images into a single PDF."""
    if not pages_bytes:
        raise ValueError("Cannot assemble PDF from zero pages")
    images = []
    for raw in pages_bytes:
        img = Image.open(io.BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)
    buf = io.BytesIO()
    images[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=images[1:],
        resolution=150.0,
    )
    return buf.getvalue()


async def _download_biedronka_pages(
    session: aiohttp.ClientSession,
    image_urls: list[str],
    ctx: Optional[Context] = None,
    concurrency: int = 4,
) -> list[bytes]:
    sem = asyncio.Semaphore(concurrency)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
        "Referer": f"{BIEDRONKA_BASE}/",
    }
    results: list[Optional[bytes]] = [None] * len(image_urls)
    completed = 0

    async def fetch(idx: int, url: str) -> None:
        nonlocal completed
        async with sem:
            async with session.get(url, headers=headers,
                                    timeout=DOWNLOAD_TIMEOUT) as r:
                r.raise_for_status()
                results[idx] = await r.read()
        completed += 1
        if ctx is not None:
            await ctx.report_progress(
                completed, len(image_urls), f"Downloaded {completed}/{len(image_urls)} pages"
            )

    await asyncio.gather(*(fetch(i, u) for i, u in enumerate(image_urls)))
    out: list[bytes] = []
    for r in results:
        if r is None:
            raise RuntimeError("Page download returned None")
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# High-level chain helpers
# ---------------------------------------------------------------------------

async def _list_lidl(ctx: Optional[Context] = None) -> list[dict]:
    out: list[dict] = []
    async with _new_session() as session:
        items = await _fetch_lidl_list(session, ctx)
        # Fetch detail in parallel with bounded concurrency
        sem = asyncio.Semaphore(4)

        async def detail(it: dict) -> dict:
            async with sem:
                try:
                    d = await _fetch_lidl_detail(session, it["slug"], ctx)
                except Exception as e:
                    if ctx is not None:
                        await ctx.warning(f"Lidl detail fetch failed for {it['slug']}: {e}")
                    return {
                        "chain": "lidl",
                        "flyer_id": it["flyer_id"],
                        "title": it["name"],
                        "name": it["name"],
                        "valid_from": None,
                        "valid_to": None,
                        "valid_from_hint": None,
                        "thumbnail_url": None,
                        "detail_url": it["detail_url"],
                        "pdf_kind": "native",
                        "_slug": it["slug"],
                    }
                return {
                    "chain": "lidl",
                    "flyer_id": d.get("flyer_id") or it["flyer_id"],
                    "title": d.get("title") or it["name"],
                    "name": d.get("name") or it["name"],
                    "valid_from": d.get("valid_from"),
                    "valid_to": d.get("valid_to"),
                    "valid_from_hint": None,
                    "thumbnail_url": d.get("thumbnail_url"),
                    "detail_url": it["detail_url"],
                    "pdf_kind": "native",
                    "_slug": it["slug"],
                    "_pdf_url": d.get("pdf_url"),
                    "_file_size": d.get("file_size"),
                }

        out = await asyncio.gather(*(detail(it) for it in items))
    return list(out)


async def _list_biedronka(ctx: Optional[Context] = None) -> list[dict]:
    async with _new_session() as session:
        items = await _fetch_biedronka_list(session, ctx)
    return [
        {
            "chain": "biedronka",
            "flyer_id": it["flyer_id"],
            "title": it["title"],
            "name": it["name"],
            "valid_from": None,
            "valid_to": None,
            "valid_from_hint": it["valid_from_hint"],
            "thumbnail_url": it["thumbnail_url"],
            "detail_url": it["detail_url"],
            "pdf_kind": "assembled",
            "_slug": it["slug"],
            "_is_adult": it["is_adult"],
        }
        for it in items
    ]


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

def _public_flyer_view(flyer: dict) -> dict:
    """Drop private fields starting with `_`."""
    return {k: v for k, v in flyer.items() if not k.startswith("_")}


@mcp.tool(
    name="list_flyers",
    description=(
        "List currently-available promotional flyers (gazetki) from Polish supermarket "
        "chains. Supports Lidl and Biedronka. Returns each flyer's id, title, validity "
        "dates (when available), thumbnail and detail URLs. Use the `flyer_id` from this "
        "response with `download_flyer` to fetch the PDF."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def list_flyers(
    chain: Literal["lidl", "biedronka", "all"] = Field(
        default="all",
        description="Which chain to query: 'lidl', 'biedronka', or 'all'.",
    ),
    region: Optional[str] = Field(
        default=None,
        description="Reserved for future use. Ignored in v1 — both chains expose national "
        "flyers and the underlying APIs accept region_id=0 only.",
    ),
    ctx: Context = None,
) -> ToolResult:
    flyers: list[dict] = []
    errors: dict[str, str] = {}

    async def collect(name: str, coro):
        try:
            flyers.extend(await coro)
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
            if ctx is not None:
                await ctx.error(f"{name} listing failed: {e}")

    tasks = []
    if chain in ("lidl", "all"):
        tasks.append(collect("lidl", _list_lidl(ctx)))
    if chain in ("biedronka", "all"):
        tasks.append(collect("biedronka", _list_biedronka(ctx)))
    if tasks:
        await asyncio.gather(*tasks)

    public = [_public_flyer_view(f) for f in flyers]

    if not public and errors:
        return ToolResult(
            content=[{"type": "text", "text": f"Failed to list flyers: {errors}"}],
            structured_content={
                "error": "listing_failed",
                "stage": "list",
                "details": errors,
            },
        )

    by_chain: dict[str, int] = {}
    for f in public:
        by_chain[f["chain"]] = by_chain.get(f["chain"], 0) + 1
    summary_lines = [f"Found {len(public)} flyer(s):"]
    for c, count in sorted(by_chain.items()):
        summary_lines.append(f"  • {c}: {count}")
    for f in public[:5]:
        valid = (
            f"{f.get('valid_from')} → {f.get('valid_to')}"
            if f.get("valid_from") or f.get("valid_to")
            else (f.get("valid_from_hint") or "—")
        )
        summary_lines.append(f"  - [{f['chain']}/{f['flyer_id']}] {f['title']} ({valid})")

    return ToolResult(
        content=[{"type": "text", "text": "\n".join(summary_lines)}],
        structured_content={
            "chain": chain,
            "count": len(public),
            "flyers": public,
            "errors": errors or None,
        },
    )


async def _download_flyer_lidl(
    flyer_id: str, save_to: Optional[str], force_refresh: bool, ctx: Optional[Context]
) -> dict:
    pdf_path, meta_path = _flyer_cache_paths("lidl", flyer_id)
    meta = _read_meta(meta_path)
    if (
        not force_refresh
        and pdf_path.exists()
        and meta is not None
        and _is_cache_fresh(meta)
    ):
        if ctx is not None:
            await ctx.info(f"Using cached Lidl flyer {flyer_id}")
        page_count = meta.get("page_count") if meta.get("page_count") is not None else _pdf_page_count(pdf_path)
        return _finalize_download(
            chain="lidl",
            pdf_path=pdf_path,
            meta=meta,
            page_count=page_count,
            save_to=save_to,
            cached=True,
        )

    async with _new_session() as session:
        # We need slug + pdf_url. Look it up via the listing.
        listing = await _fetch_lidl_list(session, ctx)
        match = next((it for it in listing if it["flyer_id"] == flyer_id), None)
        if match is None:
            raise LookupError(f"Lidl flyer_id={flyer_id} not present in current listing")
        detail = await _fetch_lidl_detail(session, match["slug"], ctx)
        pdf_url = detail.get("pdf_url")
        if not pdf_url:
            raise RuntimeError(f"Lidl detail missing pdfUrl for {flyer_id}")
        if ctx is not None:
            await ctx.info(f"Downloading Lidl PDF: {pdf_url}")
        pdf_bytes = await _download_lidl_pdf(session, pdf_url, ctx)

    pdf_path.write_bytes(pdf_bytes)
    page_count = _pdf_page_count(pdf_path)
    new_meta = {
        "chain": "lidl",
        "flyer_id": flyer_id,
        "slug": match["slug"],
        "title": detail.get("title"),
        "name": detail.get("name"),
        "valid_from": detail.get("valid_from"),
        "valid_to": detail.get("valid_to"),
        "pdf_url": pdf_url,
        "thumbnail_url": detail.get("thumbnail_url"),
        "byte_count": len(pdf_bytes),
        "sha256": _sha256_bytes(pdf_bytes),
        "page_count": page_count,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "pdf_kind": "native",
        "source_urls": [pdf_url],
    }
    _write_meta(meta_path, new_meta)
    return _finalize_download(
        chain="lidl",
        pdf_path=pdf_path,
        meta=new_meta,
        page_count=page_count,
        save_to=save_to,
        cached=False,
    )


async def _download_flyer_biedronka(
    flyer_id: str, save_to: Optional[str], force_refresh: bool, ctx: Optional[Context]
) -> dict:
    pdf_path, meta_path = _flyer_cache_paths("biedronka", flyer_id)
    meta = _read_meta(meta_path)
    if (
        not force_refresh
        and pdf_path.exists()
        and meta is not None
        and _is_cache_fresh(meta)
    ):
        if ctx is not None:
            await ctx.info(f"Using cached Biedronka flyer {flyer_id}")
        page_count = meta.get("page_count") if meta.get("page_count") is not None else _pdf_page_count(pdf_path)
        return _finalize_download(
            chain="biedronka",
            pdf_path=pdf_path,
            meta=meta,
            page_count=page_count,
            save_to=save_to,
            cached=True,
        )

    async with _new_session() as session:
        listing = await _fetch_biedronka_list(session, ctx)
        match = next((it for it in listing if it["flyer_id"] == flyer_id), None)
        if match is None:
            raise LookupError(
                f"Biedronka flyer_id={flyer_id} not present in current listing"
            )
        if ctx is not None:
            await ctx.info(f"Resolving Biedronka leaflet UUID via {match['detail_url']}")
        leaflet_uuid = await _fetch_biedronka_uuid_from_press(
            session, match["detail_url"], ctx
        )
        if not leaflet_uuid:
            raise RuntimeError(
                f"Biedronka press page missing galleryLeaflet UUID: {match['detail_url']}"
            )
        if ctx is not None:
            await ctx.info(f"Fetching Biedronka leaflet API for {leaflet_uuid}")
        image_urls = await _fetch_biedronka_leaflet_api(session, leaflet_uuid, ctx)
        if not image_urls:
            raise RuntimeError(
                f"Biedronka leaflet API returned zero pages for {leaflet_uuid}"
            )
        pages = await _download_biedronka_pages(session, image_urls, ctx)

    pdf_bytes = _assemble_biedronka_pdf(pages)
    pdf_path.write_bytes(pdf_bytes)
    page_count = _pdf_page_count(pdf_path) or len(image_urls)
    new_meta = {
        "chain": "biedronka",
        "flyer_id": flyer_id,
        "slug": match["slug"],
        "title": match["title"],
        "name": match["name"],
        "valid_from": None,
        "valid_to": None,
        "valid_from_hint": match.get("valid_from_hint"),
        "leaflet_uuid": leaflet_uuid,
        "thumbnail_url": match.get("thumbnail_url"),
        "byte_count": len(pdf_bytes),
        "sha256": _sha256_bytes(pdf_bytes),
        "page_count": page_count,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "pdf_kind": "assembled",
        "source_urls": list(image_urls),
    }
    _write_meta(meta_path, new_meta)
    return _finalize_download(
        chain="biedronka",
        pdf_path=pdf_path,
        meta=new_meta,
        page_count=page_count,
        save_to=save_to,
        cached=False,
    )


def _finalize_download(
    *,
    chain: str,
    pdf_path: Path,
    meta: dict,
    page_count: Optional[int],
    save_to: Optional[str],
    cached: bool,
) -> dict:
    final_path = pdf_path
    if save_to:
        dest = Path(save_to).expanduser()
        if dest.is_dir():
            dest = dest / pdf_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(pdf_path.read_bytes())
        final_path = dest
    return {
        "chain": chain,
        "flyer_id": meta.get("flyer_id"),
        "file_path": str(final_path),
        "sha256": meta.get("sha256"),
        "byte_count": meta.get("byte_count"),
        "page_count": page_count,
        "pdf_kind": meta.get("pdf_kind"),
        "source_urls": meta.get("source_urls", []),
        "title": meta.get("title"),
        "valid_from": meta.get("valid_from"),
        "valid_to": meta.get("valid_to"),
        "cached": cached,
    }


@mcp.tool(
    name="download_flyer",
    description=(
        "Download a specific promotional flyer as a PDF, identified by chain + flyer_id "
        "(both fields appear in `list_flyers`). Lidl flyers are downloaded directly; "
        "Biedronka flyers are assembled from per-page PNGs. The PDF is cached on disk "
        "and copied to `save_to` when supplied."
    ),
    annotations={"readOnlyHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def download_flyer(
    chain: Literal["lidl", "biedronka"] = Field(
        description="Chain that owns the flyer."
    ),
    flyer_id: str = Field(
        min_length=1,
        description="The flyer_id returned by list_flyers (UUID for Lidl, short id for Biedronka).",
    ),
    save_to: Optional[str] = Field(
        default=None,
        description="Optional output path. May be a directory or full file path. "
        "If unset, only the cache copy is produced.",
    ),
    region: Optional[str] = Field(
        default=None,
        description="Reserved for future use. Ignored in v1.",
    ),
    force_refresh: bool = Field(
        default=False,
        description="Bypass the disk cache and re-download.",
    ),
    ctx: Context = None,
) -> ToolResult:
    try:
        if chain == "lidl":
            result = await _download_flyer_lidl(flyer_id, save_to, force_refresh, ctx)
        else:
            result = await _download_flyer_biedronka(
                flyer_id, save_to, force_refresh, ctx
            )
    except LookupError as e:
        return ToolResult(
            content=[{"type": "text", "text": str(e)}],
            structured_content={
                "error": "not_found",
                "stage": "list",
                "chain": chain,
                "flyer_id": flyer_id,
                "details": str(e),
            },
        )
    except aiohttp.ClientError as e:
        return ToolResult(
            content=[{"type": "text", "text": f"HTTP error: {e}"}],
            structured_content={
                "error": "http_error",
                "stage": "download",
                "chain": chain,
                "flyer_id": flyer_id,
                "details": str(e),
            },
        )
    except Exception as e:
        return ToolResult(
            content=[{"type": "text", "text": f"Download failed: {e}"}],
            structured_content={
                "error": "download_failed",
                "stage": "download",
                "chain": chain,
                "flyer_id": flyer_id,
                "details": f"{type(e).__name__}: {e}",
            },
        )

    size_kb = (result.get("byte_count") or 0) / 1024
    pages = result.get("page_count")
    text = (
        f"{'(cached) ' if result['cached'] else ''}"
        f"Downloaded {chain} flyer {flyer_id} → {result['file_path']} "
        f"({size_kb:.1f} KB"
        + (f", {pages} pages" if pages else "")
        + ")"
    )
    return ToolResult(
        content=[{"type": "text", "text": text}],
        structured_content=result,
    )


@mcp.tool(
    name="get_current_flyers",
    description=(
        "Convenience tool: return the headline current flyer for each chain (the first "
        "entry in the listing — typically the most recently-published gazetka)."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def get_current_flyers(
    chain: Optional[Literal["lidl", "biedronka"]] = Field(
        default=None,
        description="Restrict to a single chain. When omitted, returns one entry per chain.",
    ),
    ctx: Context = None,
) -> ToolResult:
    chains_to_query: list[str] = (
        [chain] if chain in ("lidl", "biedronka") else ["lidl", "biedronka"]
    )
    out: dict[str, Optional[dict]] = {}
    errors: dict[str, str] = {}
    for c in chains_to_query:
        try:
            items = await (_list_lidl(ctx) if c == "lidl" else _list_biedronka(ctx))
            out[c] = _public_flyer_view(items[0]) if items else None
        except Exception as e:
            errors[c] = f"{type(e).__name__}: {e}"
            out[c] = None
            if ctx is not None:
                await ctx.error(f"{c} current-flyer fetch failed: {e}")
    lines = ["Current flyers:"]
    for c, f in out.items():
        if f is None:
            lines.append(f"  • {c}: <none>" + (f" ({errors[c]})" if c in errors else ""))
        else:
            lines.append(f"  • {c}: {f['title']} [{f['flyer_id']}]")
    return ToolResult(
        content=[{"type": "text", "text": "\n".join(lines)}],
        structured_content={
            "current": out,
            "errors": errors or None,
        },
    )


if __name__ == "__main__":
    mcp.run()
