"""FastMCP Server for Poznań city events (Co? Gdzie? Kiedy?).

Scrapes https://www.poznan.pl/mim/events/ — the official municipal events
calendar maintained by the City of Poznań — and exposes it as MCP tools.

The page renders 20 events per "page" via an XHR endpoint:
    /mim/events/events.html?co=list&lang=pl&category={ID}&p={PAGE}

with `category` empty for "all" and a numeric category id otherwise (e.g.
214 = Sport, 217 = Muzyka, 218 = Sztuka). `p` is zero-indexed.
Detail pages live at:
    /mim/events/{slug},{numeric_id}.html
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from fastmcp import Context, FastMCP
from fastmcp.tools.tool import ToolResult
from pydantic import Field

mcp = FastMCP("Poznan Events")

BASE_URL = "https://www.poznan.pl"
EVENTS_HOME = f"{BASE_URL}/mim/events/"
LIST_ENDPOINT = f"{BASE_URL}/mim/events/events.html"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)
MAX_PAGES_PER_CALL = 5

# Detail URLs look like /mim/events/some-slug,179433.html — capture the trailing id.
_DETAIL_ID_RE = re.compile(r",(?P<id>\d+)\.html$")
_POLISH_DATE_RE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$")
_POLISH_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _default_headers() -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
    }


async def _fetch_html(
    session: aiohttp.ClientSession, url: str, params: Optional[dict] = None
) -> str:
    async with session.get(
        url, headers=_default_headers(), params=params,
        timeout=REQUEST_TIMEOUT, allow_redirects=True,
    ) as r:
        r.raise_for_status()
        return await r.text()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _absolute(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    return urljoin(BASE_URL, url)


def _polish_date_to_iso(value: str) -> Optional[str]:
    m = _POLISH_DATE_RE.match(value.strip())
    if not m:
        return None
    day, month, year = m.groups()
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def _normalize_time(value: str) -> Optional[str]:
    m = _POLISH_TIME_RE.match(value.strip())
    if not m:
        return None
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def _extract_event_id(href: str) -> Optional[str]:
    m = _DETAIL_ID_RE.search(href)
    return m.group("id") if m else None


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_event_box(box) -> Optional[dict]:
    """Pull a single event card out of the listing HTML. Returns None if the
    card is missing the title link (defensive — the source occasionally drops
    cards mid-render)."""
    title_link = box.select_one(".description-event-title-link")
    if not title_link or not title_link.get("href"):
        return None
    href = title_link["href"]
    detail_url = _absolute(href)
    event_id = _extract_event_id(href)

    title_el = box.select_one(".description-event-title")
    title = _clean_text(title_el.get_text(" ", strip=True)) if title_el else ""

    # Two <time> elements: first = date (DD.MM.YYYY), second = time (HH:MM).
    times = [t.get_text(strip=True) for t in box.select("time")]
    raw_date = times[0] if len(times) >= 1 else None
    raw_time = times[1] if len(times) >= 2 else None

    place_el = box.select_one(".description-event-place")
    place = _clean_text(place_el.get_text(" ", strip=True)) if place_el else None

    categories: list[dict] = []
    seen_cat_ids: set[str] = set()
    for link in box.select(".description-event-category-link"):
        cat_id = link.get("data-id")
        name = _clean_text(link.get_text(" ", strip=True))
        if cat_id and cat_id not in seen_cat_ids:
            seen_cat_ids.add(cat_id)
            categories.append({"id": cat_id, "name": name})

    img = box.select_one(".image-events img")
    thumbnail = _absolute(img.get("data-src") or img.get("src")) if img else None

    return {
        "event_id": event_id,
        "title": title,
        "date": _polish_date_to_iso(raw_date) if raw_date else None,
        "date_raw": raw_date,
        "time": _normalize_time(raw_time) if raw_time else None,
        "place": place,
        "categories": categories,
        "thumbnail_url": thumbnail,
        "detail_url": detail_url,
    }


def _parse_event_list(html_text: str) -> list[dict]:
    soup = BeautifulSoup(html_text, "lxml")
    out: list[dict] = []
    for box in soup.select(".event-box"):
        item = _parse_event_box(box)
        if item is not None:
            out.append(item)
    return out


def _parse_categories(html_text: str) -> list[dict]:
    """Pull the category catalog off the homepage. Categories appear as
    `<a href="/mim/events/{slug},c,{id}/">{Name}</a>` in the navigation."""
    soup = BeautifulSoup(html_text, "lxml")
    cats: dict[str, dict] = {}
    cat_re = re.compile(r"/mim/events/([^/,]+),c,(\d+)/?$")
    for a in soup.select("a[href]"):
        m = cat_re.search(a["href"])
        if not m:
            continue
        slug, cat_id = m.groups()
        if cat_id in cats:
            continue
        name = _clean_text(a.get_text(" ", strip=True))
        if not name:
            continue
        cats[cat_id] = {"id": cat_id, "slug": slug, "name": name}
    return sorted(cats.values(), key=lambda c: c["name"].lower())


def _parse_event_detail(html_text: str, source_url: Optional[str] = None) -> dict:
    soup = BeautifulSoup(html_text, "lxml")

    # Title — prefer og:title, fall back to .event-title h-tag, then <title>.
    og_title = soup.select_one('meta[property="og:title"]')
    title = og_title.get("content").strip() if og_title and og_title.get("content") else None
    if not title:
        h = soup.select_one(".event-title h1, .event-title h2")
        title = _clean_text(h.get_text()) if h else None

    # Date / time / place — the print block has clean single-purpose elements.
    raw_date = _clean_text(d.get_text()) if (d := soup.select_one(".event-print-date")) else None
    place = _clean_text(p.get_text()) if (p := soup.select_one(".event-print-place")) else None

    # Time = the .event_details_data_box that is *not* the date/place box.
    raw_time = None
    for box in soup.select(".event_details_data_box"):
        classes = box.get("class") or []
        if "event-print-date" in classes or "event-print-place" in classes:
            continue
        text = _clean_text(box.get_text())
        if _POLISH_TIME_RE.match(text):
            raw_time = text
            break

    # Categories — every <a> inside .event-print-categories is a category link.
    categories: list[dict] = []
    cat_link_re = re.compile(r"/mim/events/([^/,]+),c,(\d+)/?")
    seen_cat_ids: set[str] = set()
    for a in soup.select(".event-print-categories a[href]"):
        m = cat_link_re.search(a["href"])
        if not m:
            continue
        slug, cat_id = m.groups()
        if cat_id in seen_cat_ids:
            continue
        seen_cat_ids.add(cat_id)
        categories.append(
            {"id": cat_id, "slug": slug, "name": _clean_text(a.get_text())}
        )

    # Description: the .event-print-describe block contains the prose plus a
    # tail of share/print/banner UI. Take the og:description as the canonical
    # short form, and grab the longer prose by stripping known tail markers.
    og_desc = soup.select_one('meta[property="og:description"]')
    short_description = (
        og_desc.get("content").strip() if og_desc and og_desc.get("content") else None
    )

    long_description = None
    describe_block = soup.select_one(".event-print-describe")
    if describe_block:
        # Drop banner/share controls before extracting text.
        for noise in describe_block.select(
            ".event-print-baner, .event-print-social, .event-print-tags, "
            ".social-mim, .qr-code, button, script, style"
        ):
            noise.decompose()
        long_description = _clean_text(describe_block.get_text(" ", strip=True))

    # Hero image
    img_el = soup.select_one(".event-print-photo img, .event-photo img")
    image_url = _absolute(img_el.get("src") or img_el.get("data-src")) if img_el else None

    event_id = None
    if source_url:
        event_id = _extract_event_id(urlparse(source_url).path)

    return {
        "event_id": event_id,
        "title": title,
        "date": _polish_date_to_iso(raw_date) if raw_date else None,
        "date_raw": raw_date,
        "time": _normalize_time(raw_time) if raw_time else None,
        "place": place,
        "categories": categories,
        "short_description": short_description,
        "description": long_description,
        "image_url": image_url,
        "detail_url": source_url,
    }


# ---------------------------------------------------------------------------
# High-level fetchers
# ---------------------------------------------------------------------------

def _normalize_category(category: Optional[str | int]) -> str:
    """Accept None / "" / int / digit-string. Returns the empty string for
    "no filter" and a digit-string id otherwise."""
    if category is None or category == "":
        return ""
    if isinstance(category, int):
        return str(category)
    text = str(category).strip()
    if not text:
        return ""
    if not text.isdigit():
        raise ValueError(
            f"category must be a numeric id (e.g. '214' for Sport); got {category!r}. "
            f"Use list_event_categories to discover ids."
        )
    return text


async def _fetch_events_page(
    session: aiohttp.ClientSession, category: str, page: int
) -> list[dict]:
    params = {"co": "list", "lang": "pl", "category": category, "p": str(page)}
    html_text = await _fetch_html(session, LIST_ENDPOINT, params=params)
    return _parse_event_list(html_text)


async def _fetch_event_detail(
    session: aiohttp.ClientSession, url: str
) -> dict:
    html_text = await _fetch_html(session, url)
    return _parse_event_detail(html_text, source_url=url)


def _resolve_event_url(event_ref: str) -> str:
    """Accept a numeric id, a slug+id (`slug,12345`), a relative path, or a
    full URL. Returns a fully-qualified detail URL."""
    ref = event_ref.strip()
    if not ref:
        raise ValueError("event_ref must be non-empty")
    if ref.startswith("http://") or ref.startswith("https://"):
        return ref
    if ref.startswith("/"):
        return urljoin(BASE_URL, ref)
    if ref.isdigit():
        # Numeric id only — the source rewrites to the canonical URL on its own,
        # so any slug works. Use a placeholder slug to keep the path well-formed.
        return f"{BASE_URL}/mim/events/event,{ref}.html"
    if "," in ref and ref.endswith(".html"):
        return f"{BASE_URL}/mim/events/{ref}"
    if "," in ref:
        return f"{BASE_URL}/mim/events/{ref}.html"
    raise ValueError(
        f"Could not parse event reference {event_ref!r}. Pass a numeric id, "
        "a slug+id (slug,12345), an /mim/events/... path, or a full URL."
    )


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="list_events",
    description=(
        "List upcoming events from the City of Poznań events calendar "
        "(Co? Gdzie? Kiedy? — https://www.poznan.pl/mim/events/). "
        "Each page returns up to 20 events. Use `category` to filter to a "
        "single category (numeric id from list_event_categories — e.g. '214' "
        "for Sport, '217' for Muzyka, '218' for Sztuka). Use `pages` to fetch "
        "multiple pages in one call (1–5). Events are returned in the source's "
        "chronological order, soonest first."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def list_events(
    category: Optional[str] = Field(
        default=None,
        description=(
            "Category filter — numeric id (e.g. '214' for Sport). Omit or pass "
            "empty string for all categories. Use list_event_categories to "
            "discover available ids."
        ),
    ),
    page: int = Field(
        default=0,
        ge=0,
        description="Zero-indexed page number. Page 0 is the first 20 events.",
    ),
    pages: int = Field(
        default=1,
        ge=1,
        le=MAX_PAGES_PER_CALL,
        description=(
            "How many consecutive pages to fetch starting at `page`. Capped at "
            f"{MAX_PAGES_PER_CALL} to keep responses bounded."
        ),
    ),
    ctx: Context = None,
) -> ToolResult:
    cat = _normalize_category(category)
    if ctx is not None:
        await ctx.info(
            f"Fetching Poznań events: category={cat or 'all'}, "
            f"pages={page}..{page + pages - 1}"
        )
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *(_fetch_events_page(session, cat, page + i) for i in range(pages)),
            return_exceptions=True,
        )

    events: list[dict] = []
    pages_meta: list[dict] = []
    errors: list[str] = []
    for offset, res in enumerate(results):
        page_no = page + offset
        if isinstance(res, Exception):
            errors.append(f"page {page_no}: {type(res).__name__}: {res}")
            pages_meta.append({"page": page_no, "count": 0, "error": str(res)})
            continue
        events.extend(res)
        pages_meta.append({"page": page_no, "count": len(res)})

    summary_lines = [
        f"Found {len(events)} event(s) "
        f"(category={cat or 'all'}, pages={page}..{page + pages - 1}):"
    ]
    for ev in events[:5]:
        when = ev["date"] or ev["date_raw"] or "?"
        if ev.get("time"):
            when = f"{when} {ev['time']}"
        summary_lines.append(f"  - [{ev['event_id'] or '?'}] {when} — {ev['title']}")
    if len(events) > 5:
        summary_lines.append(f"  ... and {len(events) - 5} more")
    if errors:
        summary_lines.append("Errors:")
        summary_lines.extend(f"  ! {e}" for e in errors)

    return ToolResult(
        content=[{"type": "text", "text": "\n".join(summary_lines)}],
        structured_content={
            "category": cat or None,
            "page": page,
            "pages_requested": pages,
            "pages": pages_meta,
            "count": len(events),
            "events": events,
            "errors": errors or None,
            "fetched_at": datetime.utcnow().isoformat() + "Z",
        },
    )


@mcp.tool(
    name="get_event",
    description=(
        "Fetch the full detail of a single Poznań event — title, date, time, "
        "place, categories, full description, and hero image. Accepts a "
        "numeric event id ('179433'), a slug+id ('parkrun-poznan,179433'), "
        "an /mim/events/... path, or a full URL — all of which appear in "
        "list_events output."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def get_event(
    event_ref: str = Field(
        min_length=1,
        description=(
            "Event reference — numeric id, slug+id, /mim/events/... path, or "
            "full https URL. The detail_url field from list_events works "
            "directly."
        ),
    ),
    ctx: Context = None,
) -> ToolResult:
    try:
        url = _resolve_event_url(event_ref)
    except ValueError as e:
        return ToolResult(
            content=[{"type": "text", "text": str(e)}],
            structured_content={"error": "invalid_reference", "details": str(e)},
        )

    if ctx is not None:
        await ctx.info(f"Fetching Poznań event detail: {url}")
    async with aiohttp.ClientSession() as session:
        try:
            event = await _fetch_event_detail(session, url)
        except aiohttp.ClientResponseError as e:
            return ToolResult(
                content=[{"type": "text", "text": f"HTTP {e.status} for {url}"}],
                structured_content={
                    "error": "http_error",
                    "status": e.status,
                    "url": url,
                },
            )
        except aiohttp.ClientError as e:
            return ToolResult(
                content=[{"type": "text", "text": f"Network error: {e}"}],
                structured_content={
                    "error": "network_error",
                    "url": url,
                    "details": str(e),
                },
            )

    when = event.get("date") or event.get("date_raw") or "?"
    if event.get("time"):
        when = f"{when} {event['time']}"
    line = f"{event.get('title') or '(untitled)'} — {when}"
    if event.get("place"):
        line += f" @ {event['place']}"
    return ToolResult(
        content=[{"type": "text", "text": line}],
        structured_content=event,
    )


@mcp.tool(
    name="list_event_categories",
    description=(
        "List the event categories used by the Poznań events calendar — name, "
        "slug, and numeric id. Use the id with list_events(category=...) to "
        "filter. Categories include Sport, Muzyka, Sztuka, Film, Teatr, "
        "Książki, Dziecko (children), Konferencje, spotkania i wykłady, etc."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def list_event_categories(ctx: Context = None) -> ToolResult:
    if ctx is not None:
        await ctx.info("Fetching Poznań event categories from homepage")
    async with aiohttp.ClientSession() as session:
        html_text = await _fetch_html(session, EVENTS_HOME)
    cats = _parse_categories(html_text)
    summary = "\n".join([f"Found {len(cats)} categories:"] + [
        f"  - {c['id']}: {c['name']} ({c['slug']})" for c in cats
    ])
    return ToolResult(
        content=[{"type": "text", "text": summary}],
        structured_content={"count": len(cats), "categories": cats},
    )


if __name__ == "__main__":
    mcp.run()
