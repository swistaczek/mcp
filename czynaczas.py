"""FastMCP Server for czynaczas.pl

Realtime Polish public transport (tram + bus) — vehicle positions, delays,
arrivals at stops. Wraps both the REST endpoints and the Socket.IO live feed.
"""

import asyncio
import json
import time
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

import aiohttp
from fastmcp import Context, FastMCP
from fastmcp.tools.tool import ToolResult
from pydantic import Field

mcp = FastMCP("Czy Na Czas")

BASE_URL = "https://czynaczas.pl"
API_BASE = f"{BASE_URL}/api"
WS_URL = "wss://czynaczas.pl/socket.io/?EIO=4&transport=websocket"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)
SUPPORTED_CITIES = ["poznan", "warsaw", "krakow", "wroclaw", "lodz"]
WARSAW_TZ = ZoneInfo("Europe/Warsaw")

DEFAULT_LISTEN_SECONDS = 6
MAX_LISTEN_SECONDS = 20


def _http_headers(city: str) -> dict[str, str]:
    return {
        "accept": "application/json, text/plain, */*",
        "accept-language": "pl-PL,pl;q=0.9,en;q=0.8",
        "referer": f"{BASE_URL}/{city}",
        "user-agent": USER_AGENT,
    }


def _ws_headers(city: str) -> dict[str, str]:
    return {
        "Origin": BASE_URL,
        "User-Agent": USER_AGENT,
        "Referer": f"{BASE_URL}/{city}",
    }


def _fold(s: str) -> str:
    """Lowercase + strip Polish diacritics for fuzzy matching."""
    if not s:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s.casefold())
        if not unicodedata.combining(c)
    )


def _validate_city(city: str) -> str:
    if city not in SUPPORTED_CITIES:
        raise ValueError(
            f"Unsupported city {city!r}. Supported: {SUPPORTED_CITIES}"
        )
    return city


async def _fetch_json(session: aiohttp.ClientSession, city: str, path: str,
                     params: Optional[dict] = None) -> Any:
    url = f"{API_BASE}/{city}/{path}"
    async with session.get(url, headers=_http_headers(city), params=params,
                           timeout=aiohttp.ClientTimeout(total=15)) as r:
        r.raise_for_status()
        return await r.json()


async def _fetch_transport(session: aiohttp.ClientSession, city: str) -> dict:
    """Bulk dataset: stops, routes, alerts. Used as the authoritative stop list."""
    return await _fetch_json(session, city, "transport")


async def _collect_live_vehicles(city: str, seconds: int,
                                 ctx: Optional[Context] = None) -> dict[str, dict]:
    """Snapshot live vehicles for a city by subscribing to its Socket.IO namespace."""
    vehicles: dict[str, dict] = {}
    async with aiohttp.ClientSession(headers=_ws_headers(city)) as sess:
        async with sess.ws_connect(WS_URL, heartbeat=None,
                                   timeout=aiohttp.ClientWSTimeout(ws_close=10.0)) as ws:
            await ws.send_str("40")
            await ws.send_str(f"40/{city},")
            ns_prefix = f"42/{city},"
            t0 = time.monotonic()
            while time.monotonic() - t0 < seconds:
                try:
                    msg = await asyncio.wait_for(
                        ws.receive(), timeout=max(0.5, seconds - (time.monotonic() - t0))
                    )
                except asyncio.TimeoutError:
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                raw = msg.data
                if raw == "2":
                    await ws.send_str("3")
                    continue
                if raw.startswith(ns_prefix):
                    try:
                        payload = json.loads(raw[len(ns_prefix):])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], dict):
                        for vid, v in payload[1].get("data", {}).items():
                            existing = vehicles.get(vid, {})
                            existing.update(v)
                            vehicles[vid] = existing
                if ctx and (time.monotonic() - t0) % 1 < 0.1:
                    await ctx.report_progress(
                        time.monotonic() - t0, seconds,
                        f"Collected {len(vehicles)} vehicles"
                    )
            await ws.close()
    return vehicles


def _seconds_since_midnight_warsaw(now: Optional[datetime] = None) -> tuple[int, datetime]:
    now = now or datetime.now(WARSAW_TZ)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((now - midnight).total_seconds()), midnight


@mcp.tool(
    name="list_supported_cities",
    description="List the Polish cities for which czynaczas.pl exposes realtime data.",
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
)
async def list_supported_cities() -> ToolResult:
    return ToolResult(
        content=[{"type": "text", "text": "Supported cities: " + ", ".join(SUPPORTED_CITIES)}],
        structured_content={"cities": SUPPORTED_CITIES, "source": "static"},
    )


@mcp.tool(
    name="find_stops",
    description=(
        "Find public-transport stops in a Polish city. Matches the query against the full "
        "stop list (3000+ stops per city) using a diacritic-insensitive substring match — "
        "so 'Dabrowskiego' matches 'Dąbrowskiego', and multi-token queries like "
        "'Polna Szpital' match stops where every token appears in the name. "
        "Use this BEFORE get_departures to resolve a user-supplied stop name to a stop_id. "
        "Returns at most `limit` stops sorted by name length (shorter = more central match)."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def find_stops(
    city: str = Field(description="City slug. Use list_supported_cities to discover."),
    query: str = Field(min_length=1, max_length=100,
                       description="Stop name fragment, e.g. 'Polna', 'Rondo Kaponiera'."),
    limit: int = Field(default=20, ge=1, le=100),
    ctx: Context = None,
) -> ToolResult:
    _validate_city(city)
    folded_q = _fold(query)
    tokens = [t for t in folded_q.replace("/", " ").split() if t]
    async with aiohttp.ClientSession() as session:
        transport = await _fetch_transport(session, city)
    stops = transport.get("stops", [])
    matches = []
    for s in stops:
        # tuple shape: [id, name, lat, lon, [zone], [line_ids]]
        name = s[1]
        if all(t in _fold(name) for t in tokens):
            matches.append({
                "id": str(s[0]),
                "name": name,
                "lat": s[2],
                "lon": s[3],
                "lines": s[5] if len(s) > 5 else [],
            })
    matches.sort(key=lambda m: (len(m["name"]), m["name"]))
    matches = matches[:limit]
    if not matches:
        text = f"No stops found in {city} matching {query!r}."
    else:
        lines = [f"{len(matches)} stops in {city} matching {query!r}:"]
        for m in matches[:10]:
            lines.append(f"  • [{m['id']}] {m['name']}  (lat={m['lat']:.4f}, lon={m['lon']:.4f})")
        if len(matches) > 10:
            lines.append(f"  … and {len(matches) - 10} more")
        text = "\n".join(lines)
    return ToolResult(
        content=[{"type": "text", "text": text}],
        structured_content={"city": city, "query": query, "result_count": len(matches),
                            "stops": matches},
    )


@mcp.tool(
    name="get_trip",
    description=(
        "Fetch a single trip's route shape (GeoJSON LineString) and ordered stop list "
        "with scheduled times. Each stop in the returned list has [trip_id, "
        "seconds_since_midnight, sequence, _, stop_id, stop_name, lat, lon]. "
        "Use trip_id values returned by get_departures to inspect a specific trip."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def get_trip(
    city: str = Field(description="City slug."),
    trip_id: str = Field(description="Trip ID, e.g. '1_1858988^N+'. Will be URL-encoded."),
    ctx: Context = None,
) -> ToolResult:
    _validate_city(city)
    async with aiohttp.ClientSession() as session:
        data = await _fetch_json(session, city, f"trip?trip_id={quote(trip_id)}")
    stops = data.get("stops", [])
    text = (f"Trip {trip_id} in {city}: {len(stops)} stops, route shape with "
            f"{len(data.get('shape', {}).get('coordinates', []))} polyline points.")
    return ToolResult(
        content=[{"type": "text", "text": text}],
        structured_content={"city": city, "trip_id": trip_id, "trip": data},
    )


@mcp.tool(
    name="get_vehicle",
    description=(
        "Fetch metadata about a specific vehicle (tram or bus): model, year, depot, "
        "low-floor, A/C, etc. Use vehicle IDs returned by get_departures."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def get_vehicle(
    city: str = Field(description="City slug."),
    vehicle_id: str = Field(description="Vehicle ID, e.g. '0/272' or '3/4202'."),
    ctx: Context = None,
) -> ToolResult:
    _validate_city(city)
    async with aiohttp.ClientSession() as session:
        data = await _fetch_json(session, city, f"vehicle?id={quote(vehicle_id)}")
    return ToolResult(
        content=[{"type": "text", "text": json.dumps(data, ensure_ascii=False)[:500]}],
        structured_content={"city": city, "vehicle_id": vehicle_id, "vehicle": data},
    )


@mcp.tool(
    name="get_departures",
    description=(
        "Get the next departures from a stop, with realtime delays and destination headsigns. "
        "This is the headline tool for questions like "
        "'kiedy najbliższy tramwaj 16?', 'kiedy coś na Ogrody z Polna?', "
        "'is my bus delayed?'. "
        "How it works: snapshots all live vehicles in the city via Socket.IO, finds those "
        "whose current trip will visit this stop in the future, and computes ETA from the "
        "scheduled arrival time at that stop plus the vehicle's reported delay. "
        "Use destination_filter to narrow to vehicles whose headsign contains a substring "
        "(diacritic- and case-insensitive) — e.g. destination_filter='Ogrody' for "
        "'na Ogrody'. Use line_filter to narrow to a single route_id like '16'. "
        "Both filters are combined when given."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": False, "openWorldHint": True},
)
async def get_departures(
    city: str = Field(description="City slug."),
    stop_id: str = Field(description="Stop ID returned by find_stops."),
    destination_filter: Optional[str] = Field(
        default=None,
        description="Optional headsign substring, e.g. 'Ogrody', 'Rondo Kaponiera'. "
                    "Diacritic- and case-insensitive.",
    ),
    line_filter: Optional[str] = Field(
        default=None,
        description="Optional line/route number as string, e.g. '16', 'N12', 'T1'.",
    ),
    limit: int = Field(default=10, ge=1, le=50),
    listen_seconds: int = Field(
        default=DEFAULT_LISTEN_SECONDS, ge=2, le=MAX_LISTEN_SECONDS,
        description=f"How long to listen for live updates. Default {DEFAULT_LISTEN_SECONDS}s.",
    ),
    ctx: Context = None,
) -> ToolResult:
    _validate_city(city)
    folded_dest = _fold(destination_filter) if destination_filter else None
    folded_line = _fold(line_filter) if line_filter else None

    # 1. Snapshot live vehicles
    vehicles = await _collect_live_vehicles(city, listen_seconds, ctx)

    # 2. Filter by destination/line and presence of trip_id
    candidates = []
    for v in vehicles.values():
        if not v.get("trip_id"):
            continue
        if folded_dest and folded_dest not in _fold(v.get("trip_headsign", "")):
            continue
        if folded_line and folded_line != _fold(str(v.get("route_id", ""))):
            continue
        candidates.append(v)

    # 3. For each candidate, fetch trip and locate this stop_id; compute ETA
    sec_now, midnight = _seconds_since_midnight_warsaw()
    departures = []
    async with aiohttp.ClientSession() as session:
        # cap concurrent trip fetches
        sem = asyncio.Semaphore(8)

        async def resolve(v):
            async with sem:
                try:
                    trip = await _fetch_json(
                        session, city, f"trip?trip_id={quote(v['trip_id'])}"
                    )
                except Exception:
                    return None
            for tup in trip.get("stops", []):
                if str(tup[4]) == str(stop_id):
                    sched_s = int(tup[1])
                    delay_s = int(v.get("delay") or 0)
                    eta_s = sched_s + delay_s
                    # only future arrivals (with a 90s grace window for "just departing")
                    if eta_s < sec_now - 90:
                        return None
                    # require the vehicle to not have already passed this stop on this trip
                    cur_seq = v.get("stop_sequence")
                    target_seq = int(tup[2])
                    if isinstance(cur_seq, int) and cur_seq > target_seq:
                        return None
                    return {
                        "line": str(v.get("route_id")),
                        "vehicle_id": v.get("id"),
                        "vehicle_no": v.get("vehicleNo"),
                        "headsign": v.get("trip_headsign"),
                        "scheduled_at": (midnight + timedelta(seconds=sched_s)).strftime("%H:%M"),
                        "eta_at": (midnight + timedelta(seconds=eta_s)).strftime("%H:%M:%S"),
                        "delay_seconds": delay_s,
                        "minutes_until": round((eta_s - sec_now) / 60, 1),
                        "current_status": v.get("current_status"),
                        "current_stop_name": v.get("stop_name"),
                        "current_stop_sequence": cur_seq,
                        "trip_id": v["trip_id"],
                    }
            return None

        results = await asyncio.gather(*(resolve(v) for v in candidates))
        departures = [r for r in results if r]

    departures.sort(key=lambda d: d["minutes_until"])
    departures = departures[:limit]

    # 4. Format text summary
    if not departures:
        filt_desc = []
        if destination_filter:
            filt_desc.append(f"na '{destination_filter}'")
        if line_filter:
            filt_desc.append(f"linia {line_filter}")
        suffix = " " + " ".join(filt_desc) if filt_desc else ""
        text = (f"Brak nadchodzących odjazdów ze stop_id={stop_id} ({city}){suffix} "
                f"w bieżącym snapshocie ({len(vehicles)} pojazdów na żywo, "
                f"{len(candidates)} po filtrze).")
    else:
        lines = [f"Najbliższe odjazdy ze stop_id={stop_id} ({city}):"]
        for d in departures:
            mu = d["minutes_until"]
            when = "TERAZ" if -1 <= mu <= 1 else f"za {int(mu)} min" if mu > 0 else f"{int(-mu)} min temu"
            ds = d["delay_seconds"]
            if ds > 30:
                delay = f" (opóźniony {ds // 60}:{ds % 60:02d})"
            elif ds < -30:
                delay = f" (przed czasem {-ds // 60}:{-ds % 60:02d})"
            else:
                delay = " (planowo)"
            lines.append(f"  • {d['line']} → {d['headsign']} — {when}{delay} "
                         f"(plan {d['scheduled_at']}, ETA {d['eta_at']})")
            lines.append(f"      teraz: {d['current_stop_name']!r} "
                         f"[{d['current_status']}]")
        text = "\n".join(lines)

    return ToolResult(
        content=[{"type": "text", "text": text}],
        structured_content={
            "city": city, "stop_id": str(stop_id),
            "filters": {"destination_filter": destination_filter,
                        "line_filter": line_filter, "limit": limit},
            "snapshot_seconds": listen_seconds,
            "vehicles_seen": len(vehicles),
            "candidates_after_filter": len(candidates),
            "departures": departures,
            "fetched_at": datetime.now(WARSAW_TZ).isoformat(),
        },
    )


if __name__ == "__main__":
    mcp.run()
