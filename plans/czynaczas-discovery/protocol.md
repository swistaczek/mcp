# Czynaczas Socket.IO Protocol — Discovered 2026-04-28

## Endpoint
`wss://czynaczas.pl/socket.io/?EIO=4&transport=websocket`

## Required headers
- `Origin: https://czynaczas.pl`
- `Referer: https://czynaczas.pl/<city>` (any city slug works)

## Subscription protocol
City data is delivered via **Socket.IO namespaces**. To subscribe, send the Socket.IO connect-namespace frame:

```
40/<city_slug>,
```

Valid `<city_slug>` values (server confirms with `40/<city>,{"sid":"..."}`):
`poznan`, `warsaw`, `krakow`, `wroclaw`, `lodz`

Invalid slugs (from JS bundle but server rejects with `44/<slug>,{"message":"Invalid namespace"}`):
`poznanMaltanka`, `warsawFerry`, `warsawDudekFerry` — these are UI sub-modes only.

You can connect to multiple namespaces on a single Socket.IO connection.

## Inbound events
After subscribing, the server **immediately** sends a snapshot, then keeps pushing updates roughly every 30 seconds per city. Frame format:

```
42/<city>,["<city>", { "data": { "<vehicle_id>": <vehicle>, ... } }]
```

The event name equals the city slug. The payload's `data` is a dict keyed by vehicle id.

## Vehicle payload fields

| field | type | description |
|---|---|---|
| `id` | string | `"<type>/<vehicleNo>"` e.g. `"3/4202"` |
| `type` | string | vehicle type code (e.g. `"3"`) |
| `vehicleNo` | string | fleet number |
| `brigade` | string | shift/duty identifier |
| `trip_id` | string | scheduled trip id (matches `/api/<city>/trip?trip_id=...`) |
| `route_id` | string | line/route number (e.g. `"401"`, `"131"`, `"58B"`) |
| `lat`, `lon` | float | WGS84 |
| `timestamp` | int | epoch milliseconds of last GPS fix |
| `live` | bool | whether GPS is currently live |
| `trip_headsign` | string | destination, e.g. `"SWARZĘDZ OS. KOŚCIUSZKOWCÓW"`, `"Bronowice Małe"` |
| `stop_sequence` | int | 0-based index along the trip |
| `current_status` | string | `IN_TRANSIT_TO` \| `STOPPED_AT` \| `AWAITING_DEPARTURE` |
| `stop_id` | string | upcoming/current stop id (matches `/api/<city>/transport.stops[].0`) |
| `stop_name` | string | upcoming/current stop name |
| `between` | float | 0..1 progress between previous and next stop |
| `delay` | int | **seconds** of delay (positive = late, negative = early); sometimes absent |
| `outside` | bool | off-route flag |
| `angle` | int | bearing in degrees |

Not every field appears on every update — some pushes are partial (only `lat/lon/timestamp/angle/brigade` for low-detail pings).

## Pings
Server sends Engine.IO ping `"2"` every ~25 s. Client must respond with `"3"` to keep the connection alive.

## REST endpoints (for reference)

- `GET /api/<city>/transport` — bulk dataset: `stops[]`, `routes{}`, `specialVehicles[]`, `alerts{alerts, lastUpdated}`, `appAlerts[]`. **3241 stops** for Poznań. Each stop is `[id, name, lat, lon, [zone], [line_ids]]`.
- `GET /api/<city>/search?s=<query>` — fuzzy stop/street search; returns `[{type:"stop", details:[id, name, lat, lon, [zone], {zone:[lines]}, dir]}]`.
- `GET /api/<city>/trip?trip_id=<id>` — `{shape:GeoJSON LineString, stops:[[trip_id, seconds_since_midnight, seq, _, stop_id, name, lat, lon], ...]}`.
- `GET /api/<city>/vehicle?id=<id>` — fleet metadata (model, year, depot, low-floor, A/C, etc.).
- `GET /api/<city>/water-level` (Warsaw only — irrelevant).

All REST endpoints require `Accept: application/json`, `Referer: https://czynaczas.pl/<city>`, browser-like `User-Agent`. Without those headers the server returns 429 with a notice to email the operator.

## Implementation implications

For the MCP, the realtime tool can be a short bounded-time listen:

```python
async with aiohttp.ClientSession(headers=...) as s:
    async with s.ws_connect(URL) as ws:
        await ws.send_str("40")                # connect default namespace
        await ws.send_str(f"40/{city},")       # subscribe city
        # collect frames for N seconds, parse 42/<city>, payload
```

That single listen yields delays + headsigns + next-stop info for **all** active vehicles in the city — enough to answer "kiedy jedzie X z Y na Z?" by filtering vehicles whose `route_id` serves the user's stop, computing arrival via `stop_sequence`/`between`/scheduled-time-from-trip, and applying `delay`.

Static data (`/api/<city>/transport`) gives the stop list and routes; cache it once at startup or per-tool-call.
