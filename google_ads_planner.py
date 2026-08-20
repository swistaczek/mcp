"""FastMCP Server for the Google Ads Keyword Planner (Google Ads API v25, REST).

Exposes the planning half of the Google Ads API — the part that answers
"is anybody searching for this, what does it cost, and what do I get for
my money" — without touching any campaign, budget or bid in a live account.
Everything here is read-only: it plans, forecasts and estimates, it never
mutates the Ads account.

Tools
  google_ads_keyword_metrics  historical search volume for keywords you name
  google_ads_keyword_ideas    keyword expansion from seed terms, a URL or a site
  google_ads_forecast_budget  clicks / cost forecast for a keyword set + max CPC
  google_ads_budget_curve     the same forecast swept across several max CPC bids
  google_ads_seasonality      4-year monthly seasonality index + YoY trend

Implementation notes
  * Calls the REST endpoints on googleads.googleapis.com with aiohttp; the
    heavyweight gRPC `google-ads` client is deliberately NOT used.
  * Auth is a GCP service account (no domain-wide impersonation, no `sub`)
    with the `https://www.googleapis.com/auth/adwords` scope. Tokens are
    minted with `google-auth` in a worker thread and cached until ~60s
    before expiry.
  * Keyword-planning endpoints are capped at 1 QPS per customer id, so every
    request goes through a process-wide throttle (>= 1.1s spacing).
  * Money is returned in account currency dollars (micros / 1_000_000).
  * Google's REST JSON omits unset proto3 optional fields entirely. An absent
    field is reported as None, never as 0 — "no data" and "zero" are different
    answers and conflating them silently ruins every downstream decision.

Environment variables
  GOOGLE_ADS_DEVELOPER_TOKEN          required  developer token of the MCC
  GOOGLE_ADS_SERVICE_ACCOUNT_KEY_B64  required  base64 of the service-account
                                                JSON key (missing '=' padding
                                                is tolerated)
  GOOGLE_ADS_LOGIN_CUSTOMER_ID        required  manager (MCC) id -> sent as the
                                                `login-customer-id` header
  GOOGLE_ADS_CUSTOMER_ID              required  client account id -> URL path
  GOOGLE_ADS_API_VERSION              optional  defaults to 'v25'

All of them are read lazily, at call time. Importing this module with no
environment set succeeds and lists every tool; the missing-credential error
surfaces as a ToolError when a tool is actually invoked.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import aiohttp
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.service_account import Credentials
from pydantic import Field

GOOGLE_ADS_API_BASE = "https://googleads.googleapis.com"
DEFAULT_API_VERSION = "v25"
ADWORDS_SCOPE = "https://www.googleapis.com/auth/adwords"

REQUEST_TIMEOUT_S = 90
MIN_REQUEST_SPACING_S = 1.1   # planning endpoints are 1 QPS per customer id
TOKEN_REFRESH_MARGIN_S = 60

MAX_HISTORICAL_KEYWORDS = 10_000
MAX_IDEA_SEEDS = 20
MAX_CURVE_BIDS = 10
MAX_MONTHS_BACK = 48          # the API keeps ~4 years of monthly history

# Geo target constants (mirrors marmot's set)
GEO_USA = 2840
GEO_UK = 2826
GEO_CANADA = 2124

# Language constants (mirrors marmot's set)
LANG_ENGLISH = 1000
LANG_SPANISH = 1003
LANG_FRENCH = 1002

DEFAULT_GEO_TARGETS = [GEO_USA]
DEFAULT_LANGUAGE = LANG_ENGLISH

MONTH_NAMES = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
]

VALID_MATCH_TYPES = ("BROAD", "PHRASE", "EXACT")
VALID_NETWORKS = ("GOOGLE_SEARCH", "GOOGLE_SEARCH_AND_PARTNERS")
EASY_COMPETITION = ("LOW", "MEDIUM")

FORECAST_NOTE = "Google Ads API v25 does not return impression or CTR forecasts."

mcp = FastMCP("Google Ads Planner")

# --- process-wide mutable state -------------------------------------------

_token_cache: dict[str, Any] = {"token": None, "expires_at": 0.0}
_token_lock = asyncio.Lock()
_throttle_lock = asyncio.Lock()
_last_request_at = 0.0


# ---------------------------------------------------------------------------
# environment / credentials
# ---------------------------------------------------------------------------


def _require_env(name: str) -> str:
    """Read a required env var at call time (never at import). ToolError if unset."""
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ToolError(
            f"Missing required environment variable {name}. The Google Ads planner "
            f"needs GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_SERVICE_ACCOUNT_KEY_B64, "
            f"GOOGLE_ADS_LOGIN_CUSTOMER_ID and GOOGLE_ADS_CUSTOMER_ID."
        )
    return value.strip()


def _api_version() -> str:
    return (os.getenv("GOOGLE_ADS_API_VERSION") or DEFAULT_API_VERSION).strip()


def _digits_only(value: str, label: str) -> str:
    """Customer ids may be written '123-456-7890'; the API wants digits only."""
    digits = re.sub(r"\D", "", value or "")
    if not digits:
        raise ToolError(f"{label} must contain digits (got {value!r}).")
    return digits


def _decode_sa_key(b64: str) -> dict:
    """Decode the base64 service-account JSON key.

    The value is routinely stored with its trailing '=' padding stripped (and
    sometimes wrapped across lines), so whitespace is removed and padding is
    restored before decoding.
    """
    if not b64 or not b64.strip():
        raise ToolError(
            "GOOGLE_ADS_SERVICE_ACCOUNT_KEY_B64 is empty — expected base64 of a "
            "service-account JSON key."
        )
    compact = re.sub(r"\s", "", b64)
    padded = compact + "=" * (-len(compact) % 4)
    try:
        info = json.loads(base64.b64decode(padded))
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same story
        raise ToolError(
            f"GOOGLE_ADS_SERVICE_ACCOUNT_KEY_B64 is not valid base64-encoded JSON: {exc}"
        ) from exc
    if not isinstance(info, dict) or "client_email" not in info or "private_key" not in info:
        raise ToolError(
            "GOOGLE_ADS_SERVICE_ACCOUNT_KEY_B64 decoded, but the JSON has no "
            "'client_email'/'private_key' — that is not a service-account key."
        )
    return info


def _mint_access_token() -> tuple[str, float]:
    """Blocking token mint. Returns (access_token, unix_expiry_seconds).

    Runs under asyncio.to_thread — google-auth's refresh is synchronous and
    performs a network round-trip, which would otherwise stall the event loop.
    """
    info = _decode_sa_key(_require_env("GOOGLE_ADS_SERVICE_ACCOUNT_KEY_B64"))
    try:
        credentials = Credentials.from_service_account_info(info, scopes=[ADWORDS_SCOPE])
        credentials.refresh(GoogleAuthRequest())
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the caller
        raise ToolError(f"Google service-account token request failed: {exc}") from exc
    if not credentials.token:
        raise ToolError("Google service-account token request returned no access token.")
    expiry = getattr(credentials, "expiry", None)
    if isinstance(expiry, datetime):
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        expires_at = expiry.timestamp()
    else:
        expires_at = time.time() + 3300
    return credentials.token, expires_at


async def _access_token() -> str:
    """Cached OAuth access token, re-minted when less than 60s of life remains."""
    async with _token_lock:
        token = _token_cache.get("token")
        expires_at = _token_cache.get("expires_at") or 0.0
        if token and expires_at - time.time() > TOKEN_REFRESH_MARGIN_S:
            return token
        token, expires_at = await asyncio.to_thread(_mint_access_token)
        _token_cache["token"] = token
        _token_cache["expires_at"] = expires_at
        return token


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------


def _micros_to_float(value: Any) -> Optional[float]:
    """Micros (int64, delivered as a JSON string) -> dollars. Absent stays absent.

    None / '' / unparseable -> None. Never 0.0: an absent bid estimate means
    Google has no data, not that the bid is free.
    """
    if value is None or value == "":
        return None
    try:
        return int(value) / 1_000_000
    except (TypeError, ValueError):
        try:
            return float(value) / 1_000_000
        except (TypeError, ValueError):
            return None


def _to_int(value: Any) -> Optional[int]:
    """int64-as-string -> int. Absent -> None (never 0)."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _to_float(value: Any) -> Optional[float]:
    """Float field -> float. Absent -> None (never 0.0)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _geo_path(geo_id: Any) -> str:
    """2840 -> 'geoTargetConstants/2840'."""
    try:
        return f"geoTargetConstants/{int(geo_id)}"
    except (TypeError, ValueError) as exc:
        raise ToolError(f"Invalid geo target id {geo_id!r} — expected a number like 2840.") from exc


def _lang_path(lang_id: Any) -> str:
    """1000 -> 'languageConstants/1000'."""
    try:
        return f"languageConstants/{int(lang_id)}"
    except (TypeError, ValueError) as exc:
        raise ToolError(f"Invalid language id {lang_id!r} — expected a number like 1000.") from exc


def _competition_bucket(index: Any) -> Optional[str]:
    """Competition index [0,100] -> LOW / MEDIUM / HIGH. None index -> None."""
    value = _to_int(index)
    if value is None:
        return None
    if value <= 33:
        return "LOW"
    if value <= 66:
        return "MEDIUM"
    return "HIGH"


def _opportunity_score(volume: Any, competition_index: Any) -> Optional[float]:
    """volume / (competition_index + 1) — high demand, low contest scores best.

    Volume is the driver; the +1 keeps a zero-competition keyword finite. An
    unknown volume yields None (there is nothing to score); an unknown
    competition index is treated as 0, the most optimistic reading.
    """
    searches = _to_int(volume)
    if searches is None:
        return None
    index = _to_int(competition_index) or 0
    return searches / (index + 1)


def _sort_key_volume(item: dict) -> tuple[int, int]:
    """Descending volume with unknown volumes pushed to the back."""
    volume = item.get("avg_monthly_searches")
    return (0 if volume is None else 1, volume or 0)


def _sort_key_opportunity(item: dict) -> tuple[int, float]:
    score = item.get("opportunity_score")
    return (0 if score is None else 1, score or 0.0)


def _parse_year_month(value: str, label: str) -> dict:
    """'2025-01' -> {'year': 2025, 'month': 'JANUARY'} (the API wants the enum name)."""
    match = re.fullmatch(r"(\d{4})-(\d{2})", (value or "").strip())
    if not match:
        raise ToolError(f"{label} must look like '2025-01' (got {value!r}).")
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise ToolError(f"{label} has an impossible month: {value!r}.")
    return {"year": year, "month": MONTH_NAMES[month - 1]}


def _parse_iso_date(value: str, label: str) -> date:
    try:
        return datetime.strptime((value or "").strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ToolError(f"{label} must be an ISO date like '2026-09-01' (got {value!r}).") from exc


def _default_forecast_period() -> tuple[str, str]:
    """Next Monday through 30 days later — always safely in the future."""
    today = date.today()
    start = today + timedelta(days=(7 - today.weekday()) or 7)
    return start.isoformat(), (start + timedelta(days=30)).isoformat()


def _resolve_forecast_period(start_date: Optional[str], end_date: Optional[str]) -> tuple[str, str]:
    """Validate/derive the forecast window: start strictly in the future, end within a year."""
    default_start, default_end = _default_forecast_period()
    start_raw = start_date or default_start
    end_raw = end_date or (
        (_parse_iso_date(start_raw, "start_date") + timedelta(days=30)).isoformat()
        if start_date else default_end
    )
    start = _parse_iso_date(start_raw, "start_date")
    end = _parse_iso_date(end_raw, "end_date")
    today = date.today()
    if start <= today:
        raise ToolError(
            f"start_date must be in the future — Google forecasts a period that has not "
            f"happened yet (got {start.isoformat()}, today is {today.isoformat()})."
        )
    if end < start:
        raise ToolError(
            f"end_date {end.isoformat()} is before start_date {start.isoformat()}."
        )
    if end > today + timedelta(days=365):
        raise ToolError(
            f"end_date must be within one year from today (got {end.isoformat()}; "
            f"the limit is {(today + timedelta(days=365)).isoformat()})."
        )
    return start.isoformat(), end.isoformat()


def _normalize_match_type(match_type: str) -> str:
    value = (match_type or "").strip().upper()
    if value not in VALID_MATCH_TYPES:
        raise ToolError(
            f"match_type must be one of {', '.join(VALID_MATCH_TYPES)} (got {match_type!r})."
        )
    return value


def _normalize_network(network: str) -> str:
    value = (network or "").strip().upper()
    if value not in VALID_NETWORKS:
        raise ToolError(
            f"network must be one of {', '.join(VALID_NETWORKS)} (got {network!r})."
        )
    return value


def _clean_keywords(keywords: Any, label: str, maximum: int) -> list[str]:
    if not keywords:
        raise ToolError(f"{label} must contain at least one keyword.")
    cleaned = [str(k).strip() for k in keywords if str(k).strip()]
    if not cleaned:
        raise ToolError(f"{label} contained only blank entries.")
    if len(cleaned) > maximum:
        raise ToolError(
            f"{label} accepts at most {maximum} keywords per request (got {len(cleaned)})."
        )
    return cleaned


def _money(value: Optional[float]) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _count(value: Optional[int]) -> str:
    return "n/a" if value is None else f"{value:,}"


# ---------------------------------------------------------------------------
# HTTP: throttled POST against the Google Ads REST API
# ---------------------------------------------------------------------------


async def _throttle() -> None:
    """Process-wide 1 QPS gate — planning endpoints reject bursts with RESOURCE_EXHAUSTED."""
    global _last_request_at
    async with _throttle_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < MIN_REQUEST_SPACING_S:
            await asyncio.sleep(MIN_REQUEST_SPACING_S - elapsed)
        _last_request_at = time.monotonic()


def _format_api_error(status: int, body: str) -> str:
    """Unwrap the Google Ads error envelope so the caller sees the real message.

    Shape: {"error": {"message": ..., "status": ...,
             "details": [{"errors": [{"errorCode": {...}, "message": ...}]}]}}
    """
    try:
        error = json.loads(body).get("error", {}) or {}
    except (ValueError, AttributeError):
        return f"Google Ads API HTTP {status}: {body[:500]}"
    parts: list[str] = []
    for detail in error.get("details", []) or []:
        for sub in detail.get("errors", []) or []:
            code = sub.get("errorCode") or {}
            code_text = ", ".join(f"{k}={v}" for k, v in code.items()) if code else "unknown"
            parts.append(f"[{code_text}] {sub.get('message', '')}".strip())
    headline = error.get("message") or f"HTTP {status}"
    status_text = error.get("status")
    summary = f"Google Ads API error (HTTP {status}"
    summary += f", {status_text})" if status_text else ")"
    summary += f": {headline}"
    if parts:
        summary += " | " + " | ".join(parts)
    return summary


async def _post(endpoint: str, payload: dict, ctx: Optional[Context] = None) -> dict:
    """POST one planning request to /{version}/customers/{customer_id}:{endpoint}.

    Handles auth, the 1 QPS throttle, and error-envelope unwrapping. Returns the
    parsed JSON body; raises ToolError with Google's own message on failure.
    """
    method = endpoint.lstrip(":")
    developer_token = _require_env("GOOGLE_ADS_DEVELOPER_TOKEN")
    login_customer_id = _digits_only(
        _require_env("GOOGLE_ADS_LOGIN_CUSTOMER_ID"), "GOOGLE_ADS_LOGIN_CUSTOMER_ID"
    )
    customer_id = _digits_only(_require_env("GOOGLE_ADS_CUSTOMER_ID"), "GOOGLE_ADS_CUSTOMER_ID")
    token = await _access_token()
    url = f"{GOOGLE_ADS_API_BASE}/{_api_version()}/customers/{customer_id}:{method}"
    headers = {
        "Authorization": f"Bearer {token}",
        "developer-token": developer_token,
        "login-customer-id": login_customer_id,
        "Content-Type": "application/json",
    }
    await _throttle()
    if ctx:
        await ctx.info(f"POST {method} (customer {customer_id})")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
            ) as response:
                body = await response.text()
                if response.status >= 400:
                    raise ToolError(_format_api_error(response.status, body))
                try:
                    return json.loads(body) if body else {}
                except ValueError as exc:
                    raise ToolError(
                        f"Google Ads API returned a non-JSON body for {method}: {body[:300]}"
                    ) from exc
    except asyncio.TimeoutError as exc:
        raise ToolError(
            f"Google Ads API request timed out after {REQUEST_TIMEOUT_S}s ({method})."
        ) from exc
    except aiohttp.ClientError as exc:
        raise ToolError(f"Google Ads API request failed ({method}): {exc}") from exc


# ---------------------------------------------------------------------------
# response parsing
# ---------------------------------------------------------------------------


def _parse_monthly_volumes(metrics: dict) -> list[dict]:
    """monthlySearchVolumes[] -> [{'year': int, 'month': 'JANUARY', 'searches': int|None}]."""
    rows: list[dict] = []
    for entry in metrics.get("monthlySearchVolumes", []) or []:
        year = _to_int(entry.get("year"))
        rows.append({
            "year": year,
            "month": entry.get("month"),
            "searches": _to_int(entry.get("monthlySearches")),
        })
    return rows


def _parse_keyword_metrics(metrics: dict) -> dict:
    """The shared numeric block of both keyword endpoints, presence-safe."""
    competition_index = _to_int(metrics.get("competitionIndex"))
    return {
        "avg_monthly_searches": _to_int(metrics.get("avgMonthlySearches")),
        "competition": metrics.get("competition") or _competition_bucket(competition_index),
        "competition_index": competition_index,
        "low_top_of_page_bid": _micros_to_float(metrics.get("lowTopOfPageBidMicros")),
        "high_top_of_page_bid": _micros_to_float(metrics.get("highTopOfPageBidMicros")),
    }


def _parse_forecast_metrics(payload: dict) -> dict:
    """campaignForecastMetrics -> the v25 fields only (no impressions, no CTR)."""
    metrics = payload.get("campaignForecastMetrics", {}) or {}
    clicks = _to_float(metrics.get("clicks"))
    cost = _micros_to_float(metrics.get("costMicros"))
    cost_per_click = cost / clicks if (cost is not None and clicks) else None
    return {
        "clicks": clicks,
        "cost": cost,
        "average_cpc": _micros_to_float(metrics.get("averageCpcMicros")),
        "conversions": _to_float(metrics.get("conversions")),
        "average_cpa": _micros_to_float(metrics.get("averageCpaMicros")),
        "cost_per_click": cost_per_click,
    }


# ---------------------------------------------------------------------------
# request builders shared by several tools
# ---------------------------------------------------------------------------


async def _historical_metrics(
    keywords: list[str],
    geo_target_ids: list[int],
    language_id: int,
    network: str,
    start_year_month: Optional[str],
    end_year_month: Optional[str],
    ctx: Optional[Context],
) -> list[dict]:
    """One :generateKeywordHistoricalMetrics call -> normalised keyword rows."""
    payload: dict[str, Any] = {
        "keywords": keywords,
        "language": _lang_path(language_id),
        "geoTargetConstants": [_geo_path(g) for g in geo_target_ids],
        "keywordPlanNetwork": network,
    }
    if start_year_month and end_year_month:
        payload["historicalMetricsOptions"] = {
            "yearMonthRange": {
                "start": _parse_year_month(start_year_month, "start_year_month"),
                "end": _parse_year_month(end_year_month, "end_year_month"),
            }
        }
    data = await _post("generateKeywordHistoricalMetrics", payload, ctx)
    rows: list[dict] = []
    for result in data.get("results", []) or []:
        metrics = result.get("keywordMetrics", {}) or {}
        row = {"text": result.get("text")}
        row.update(_parse_keyword_metrics(metrics))
        row["monthly_volumes"] = _parse_monthly_volumes(metrics)
        row["close_variants"] = list(result.get("closeVariants", []) or [])
        rows.append(row)
    return rows


async def _forecast_once(
    keywords: list[str],
    match_type: str,
    max_cpc_bid: float,
    daily_budget: Optional[float],
    start_date: str,
    end_date: str,
    geo_target_ids: list[int],
    language_id: int,
    ctx: Optional[Context],
) -> dict:
    """One :generateKeywordForecastMetrics call.

    The v25 request shape is camelCase and unforgiving — several plausible-looking
    field names (dailyTargetSpendMicros, geoModifiers, biddableKeywords) were
    removed in v24 and are rejected outright. Do not "improve" this payload.
    """
    bidding: dict[str, str] = {"maxCpcBidMicros": str(int(round(max_cpc_bid * 1_000_000)))}
    if daily_budget is not None:
        bidding["dailyBudgetMicros"] = str(int(round(daily_budget * 1_000_000)))
    payload = {
        "campaign": {
            "languageConstants": [_lang_path(language_id)],
            "geoTargetConstants": [_geo_path(g) for g in geo_target_ids],
            "biddingStrategy": {"manualCpcBiddingStrategy": bidding},
            "adGroups": [
                {"keywords": [{"text": text, "matchType": match_type} for text in keywords]}
            ],
        },
        "forecastPeriod": {"startDate": start_date, "endDate": end_date},
    }
    data = await _post("generateKeywordForecastMetrics", payload, ctx)
    return _parse_forecast_metrics(data)


# ---------------------------------------------------------------------------
# seasonality maths
# ---------------------------------------------------------------------------


def _seasonality_window(months_back: int) -> tuple[str, str]:
    """Year-month range ending with the last complete month."""
    today = date.today()
    end_year, end_month = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    total = end_year * 12 + (end_month - 1) - (months_back - 1)
    start_year, start_month = divmod(total, 12)
    return f"{start_year:04d}-{start_month + 1:02d}", f"{end_year:04d}-{end_month:02d}"


def _index_months(monthly: list[dict]) -> tuple[float, list[dict]]:
    """Attach a seasonality index (month / mean month) to each month row."""
    values = [row["searches"] for row in monthly if row.get("searches") is not None]
    mean = (sum(values) / len(values)) if values else 0.0
    indexed = []
    for row in monthly:
        searches = row.get("searches")
        indexed.append({
            "year": row.get("year"),
            "month": row.get("month"),
            "searches": searches,
            "index": (searches / mean) if (searches is not None and mean > 0) else None,
        })
    return mean, indexed


def _yoy_change(monthly: list[dict]) -> Optional[float]:
    """Last 12 months vs the 12 before them, as a fraction (0.25 = +25%)."""
    values = [row.get("searches") or 0 for row in monthly]
    if len(values) < 24:
        return None
    recent = sum(values[-12:])
    prior = sum(values[-24:-12])
    if prior <= 0:
        return None
    return (recent - prior) / prior


def _extreme_month(indexed: list[dict], highest: bool) -> Optional[str]:
    candidates = [row for row in indexed if row.get("index") is not None]
    if not candidates:
        return None
    chooser = max if highest else min
    return chooser(candidates, key=lambda row: row["index"])["month"]


def _aggregate_seasonality(keyword_rows: list[dict]) -> dict:
    """Calendar-month index across every keyword and every year in the window."""
    totals: dict[str, int] = {}
    for keyword in keyword_rows:
        for row in keyword.get("monthly", []):
            searches = row.get("searches")
            month = row.get("month")
            if searches is None or not month:
                continue
            totals[month] = totals.get(month, 0) + searches
    if not totals:
        return {"monthly_index": {}, "peak_month": None, "trough_month": None}
    mean = sum(totals.values()) / len(totals)
    monthly_index = {
        month: (totals[month] / mean if mean > 0 else None)
        for month in MONTH_NAMES
        if month in totals
    }
    peak = max(totals, key=lambda m: totals[m])
    trough = min(totals, key=lambda m: totals[m])
    return {"monthly_index": monthly_index, "peak_month": peak, "trough_month": trough}


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="google_ads_keyword_metrics",
    description=(
        "Historical Google search demand for keywords you name explicitly — average "
        "monthly searches, competition, and the top-of-page bid range advertisers pay. "
        "Use this when you already have the terms ('running shoes', 'mkbhd merch', "
        "'custom t shirts') and want the numbers; use google_ads_keyword_ideas when you "
        "need to discover terms instead. Volume is the 12-month average over the chosen "
        "geo and language, rounded into Google's buckets. Competition is LOW / MEDIUM / "
        "HIGH with a 0-100 competition_index (0-33 LOW, 34-66 MEDIUM, 67-100 HIGH) that "
        "measures how many advertisers bid on the term, NOT how hard it is to rank "
        "organically. low_top_of_page_bid / high_top_of_page_bid are the 20th/80th "
        "percentile of what advertisers paid, in the Ads account's currency. "
        "Any of these can come back null: Google omits metrics for low-volume or "
        "commercially uninteresting keywords, and null means 'no data', NOT zero. "
        "monthly_volumes gives up to 4 years of month-by-month searches; close_variants "
        "lists the near-duplicate queries Google folded into the same row (so 'running "
        "shoe' and 'running shoes' report one shared volume). Accepts up to 10,000 "
        "keywords per call. Pass start_year_month/end_year_month ('2025-01' to '2025-12') "
        "to pin the history window."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def google_ads_keyword_metrics(
    keywords: list[str] = Field(
        description=(
            "Keywords to look up, e.g. ['running shoes', 'trail running shoes', "
            "'mkbhd merch']. Max 10,000 per call. Case-insensitive; Google normalises them."
        ),
    ),
    geo_target_ids: list[int] = Field(
        default=DEFAULT_GEO_TARGETS,
        description=(
            "Google geo target constant ids. 2840 = United States (default), "
            "2826 = United Kingdom, 2124 = Canada. Multiple ids are summed into one "
            "combined market, e.g. [2840, 2124] for US+Canada."
        ),
    ),
    language_id: int = Field(
        default=LANG_ENGLISH,
        description="Language constant id: 1000 = English (default), 1003 = Spanish, 1002 = French.",
    ),
    network: str = Field(
        default="GOOGLE_SEARCH",
        description=(
            "'GOOGLE_SEARCH' (default, google.com only — the conservative number) or "
            "'GOOGLE_SEARCH_AND_PARTNERS' (adds search partner sites; larger volumes)."
        ),
    ),
    start_year_month: Optional[str] = Field(
        default=None,
        description=(
            "Optional start of the history window as 'YYYY-MM', e.g. '2025-01'. Only "
            "applied when end_year_month is given too; otherwise Google picks the window."
        ),
    ),
    end_year_month: Optional[str] = Field(
        default=None,
        description="Optional end of the history window as 'YYYY-MM', e.g. '2025-12'.",
    ),
    ctx: Context = None,
) -> ToolResult:
    cleaned = _clean_keywords(keywords, "keywords", MAX_HISTORICAL_KEYWORDS)
    resolved_network = _normalize_network(network)
    geo_targets = list(geo_target_ids or DEFAULT_GEO_TARGETS)
    if ctx:
        await ctx.info(f"Fetching historical metrics for {len(cleaned)} keyword(s)")
    rows = await _historical_metrics(
        cleaned, geo_targets, language_id, resolved_network,
        start_year_month, end_year_month, ctx,
    )
    if ctx and not rows:
        await ctx.warning("Google returned no rows — the keywords may have no measurable volume.")
    ranked = sorted(rows, key=_sort_key_volume, reverse=True)
    headline = ", ".join(
        f"{row['text']} {_count(row['avg_monthly_searches'])}/mo"
        f" ({row['competition'] or 'competition n/a'})"
        for row in ranked[:5]
    )
    text = (
        f"Historical metrics for {len(cleaned)} keyword(s), {len(rows)} returned "
        f"(geo {geo_targets}, language {language_id}, {resolved_network})."
    )
    if headline:
        text += f" Top: {headline}."
    return ToolResult(
        content=text,
        structured_content={
            "keywords": rows,
            "requested": len(cleaned),
            "returned": len(rows),
            "geo_target_ids": geo_targets,
            "language_id": language_id,
            "network": resolved_network,
        },
    )


@mcp.tool(
    name="google_ads_keyword_ideas",
    description=(
        "Discover keywords: expand a handful of seed terms, a landing page URL, or a whole "
        "site into related queries people actually search, each with volume, competition "
        "and bid range. Provide EXACTLY ONE seed kind — seed_keywords (up to 20 terms, e.g. "
        "['running shoes', 'trail shoes']), seed_url (one page, e.g. "
        "'https://example.com/shop/running-shoes') or seed_site (a domain, e.g. "
        "'example.com', crawled for themes). Every idea carries an opportunity_score = "
        "avg_monthly_searches / (competition_index + 1), which favours high-demand terms "
        "few advertisers are bidding on; it is a ranking heuristic, not a Google metric, "
        "and is null when Google reports no volume. Three ready-made shortlists come back "
        "alongside the full list: top_by_volume (15 biggest markets), top_opportunities "
        "(10 best demand-to-competition ratios), and easy_targets (10 with LOW or MEDIUM "
        "competition — the cheap entry points). Nulls mean 'Google has no data', never "
        "zero. Results are capped at `limit`; this tool never auto-paginates."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def google_ads_keyword_ideas(
    seed_keywords: Optional[list[str]] = Field(
        default=None,
        description=(
            "Seed terms to expand, e.g. ['custom t shirts', 'band merch']. "
            "Maximum 20 — Google rejects more. Mutually exclusive with seed_url/seed_site."
        ),
    ),
    seed_url: Optional[str] = Field(
        default=None,
        description=(
            "A single page URL to derive ideas from, e.g. "
            "'https://example.com/collections/hoodies'. Mutually exclusive with the other seeds."
        ),
    ),
    seed_site: Optional[str] = Field(
        default=None,
        description=(
            "A whole site to derive ideas from, e.g. 'example.com'. Broader and vaguer than "
            "seed_url. Mutually exclusive with the other seeds."
        ),
    ),
    geo_target_ids: list[int] = Field(
        default=DEFAULT_GEO_TARGETS,
        description="Geo target constant ids: 2840 = US (default), 2826 = UK, 2124 = Canada.",
    ),
    language_id: int = Field(
        default=LANG_ENGLISH,
        description="Language constant id: 1000 = English (default), 1003 = Spanish, 1002 = French.",
    ),
    network: str = Field(
        default="GOOGLE_SEARCH",
        description="'GOOGLE_SEARCH' (default) or 'GOOGLE_SEARCH_AND_PARTNERS'.",
    ),
    limit: int = Field(
        default=50,
        description="Maximum ideas to return, e.g. 50 (default) or 200. Only the first page is read.",
        ge=1,
        le=1000,
    ),
    ctx: Context = None,
) -> ToolResult:
    provided = [
        ("keywords", seed_keywords if seed_keywords else None),
        ("url", (seed_url or "").strip() or None),
        ("site", (seed_site or "").strip() or None),
    ]
    chosen = [(kind, value) for kind, value in provided if value]
    if len(chosen) != 1:
        raise ToolError(
            "Provide exactly one seed: seed_keywords, seed_url or seed_site "
            f"(got {len(chosen)})."
        )
    seed_type, seed_value = chosen[0]
    resolved_network = _normalize_network(network)
    geo_targets = list(geo_target_ids or DEFAULT_GEO_TARGETS)

    payload: dict[str, Any] = {
        "language": _lang_path(language_id),
        "geoTargetConstants": [_geo_path(g) for g in geo_targets],
        "keywordPlanNetwork": resolved_network,
    }
    if seed_type == "keywords":
        seeds = _clean_keywords(seed_value, "seed_keywords", MAX_IDEA_SEEDS)
        payload["keywordSeed"] = {"keywords": seeds}
        seed_value = seeds
    elif seed_type == "url":
        payload["urlSeed"] = {"url": seed_value}
    else:
        payload["siteSeed"] = {"site": seed_value}

    if ctx:
        await ctx.info(f"Generating keyword ideas from {seed_type} seed")
    data = await _post("generateKeywordIdeas", payload, ctx)

    ideas: list[dict] = []
    for result in data.get("results", []) or []:
        metrics = result.get("keywordIdeaMetrics", {}) or {}
        idea = {"text": result.get("text")}
        idea.update(_parse_keyword_metrics(metrics))
        idea["opportunity_score"] = _opportunity_score(
            idea["avg_monthly_searches"], idea["competition_index"]
        )
        ideas.append(idea)

    ideas.sort(key=_sort_key_volume, reverse=True)
    ideas = ideas[:limit]
    if ctx and not ideas:
        await ctx.warning("Google returned no keyword ideas for this seed.")

    top_by_volume = ideas[:15]
    # Only ideas Google actually scored can rank as opportunities. An idea with no
    # metrics block has opportunity_score None, meaning "no data" — ranking it
    # alongside scored ideas would present an unknown as a recommendation.
    scored = [idea for idea in ideas if idea.get("opportunity_score") is not None]
    top_opportunities = sorted(scored, key=_sort_key_opportunity, reverse=True)[:10]
    easy_targets = [i for i in ideas if (i.get("competition") in EASY_COMPETITION)][:10]

    best = ideas[0] if ideas else None
    text = (
        f"{len(ideas)} keyword idea(s) from {seed_type} seed "
        f"(geo {geo_targets}, language {language_id}, {resolved_network})."
    )
    if best:
        text += (
            f" Biggest: '{best['text']}' at {_count(best['avg_monthly_searches'])}/mo "
            f"({best['competition'] or 'competition n/a'})."
        )
    if easy_targets:
        text += f" {len(easy_targets)} low/medium-competition target(s) shortlisted."
    return ToolResult(
        content=text,
        structured_content={
            "ideas": ideas,
            "top_by_volume": top_by_volume,
            "top_opportunities": top_opportunities,
            "easy_targets": easy_targets,
            "seed": {"type": seed_type, "value": seed_value},
            "total_returned": len(ideas),
        },
    )


@mcp.tool(
    name="google_ads_forecast_budget",
    description=(
        "Forecast what a keyword set would deliver on Google Search over a future date "
        "range at a given max CPC bid: expected clicks, total cost, average CPC, "
        "conversions and average CPA — all in the Ads account's currency. This is "
        "Google's own Keyword Planner forecast, not an extrapolation from historical "
        "volume, so it accounts for auction dynamics: raising max_cpc_bid wins more "
        "auctions and buys more clicks, with diminishing returns. match_type controls "
        "reach: BROAD (widest, cheapest per click, least precise), PHRASE, or EXACT "
        "(narrowest, most qualified). Optionally cap spend with daily_budget. The "
        "forecast period must START IN THE FUTURE and END WITHIN ONE YEAR; leave the "
        "dates empty for the default next-Monday-plus-30-days window. IMPORTANT: Google "
        "Ads API v25 removed impression and CTR forecasts, so this tool returns neither "
        "— any impression figure would be fabricated. Nulls mean Google declined to "
        "forecast that metric (usually too little data), not zero. For a bid-by-bid "
        "comparison use google_ads_budget_curve instead of calling this repeatedly."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def google_ads_forecast_budget(
    keywords: list[str] = Field(
        description="Keywords for the hypothetical ad group, e.g. ['custom t shirts', 'band merch'].",
    ),
    max_cpc_bid: float = Field(
        description=(
            "Maximum cost-per-click bid in account currency dollars, e.g. 2.0 for $2.00. "
            "This is the main lever: higher bid = more clicks at a higher average CPC."
        ),
        gt=0,
    ),
    match_type: str = Field(
        default="BROAD",
        description="'BROAD' (default), 'PHRASE' or 'EXACT'. Applied to every keyword in the set.",
    ),
    daily_budget: Optional[float] = Field(
        default=None,
        description=(
            "Optional daily spend cap in dollars, e.g. 50.0 for $50/day. Omit for an "
            "uncapped forecast that shows the full available demand."
        ),
        gt=0,
    ),
    start_date: Optional[str] = Field(
        default=None,
        description=(
            "Forecast start as 'YYYY-MM-DD', e.g. '2026-09-01'. Must be in the future. "
            "Defaults to next Monday."
        ),
    ),
    end_date: Optional[str] = Field(
        default=None,
        description=(
            "Forecast end as 'YYYY-MM-DD', e.g. '2026-09-30'. Must be within one year of "
            "today. Defaults to 30 days after the start date."
        ),
    ),
    geo_target_ids: list[int] = Field(
        default=DEFAULT_GEO_TARGETS,
        description="Geo target constant ids: 2840 = US (default), 2826 = UK, 2124 = Canada.",
    ),
    language_id: int = Field(
        default=LANG_ENGLISH,
        description="Language constant id: 1000 = English (default), 1003 = Spanish, 1002 = French.",
    ),
    ctx: Context = None,
) -> ToolResult:
    cleaned = _clean_keywords(keywords, "keywords", MAX_HISTORICAL_KEYWORDS)
    resolved_match = _normalize_match_type(match_type)
    start, end = _resolve_forecast_period(start_date, end_date)
    geo_targets = list(geo_target_ids or DEFAULT_GEO_TARGETS)
    if ctx:
        await ctx.info(
            f"Forecasting {len(cleaned)} keyword(s) at ${max_cpc_bid:.2f} max CPC, {start}..{end}"
        )
    forecast = await _forecast_once(
        cleaned, resolved_match, max_cpc_bid, daily_budget, start, end,
        geo_targets, language_id, ctx,
    )
    if ctx and forecast["clicks"] is None:
        await ctx.warning("Google returned no click forecast — too little data for these keywords.")
    clicks = forecast["clicks"]
    text = (
        f"Forecast {start}..{end} at ${max_cpc_bid:.2f} max CPC "
        f"({resolved_match}, {len(cleaned)} keyword(s)): "
        f"{'n/a' if clicks is None else format(clicks, ',.0f')} clicks, "
        f"{_money(forecast['cost'])} cost, {_money(forecast['average_cpc'])} avg CPC. "
        f"{FORECAST_NOTE}"
    )
    return ToolResult(
        content=text,
        structured_content={
            "clicks": forecast["clicks"],
            "cost": forecast["cost"],
            "average_cpc": forecast["average_cpc"],
            "conversions": forecast["conversions"],
            "average_cpa": forecast["average_cpa"],
            "cost_per_click": forecast["cost_per_click"],
            "period": {"start_date": start, "end_date": end},
            "max_cpc_bid": max_cpc_bid,
            "daily_budget": daily_budget,
            "keywords": cleaned,
            "match_type": resolved_match,
            "note": FORECAST_NOTE,
        },
    )


@mcp.tool(
    name="google_ads_budget_curve",
    description=(
        "Sweep a keyword set across several max CPC bids and return the resulting "
        "clicks/cost curve — the answer to 'how much more traffic does another dollar of "
        "bid actually buy?'. Runs one forecast per bid, sequentially, because the "
        "planning API allows only 1 request per second; six bids therefore take roughly "
        "seven seconds and progress is reported as it goes. Each row adds marginal "
        "economics against the previous (cheaper) bid: marginal_clicks, marginal_cost and "
        "marginal_cost_per_click = the true incremental price of the extra clicks that "
        "bid step buys, which is always higher than the headline average CPC. "
        "best_efficiency_bid names the step with the cheapest incremental clicks — the "
        "point where you are still buying traffic efficiently. The first row has null "
        "marginals (nothing to compare against). Bids are dollars, e.g. "
        "[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]; at most 10 per call. Like the single forecast, "
        "no impression or CTR data exists in Google Ads API v25."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def google_ads_budget_curve(
    keywords: list[str] = Field(
        description="Keywords for the hypothetical ad group, e.g. ['running shoes', 'trail shoes'].",
    ),
    match_type: str = Field(
        default="BROAD",
        description="'BROAD' (default), 'PHRASE' or 'EXACT'. Applied to every keyword in the set.",
    ),
    bids: list[float] = Field(
        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        description=(
            "Max CPC bids to test, in dollars, e.g. [0.25, 0.5, 1.0, 2.0, 4.0, 8.0] "
            "(default). Maximum 10 — each one costs a second of wall clock."
        ),
    ),
    start_date: Optional[str] = Field(
        default=None,
        description="Forecast start 'YYYY-MM-DD', e.g. '2026-09-01'. Future only. Defaults to next Monday.",
    ),
    end_date: Optional[str] = Field(
        default=None,
        description=(
            "Forecast end 'YYYY-MM-DD', e.g. '2026-09-30'. Within one year of today. "
            "Defaults to start + 30 days."
        ),
    ),
    geo_target_ids: list[int] = Field(
        default=DEFAULT_GEO_TARGETS,
        description="Geo target constant ids: 2840 = US (default), 2826 = UK, 2124 = Canada.",
    ),
    language_id: int = Field(
        default=LANG_ENGLISH,
        description="Language constant id: 1000 = English (default), 1003 = Spanish, 1002 = French.",
    ),
    ctx: Context = None,
) -> ToolResult:
    cleaned = _clean_keywords(keywords, "keywords", MAX_HISTORICAL_KEYWORDS)
    resolved_match = _normalize_match_type(match_type)
    if not bids:
        raise ToolError("bids must contain at least one max CPC bid, e.g. [0.5, 1.0, 2.0].")
    if len(bids) > MAX_CURVE_BIDS:
        raise ToolError(
            f"bids accepts at most {MAX_CURVE_BIDS} values per call (got {len(bids)}) — "
            f"the planning API is limited to 1 request per second."
        )
    try:
        ladder = sorted(float(b) for b in bids)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"bids must be numbers in dollars, e.g. [0.5, 1.0, 2.0] (got {bids!r}).") from exc
    if any(b <= 0 for b in ladder):
        raise ToolError(f"every bid must be greater than 0 (got {bids!r}).")

    start, end = _resolve_forecast_period(start_date, end_date)
    geo_targets = list(geo_target_ids or DEFAULT_GEO_TARGETS)
    if ctx:
        await ctx.info(f"Sweeping {len(ladder)} bid(s) over {len(cleaned)} keyword(s), {start}..{end}")

    curve: list[dict] = []
    previous: Optional[dict] = None
    for position, bid in enumerate(ladder):
        forecast = await _forecast_once(
            cleaned, resolved_match, bid, None, start, end, geo_targets, language_id, ctx,
        )
        row = {
            "max_cpc_bid": bid,
            "clicks": forecast["clicks"],
            "cost": forecast["cost"],
            "average_cpc": forecast["average_cpc"],
            "marginal_clicks": None,
            "marginal_cost": None,
            "marginal_cost_per_click": None,
        }
        if previous is not None:
            if row["clicks"] is not None and previous["clicks"] is not None:
                row["marginal_clicks"] = row["clicks"] - previous["clicks"]
            if row["cost"] is not None and previous["cost"] is not None:
                row["marginal_cost"] = row["cost"] - previous["cost"]
            if row["marginal_clicks"] and row["marginal_cost"] is not None and row["marginal_clicks"] > 0:
                row["marginal_cost_per_click"] = row["marginal_cost"] / row["marginal_clicks"]
        curve.append(row)
        previous = row
        if ctx:
            await ctx.report_progress(
                position + 1, len(ladder), message=f"forecast at ${bid:.2f} max CPC"
            )

    efficient = [r for r in curve if r["marginal_cost_per_click"] is not None]
    best_efficiency_bid = (
        min(efficient, key=lambda r: r["marginal_cost_per_click"])["max_cpc_bid"]
        if efficient else None
    )
    top = curve[-1]
    text = (
        f"Budget curve over {len(ladder)} bid(s) for {len(cleaned)} keyword(s), {start}..{end} "
        f"({resolved_match}). At the top bid ${top['max_cpc_bid']:.2f}: "
        f"{'n/a' if top['clicks'] is None else format(top['clicks'], ',.0f')} clicks for "
        f"{_money(top['cost'])}."
    )
    if best_efficiency_bid is not None:
        text += f" Cheapest incremental clicks at ${best_efficiency_bid:.2f}."
    text += f" {FORECAST_NOTE}"
    return ToolResult(
        content=text,
        structured_content={
            "curve": curve,
            "period": {"start_date": start, "end_date": end},
            "keywords": cleaned,
            "match_type": resolved_match,
            "best_efficiency_bid": best_efficiency_bid,
        },
    )


@mcp.tool(
    name="google_ads_seasonality",
    description=(
        "Month-by-month seasonality for a set of keywords, derived from up to four years "
        "of Google search history (the maximum the API retains). For every keyword and "
        "for the combined aggregate it returns a seasonality index per month where 1.0 = "
        "an average month, 1.8 = 80% above average, 0.4 = 60% below — so 'christmas "
        "sweater' peaks in NOVEMBER/DECEMBER and troughs in JUNE. peak_month and "
        "trough_month name the extremes, and yoy_change compares the last 12 months "
        "against the 12 before them as a fraction (0.25 = +25% year over year, -0.1 = "
        "-10%), which separates a genuinely growing market from one that merely looks "
        "busy in season. Use it to time campaign launches, budget flighting and content "
        "calendars. Months where Google reports no data appear with null searches and a "
        "null index rather than a zero, and yoy_change is null when fewer than 24 months "
        "of history came back."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def google_ads_seasonality(
    keywords: list[str] = Field(
        description=(
            "Keywords to profile, e.g. ['christmas sweater', 'halloween costume', "
            "'running shoes']. Keep the set thematically related for a meaningful aggregate."
        ),
    ),
    months_back: int = Field(
        default=48,
        description=(
            "How many months of history to request, e.g. 48 (default, the API's 4-year "
            "maximum) or 24 for a two-year view. Values above 48 are clamped."
        ),
        ge=1,
    ),
    geo_target_ids: list[int] = Field(
        default=DEFAULT_GEO_TARGETS,
        description="Geo target constant ids: 2840 = US (default), 2826 = UK, 2124 = Canada.",
    ),
    language_id: int = Field(
        default=LANG_ENGLISH,
        description="Language constant id: 1000 = English (default), 1003 = Spanish, 1002 = French.",
    ),
    ctx: Context = None,
) -> ToolResult:
    cleaned = _clean_keywords(keywords, "keywords", MAX_HISTORICAL_KEYWORDS)
    months = min(int(months_back), MAX_MONTHS_BACK)
    geo_targets = list(geo_target_ids or DEFAULT_GEO_TARGETS)
    start_ym, end_ym = _seasonality_window(months)
    if ctx:
        await ctx.info(
            f"Fetching {months} months of history ({start_ym}..{end_ym}) for "
            f"{len(cleaned)} keyword(s)"
        )
    rows = await _historical_metrics(
        cleaned, geo_targets, language_id, "GOOGLE_SEARCH", start_ym, end_ym, ctx,
    )
    if ctx and not rows:
        await ctx.warning("Google returned no history — the keywords may have no measurable volume.")

    keyword_rows: list[dict] = []
    for row in rows:
        monthly = row.get("monthly_volumes", [])
        mean, indexed = _index_months(monthly)
        total = sum(m["searches"] for m in monthly if m.get("searches") is not None)
        keyword_rows.append({
            "text": row.get("text"),
            "total_searches": total,
            "mean_monthly": mean,
            "monthly": indexed,
            "peak_month": _extreme_month(indexed, highest=True),
            "trough_month": _extreme_month(indexed, highest=False),
            "yoy_change": _yoy_change(indexed),
        })

    aggregate = _aggregate_seasonality(keyword_rows)
    text = (
        f"Seasonality over {months} month(s) ({start_ym}..{end_ym}) for "
        f"{len(keyword_rows)} keyword(s), geo {geo_targets}."
    )
    if aggregate["peak_month"]:
        text += (
            f" Aggregate peaks in {aggregate['peak_month']}, troughs in "
            f"{aggregate['trough_month']}."
        )
    yoys = [k["yoy_change"] for k in keyword_rows if k["yoy_change"] is not None]
    if yoys:
        text += f" Mean YoY change {sum(yoys) / len(yoys):+.1%}."
    return ToolResult(
        content=text,
        structured_content={
            "keywords": keyword_rows,
            "aggregate": aggregate,
            "months_requested": months,
            "geo_target_ids": geo_targets,
            "language_id": language_id,
        },
    )


if __name__ == "__main__":
    mcp.run()
