"""FastMCP Server for the Polish KRS (Krajowy Rejestr Sądowy) Open API.

Wraps the public api-krs.ms.gov.pl endpoints to fetch the current odpis
("odpis aktualny") and the full historical odpis ("odpis pełny") of a KRS
entity (company, foundation, association, ...).

Port of https://github.com/pkolawa/krs-poland-mcp-server (TypeScript) to
FastMCP, with the response shape adapted to what the live API actually returns
(`{"odpis": {"naglowekA": ..., "dane": {"dzial1": {...}}}}`).
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import aiohttp
from fastmcp import Context, FastMCP
from fastmcp.tools.tool import ToolResult
from pydantic import Field

mcp = FastMCP("KRS Poland")

KRS_API_BASE = "https://api-krs.ms.gov.pl/api/krs"
USER_AGENT = "krs-mcp/1.0"
REQUEST_TIMEOUT_S = 15
RETRY_DELAYS_S = (2, 5)


def _build_extract_urls(extract_type: str, rejestr: str, krs: str) -> list[str]:
    """Both the working query-param form and the path-form fallback (in case MS
    routing changes). The query-param form is the one that works today."""
    return [
        f"{KRS_API_BASE}/{extract_type}/{krs}?rejestr={rejestr}&format=json",
        f"{KRS_API_BASE}/{extract_type}/{rejestr}/{krs}?format=json",
    ]


async def _fetch_one(session: aiohttp.ClientSession, url: str) -> tuple[int, Any]:
    """Single request, no retry. Returns (status, parsed_json_or_None).

    Status 204 → empty body, returns (204, None). Non-2xx (other than 204) raises.
    """
    async with session.get(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
    ) as r:
        if r.status == 204:
            return 204, None
        if r.status == 404:
            return 404, None
        r.raise_for_status()
        return r.status, await r.json()


async def _fetch_extract(extract_type: str, rejestr: str, krs: str,
                         ctx: Optional[Context] = None) -> tuple[int, Any]:
    """Try each candidate URL with bounded retries. Returns (final_status, body)."""
    last_status = 0
    for url in _build_extract_urls(extract_type, rejestr, krs):
        async with aiohttp.ClientSession() as session:
            for attempt in range(len(RETRY_DELAYS_S) + 1):
                try:
                    status, body = await _fetch_one(session, url)
                    if status == 200 and body is not None:
                        return 200, body
                    # 204/404: don't retry — try the next URL form
                    last_status = status
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    if ctx:
                        await ctx.warning(f"KRS request failed ({url}): {exc}")
                    if attempt < len(RETRY_DELAYS_S):
                        await asyncio.sleep(RETRY_DELAYS_S[attempt])
                    else:
                        last_status = -1
    return last_status, None


def _format_headline(extract: dict) -> str:
    """One-line summary from the real API shape:
    `odpis.dane.dzial1.danePodmiotu.nazwa` + `odpis.naglowekA.numerKRS`."""
    odpis = extract.get("odpis", extract) if isinstance(extract, dict) else {}
    naglowek = odpis.get("naglowekA", {}) or {}
    podmiot = (
        odpis.get("dane", {}).get("dzial1", {}).get("danePodmiotu", {}) or {}
    )
    nazwa = podmiot.get("nazwa") or "Nieznana nazwa"
    numer = naglowek.get("numerKRS") or "?"
    forma = podmiot.get("formaPrawna")
    ids = podmiot.get("identyfikatory", {}) or {}
    nip = ids.get("nip")
    regon = ids.get("regon")
    bits = [f"{nazwa} (KRS {numer})"]
    if forma:
        bits.append(f"— {forma}")
    extras = []
    if nip:
        extras.append(f"NIP {nip}")
    if regon:
        extras.append(f"REGON {regon}")
    if extras:
        bits.append("[" + ", ".join(extras) + "]")
    return " ".join(bits)


def _validate_krs(krs: str) -> str:
    if not (isinstance(krs, str) and len(krs) == 10 and krs.isdigit()
            and krs.startswith("0")):
        raise ValueError(
            f"Numer KRS musi być 10-cyfrowym ciągiem zaczynającym się od 0 "
            f"(otrzymano: {krs!r})."
        )
    return krs


def _normalize_rejestr(rejestr: str) -> str:
    if not isinstance(rejestr, str) or len(rejestr) != 1 or rejestr.upper() not in {"P", "S"}:
        raise ValueError(
            "Rejestr musi być pojedynczą literą: P (przedsiębiorców) "
            f"lub S (stowarzyszeń). Otrzymano: {rejestr!r}."
        )
    return rejestr.upper()


def _empty_message(extract_type_pl: str, krs: str, rejestr: str, status: int) -> str:
    if status == 204:
        other = "S" if rejestr == "P" else "P"
        return (
            f"KRS {krs} istnieje, ale rejestr {rejestr} nie zawiera danych "
            f"({extract_type_pl}). Spróbuj rejestru {other}."
        )
    if status == 404:
        return (
            f"Nie znaleziono KRS {krs} w rejestrze {rejestr} ({extract_type_pl})."
        )
    return (
        f"Nie udało się pobrać {extract_type_pl} dla KRS {krs} (rejestr {rejestr}). "
        f"Status: {status}."
    )


async def _run_extract(extract_type: str, extract_type_pl: str,
                       krs: str, rejestr: str,
                       ctx: Optional[Context]) -> ToolResult:
    krs = _validate_krs(krs)
    rejestr = _normalize_rejestr(rejestr)
    if ctx:
        await ctx.info(f"Fetching {extract_type_pl} for KRS {krs} (rejestr {rejestr})")
    status, body = await _fetch_extract(extract_type, rejestr, krs, ctx)
    if status != 200 or not isinstance(body, dict):
        text = _empty_message(extract_type_pl, krs, rejestr, status)
        return ToolResult(
            content=[{"type": "text", "text": text}],
            structured_content={
                "krs": krs,
                "rejestr": rejestr,
                "extract_type": extract_type,
                "found": False,
                "http_status": status,
            },
        )
    headline = _format_headline(body)
    text = f"{extract_type_pl.capitalize()} – {headline}"
    return ToolResult(
        content=[{"type": "text", "text": text}],
        structured_content={
            "krs": krs,
            "rejestr": rejestr,
            "extract_type": extract_type,
            "found": True,
            "headline": headline,
            "extract": body,
        },
    )


@mcp.tool(
    name="get_krs_current_extract",
    description=(
        "Get the current KRS extract ('odpis aktualny') — the live snapshot of a "
        "Polish entity registered in the National Court Register. Returns the "
        "registered name, legal form (formaPrawna), KRS number, NIP, REGON, "
        "registered office address (siedziba), management board, share capital, "
        "and PKD codes (działalność). "
        "KRS numbers are always 10 digits starting with 0 (e.g. '0001236495' for "
        "Patron Development sp. k., '0000635012' for Allegro sp. z o.o.). "
        "rejestr='P' covers commercial entities (companies, partnerships); "
        "rejestr='S' covers associations, foundations, and non-profit organisations. "
        "If you don't know which register to query, try 'P' first — most queries "
        "are for businesses. A 204 response means the KRS exists but is in the "
        "other register; the tool will hint at that."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def get_krs_current_extract(
    krs: str = Field(
        description="KRS number — exactly 10 digits, starts with '0', e.g. '0001236495'.",
        pattern=r"^0\d{9}$",
    ),
    rejestr: str = Field(
        default="P",
        description=(
            "Register: 'P' (przedsiębiorców — companies/partnerships) or "
            "'S' (stowarzyszeń — associations/foundations). Case-insensitive. "
            "Defaults to 'P'."
        ),
    ),
    ctx: Context = None,
) -> ToolResult:
    return await _run_extract("OdpisAktualny", "odpis aktualny", krs, rejestr, ctx)


@mcp.tool(
    name="get_krs_full_extract",
    description=(
        "Get the full KRS extract ('odpis pełny') — includes complete historical "
        "data: every change to the company since registration (former names, past "
        "board members, capital changes, address changes, registration entries). "
        "Use this when you need history; use get_krs_current_extract for the "
        "current state. Same KRS/rejestr semantics as the current-extract tool."
    ),
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def get_krs_full_extract(
    krs: str = Field(
        description="KRS number — exactly 10 digits, starts with '0', e.g. '0000635012'.",
        pattern=r"^0\d{9}$",
    ),
    rejestr: str = Field(
        default="P",
        description=(
            "Register: 'P' (przedsiębiorców) or 'S' (stowarzyszeń). "
            "Case-insensitive. Defaults to 'P'."
        ),
    ),
    ctx: Context = None,
) -> ToolResult:
    return await _run_extract("OdpisPelny", "odpis pełny", krs, rejestr, ctx)


if __name__ == "__main__":
    mcp.run()
