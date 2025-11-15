"""
FastMCP Domain Checker Server

Checks domain registration status via WHOIS queries with DNS fallback.
Supports batch checking up to 50 domains with progress reporting.
"""

import asyncio
import socket
from datetime import datetime, timezone
from typing import Any

from fastmcp import FastMCP, Context
from fastmcp.tools.tool import ToolResult
from pydantic import Field


mcp = FastMCP("Domain Checker")


WHOIS_SERVERS = {
    # Generic TLDs
    "com": "whois.verisign-grs.com",
    "net": "whois.verisign-grs.com",
    "org": "whois.pir.org",
    "info": "whois.nic.info",
    "biz": "whois.nic.biz",
    "mobi": "whois.nic.mobi",
    "edu": "whois.educause.edu",
    "gov": "whois.dotgov.gov",
    "mil": "whois.nic.mil",
    "int": "whois.iana.org",
    # Country/Region TLDs
    "cn": "whois.cnnic.cn",
    "hk": "whois.hkirc.hk",
    "tw": "whois.twnic.net.tw",
    "pl": "whois.dns.pl",
    # Popular New TLDs
    "io": "whois.nic.io",
    "ai": "whois.nic.ai",
    "me": "whois.nic.me",
    "cc": "whois.nic.cc",
    "tv": "whois.nic.tv",
    "co": "whois.nic.co",
    "xyz": "whois.nic.xyz",
    "top": "whois.nic.top",
    "vip": "whois.nic.vip",
    "club": "whois.nic.club",
    "shop": "whois.nic.shop",
    "site": "whois.nic.site",
    "wang": "whois.nic.wang",
    "xin": "whois.nic.xin",
    "app": "whois.nic.google",
    "dev": "whois.nic.google",
    "cloud": "whois.nic.cloud",
    "online": "whois.nic.online",
    "store": "whois.nic.store",
    # Chinese Regional TLDs
    "com.cn": "whois.cnnic.cn",
    "net.cn": "whois.cnnic.cn",
    "org.cn": "whois.cnnic.cn",
    "gov.cn": "whois.cnnic.cn",
    # Chinese IDN TLDs (Unicode)
    "中国": "whois.cnnic.cn",
    "公司": "whois.cnnic.cn",
    "网络": "whois.cnnic.cn",
    "商城": "whois.cnnic.cn",
    "网店": "whois.cnnic.cn",
    "中文网": "whois.cnnic.cn",
}


NOT_FOUND_PATTERNS = {
    "default": [
        "Domain not found",
        "No match for",
        "NOT FOUND",
        "No Data Found",
        "No entries found",
    ],
    "cn": ["no matching record", "No matching record"],
    "com": ["No match for", "NOT FOUND", "No Data Found"],
    "net": ["No match for", "NOT FOUND", "No Data Found"],
    "pl": ["No information available about domain name", "in the Registry NASK database"],
}


def contains_chinese_characters(text: str) -> bool:
    """Check if string contains Chinese/Han characters."""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":  # Unicode Han script range
            return True
    return False


def extract_tld(domain: str) -> str:
    """Extract TLD from domain name. Supports compound TLDs (.com.cn) and Chinese/IDN TLDs."""
    parts = domain.lower().split(".")

    # Check Chinese/IDN TLDs (only last part, as TLDs are at the end)
    if contains_chinese_characters(parts[-1]):
        if parts[-1] in WHOIS_SERVERS:
            return parts[-1]
        return ""

    # Check compound TLDs (e.g., .com.cn)
    if len(parts) >= 3:
        compound_tld = f"{parts[-2]}.{parts[-1]}"
        if compound_tld in WHOIS_SERVERS:
            return compound_tld

    # Check single-level TLD
    if parts[-1] in WHOIS_SERVERS:
        return parts[-1]

    return ""


def get_not_found_patterns(tld: str) -> list[str]:
    """Get WHOIS 'not found' patterns for specific TLD."""
    return NOT_FOUND_PATTERNS.get(tld, NOT_FOUND_PATTERNS["default"])


async def query_whois(domain: str, tld: str) -> tuple[str | None, str | None]:
    """Query WHOIS server. Returns (response, error)."""
    whois_server = WHOIS_SERVERS.get(tld, "whois.iana.org")

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(whois_server, 43), timeout=10.0
        )

        writer.write(f"{domain}\r\n".encode())
        await writer.drain()

        response = await asyncio.wait_for(reader.read(4096), timeout=10.0)

        writer.close()
        await writer.wait_closed()

        return response.decode("utf-8", errors="ignore"), None

    except asyncio.TimeoutError:
        return None, "WHOIS query timeout"
    except Exception as e:
        return None, f"WHOIS query failed: {str(e)}"


async def check_dns_record(domain: str) -> tuple[bool, str | None]:
    """Check DNS records for domain. Returns (has_records, error)."""
    try:
        await asyncio.to_thread(socket.gethostbyname, domain)
        return True, None
    except socket.gaierror:
        return False, None
    except Exception as e:
        return False, f"DNS lookup failed: {str(e)}"


async def check_single_domain(domain: str, ctx: Context) -> dict[str, Any]:
    """Check if domain is registered. Returns status dict."""
    tld = extract_tld(domain)
    if not tld or tld not in WHOIS_SERVERS:
        await ctx.warning(f"{domain}: Unsupported TLD '{tld}'")
        return {
            "registered": False,
            "method": "unsupported",
            "reason": f"Unsupported top-level domain: {tld}",
        }

    await ctx.debug(f"{domain}: Querying WHOIS server {WHOIS_SERVERS[tld]}")
    whois_response, whois_error = await query_whois(domain, tld)

    if whois_error:
        await ctx.warning(f"{domain}: WHOIS failed ({whois_error}), trying DNS")
        dns_has_records, dns_error = await check_dns_record(domain)

        if dns_error is None:
            result = {
                "registered": dns_has_records,
                "method": "dns",
                "reason": "DNS resolution successful"
                if dns_has_records
                else "No DNS records found",
            }
            await ctx.info(
                f"{domain}: {'Registered' if dns_has_records else 'Available'} (DNS)"
            )
            return result
        else:
            await ctx.error(f"{domain}: Both WHOIS and DNS failed")
            return {
                "registered": False,
                "method": "failed",
                "reason": "Both WHOIS and DNS queries failed",
            }

    not_found_patterns = get_not_found_patterns(tld)
    for pattern in not_found_patterns:
        if pattern in whois_response:
            await ctx.info(f"{domain}: Available (WHOIS)")
            return {
                "registered": False,
                "method": "whois",
                "reason": "Domain available according to WHOIS",
            }

    await ctx.info(f"{domain}: Registered (WHOIS)")
    return {
        "registered": True,
        "method": "whois",
        "reason": "Domain registered according to WHOIS",
    }


@mcp.tool(
    name="check_domains",
    description="Check domain registration status via WHOIS/DNS. Max 50 domains.",
    annotations={"readOnlyHint": True, "openWorldHint": True},
)
async def check_domains(
    domains: list[str] = Field(
        min_length=1,
        max_length=50,
        description="List of domain names to check (1-50)",
    ),
    ctx: Context = None,
) -> ToolResult:
    """
    Check if multiple domain names are registered.

    Queries WHOIS servers for each domain, with DNS fallback if WHOIS fails.
    Reports progress in real-time for batch operations.
    """
    total = len(domains)
    start_time = datetime.now(timezone.utc)

    await ctx.info(f"Starting domain check for {total} domain(s)")
    await ctx.report_progress(0, total, "Starting domain check")

    results = []
    stats = {"registered": 0, "available": 0, "failed": 0}
    tlds_queried = set()

    for i, domain in enumerate(domains):
        await ctx.report_progress(i, total, f"Checking: {domain}")
        status = await check_single_domain(domain, ctx)
        results.append((domain, status))

        if status["method"] == "failed" or status["method"] == "unsupported":
            stats["failed"] += 1
        elif status["registered"]:
            stats["registered"] += 1
        else:
            stats["available"] += 1

        tld = extract_tld(domain)
        if tld:
            tlds_queried.add(tld)

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    await ctx.report_progress(total, total, f"Complete: checked {total} domain(s)")

    detailed_results = {
        domain: {
            "registered": status["registered"],
            "available": not status["registered"]
            and status["method"] not in ["failed", "unsupported"],
            "method": status["method"],
            "reason": status["reason"],
        }
        for domain, status in results
    }

    # Build convenience lists for quick access
    available_domains = [
        domain
        for domain, status in results
        if not status["registered"] and status["method"] not in ["failed", "unsupported"]
    ]
    registered_domains = [
        domain for domain, status in results if status["registered"]
    ]
    failed_domains = [
        domain
        for domain, status in results
        if status["method"] in ["failed", "unsupported"]
    ]

    structured_response = {
        "results": detailed_results,
        "available_domains": available_domains,
        "registered_domains": registered_domains,
        "failed_domains": failed_domains,
        "summary": {
            "total": total,
            "duration_seconds": round(duration, 2),
            "checked_at": start_time.isoformat(),
        },
    }

    # Human-readable summary with domain names
    summary_parts = [f"✓ Checked {total} domain(s) in {duration:.2f}s\n"]

    if available_domains:
        summary_parts.append(f"AVAILABLE ({len(available_domains)}):")
        for domain in available_domains:
            summary_parts.append(f"  • {domain}")
        summary_parts.append("")

    if registered_domains:
        summary_parts.append(f"REGISTERED ({len(registered_domains)}):")
        for domain in registered_domains:
            summary_parts.append(f"  • {domain}")
        summary_parts.append("")

    if failed_domains:
        summary_parts.append(f"FAILED ({len(failed_domains)}):")
        for domain in failed_domains:
            reason = detailed_results[domain]["reason"]
            summary_parts.append(f"  • {domain} ({reason})")
        summary_parts.append("")

    summary = "\n".join(summary_parts).rstrip()

    return ToolResult(
        content=[{"type": "text", "text": summary}],
        structured_content=structured_response,
    )
