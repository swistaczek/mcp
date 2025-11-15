"""
FastMCP Domain Checker Server

Checks domain registration status via WHOIS queries with DNS fallback.
Optional OVH browser verification for domains reported as available.
Supports batch checking up to 50 domains with progress reporting.
"""

import asyncio
import socket
from datetime import datetime, timezone
from typing import Any

from fastmcp import FastMCP, Context
from fastmcp.tools.tool import ToolResult
from pydantic import Field

# Optional OVH verification support
try:
    from ovh_verifier import OVHVerifier, PLAYWRIGHT_AVAILABLE
    OVH_VERIFIER_AVAILABLE = True
except ImportError:
    OVH_VERIFIER_AVAILABLE = False
    PLAYWRIGHT_AVAILABLE = False


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


async def verify_with_ovh(
    available_domains: list[str], ctx: Context
) -> dict[str, dict[str, Any]]:
    """
    Verify domains using OVH browser automation.

    Returns dict of domain -> verification result with keys:
        - ovh_available: bool | None (True only for standard registration)
        - ovh_verified: bool
        - ovh_price: str | None
        - ovh_price_type: 'standard' | 'premium' | 'third_party' | None
        - ovh_is_aftermarket: bool
        - ovh_aftermarket_type: str | None
        - ovh_error: str | None
    """
    if not OVH_VERIFIER_AVAILABLE:
        await ctx.warning("OVH verifier module not available")
        return {
            domain: {
                "ovh_available": None,
                "ovh_verified": False,
                "ovh_price": None,
                "ovh_price_type": None,
                "ovh_is_aftermarket": False,
                "ovh_aftermarket_type": None,
                "ovh_error": "OVH verifier module not installed",
            }
            for domain in available_domains
        }

    if not PLAYWRIGHT_AVAILABLE:
        await ctx.warning("Playwright not installed for OVH verification")
        return {
            domain: {
                "ovh_available": None,
                "ovh_verified": False,
                "ovh_price": None,
                "ovh_price_type": None,
                "ovh_is_aftermarket": False,
                "ovh_aftermarket_type": None,
                "ovh_error": "Playwright not installed",
            }
            for domain in available_domains
        }

    await ctx.info(f"Starting OVH verification for {len(available_domains)} domain(s)")

    async with OVHVerifier() as verifier:
        if not verifier._initialized:
            error_msg = verifier._error or "Failed to initialize OVH verifier"
            await ctx.error(f"OVH initialization failed: {error_msg}")
            return {
                domain: {
                    "ovh_available": None,
                    "ovh_verified": False,
                    "ovh_price": None,
                    "ovh_price_type": None,
                    "ovh_is_aftermarket": False,
                    "ovh_aftermarket_type": None,
                    "ovh_error": error_msg,
                }
                for domain in available_domains
            }

        results = {}
        for i, domain in enumerate(available_domains):
            await ctx.debug(f"OVH verifying {domain} ({i+1}/{len(available_domains)})")
            ovh_result = await verifier.verify_domain(domain)

            results[domain] = {
                "ovh_available": ovh_result.get("available"),
                "ovh_verified": ovh_result.get("verified", False),
                "ovh_price": ovh_result.get("price"),
                "ovh_price_type": ovh_result.get("price_type"),
                "ovh_is_aftermarket": ovh_result.get("is_aftermarket", False),
                "ovh_aftermarket_type": ovh_result.get("aftermarket_type"),
                "ovh_error": ovh_result.get("error"),
            }

            if ovh_result.get("verified"):
                if ovh_result.get("is_aftermarket"):
                    aftermarket_type = ovh_result.get("aftermarket_type", "aftermarket")
                    price = ovh_result.get("price", "N/A")
                    await ctx.warning(f"{domain}: AFTERMARKET domain ({aftermarket_type}) - {price}")
                elif ovh_result.get("available"):
                    await ctx.info(f"{domain}: OVH CONFIRMED available (standard registration)")
                else:
                    await ctx.warning(f"{domain}: OVH says REGISTERED (false positive!)")
            else:
                await ctx.warning(f"{domain}: OVH verification failed - {ovh_result.get('error')}")

            # Small delay between verifications
            await asyncio.sleep(0.3)

        return results


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
    verify_available: bool = Field(
        default=False,
        description="Verify 'available' results using OVH browser automation (slower but catches false positives)",
    ),
    ctx: Context = None,
) -> ToolResult:
    """
    Check if multiple domain names are registered.

    Queries WHOIS servers for each domain, with DNS fallback if WHOIS fails.
    Optional OVH verification for domains reported as available to catch false positives.
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

    # OVH verification for available domains (if requested)
    ovh_verification = {}
    false_positives = []
    aftermarket_domains = []
    confirmed_available = []
    ovh_failed = []

    if verify_available and available_domains:
        await ctx.info(f"OVH verification requested for {len(available_domains)} available domain(s)")
        ovh_verification = await verify_with_ovh(available_domains, ctx)

        # Process OVH results to categorize domains
        for domain in available_domains:
            ovh_result = ovh_verification.get(domain, {})
            ovh_available = ovh_result.get("ovh_available")
            ovh_verified = ovh_result.get("ovh_verified", False)
            ovh_is_aftermarket = ovh_result.get("ovh_is_aftermarket", False)

            # Update detailed results with OVH info
            detailed_results[domain]["ovh_verification"] = ovh_result

            if ovh_verified:
                if ovh_is_aftermarket:
                    # AFTERMARKET: Domain is on secondary market (Premium or third-party)
                    aftermarket_domains.append(domain)
                    detailed_results[domain]["available"] = False
                    aftermarket_type = ovh_result.get("ovh_aftermarket_type", "aftermarket")
                    price = ovh_result.get("ovh_price", "N/A")
                    detailed_results[domain]["reason"] = f"AFTERMARKET: Domain on secondary market ({aftermarket_type}) - {price}"
                    detailed_results[domain]["aftermarket"] = True
                    detailed_results[domain]["aftermarket_type"] = aftermarket_type
                    detailed_results[domain]["aftermarket_price"] = price
                elif ovh_available is False:
                    # FALSE POSITIVE: WHOIS/DNS says available, OVH says registered
                    false_positives.append(domain)
                    detailed_results[domain]["available"] = False
                    detailed_results[domain]["reason"] = "FALSE POSITIVE: OVH verification shows domain is registered"
                    detailed_results[domain]["false_positive"] = True
                elif ovh_available is True:
                    confirmed_available.append(domain)
                    price = ovh_result.get("ovh_price", "N/A")
                    detailed_results[domain]["reason"] = f"OVH CONFIRMED available ({price})"
                    detailed_results[domain]["ovh_confirmed"] = True
                    detailed_results[domain]["standard_price"] = price
            else:
                ovh_failed.append(domain)
                detailed_results[domain]["reason"] += f" (OVH verification failed: {ovh_result.get('ovh_error', 'unknown')})"

        # Update available_domains list to exclude false positives and aftermarket
        available_domains = [d for d in available_domains if d not in false_positives and d not in aftermarket_domains]

    ovh_duration = (datetime.now(timezone.utc) - start_time).total_seconds() - duration

    structured_response = {
        "results": detailed_results,
        "available_domains": available_domains,
        "aftermarket_domains": aftermarket_domains,
        "registered_domains": registered_domains,
        "failed_domains": failed_domains,
        "summary": {
            "total": total,
            "duration_seconds": round(duration, 2),
            "checked_at": start_time.isoformat(),
        },
    }

    # Add OVH verification summary if used
    if verify_available:
        structured_response["ovh_verification"] = {
            "enabled": True,
            "confirmed_available": confirmed_available,
            "aftermarket": aftermarket_domains,
            "false_positives": false_positives,
            "verification_failed": ovh_failed,
            "duration_seconds": round(ovh_duration, 2),
        }
    else:
        structured_response["ovh_verification"] = {"enabled": False}

    # Human-readable summary with domain names
    summary_parts = [f"✓ Checked {total} domain(s) in {duration:.2f}s\n"]

    if verify_available and (available_domains or aftermarket_domains):
        summary_parts.append(f"OVH verification completed in {ovh_duration:.2f}s\n")

    if available_domains:
        if verify_available and confirmed_available:
            summary_parts.append(f"AVAILABLE - OVH CONFIRMED ({len(confirmed_available)}):")
            for domain in confirmed_available:
                ovh_price = ovh_verification.get(domain, {}).get("ovh_price", "N/A")
                summary_parts.append(f"  ✅ {domain} ({ovh_price})")
        else:
            summary_parts.append(f"AVAILABLE ({len(available_domains)}):")
            for domain in available_domains:
                method = detailed_results[domain]["method"]
                if method == "dns":
                    summary_parts.append(f"  ⚠️  {domain} (DNS-based, may be false positive)")
                else:
                    summary_parts.append(f"  • {domain}")
        summary_parts.append("")

    if aftermarket_domains:
        summary_parts.append(f"💰 AFTERMARKET - Secondary Market ({len(aftermarket_domains)}):")
        for domain in aftermarket_domains:
            ovh_result = ovh_verification.get(domain, {})
            aftermarket_type = ovh_result.get("ovh_aftermarket_type", "aftermarket")
            price = ovh_result.get("ovh_price", "N/A")
            summary_parts.append(f"  💰 {domain} ({aftermarket_type}) - {price}")
        summary_parts.append("")

    if false_positives:
        summary_parts.append(f"❌ FALSE POSITIVES ({len(false_positives)}):")
        for domain in false_positives:
            summary_parts.append(f"  ❌ {domain} (WHOIS/DNS says available, OVH says registered)")
        summary_parts.append("")

    if ovh_failed:
        summary_parts.append(f"⚠️  OVH VERIFICATION FAILED ({len(ovh_failed)}):")
        for domain in ovh_failed:
            error = ovh_verification.get(domain, {}).get("ovh_error", "unknown")
            summary_parts.append(f"  ⚠️  {domain} ({error})")
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
