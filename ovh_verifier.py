"""
OVH Domain Verification Module

Browser-based verification of domain availability using OVH's order page.
Used as secondary verification layer when WHOIS reports domain as available.

The order page checks each domain through a JSON endpoint,
``/engine/api/v1/order/cart/{cart_id}/domain?domain=…``, guarded by a bot
challenge that only the page's own JavaScript can solve: the first call
answers ``400 Client::BadRequest::ChallengeRequired``, the page solves it and
retries. So the browser still types into the search box — that is what gets
the challenge solved — but the verdict is read from the endpoint's JSON, not
from the rendered results table, whose labels and markup OVH redesigns
without notice.
"""

import asyncio
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext, Response
    from playwright.async_api import Error as PlaywrightError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


ORDER_URL = "https://order.eu.ovhcloud.com/pl/order/webcloud/?#/webCloud/domain/select?selection=~()"
CHECK_PATH = re.compile(r"/engine/api/v1/order/cart/[^/]+/domain$")
CHECK_TIMEOUT_SECONDS = 20

# Aftermarket labels as OVH's UI shows them, kept as the public vocabulary
PREMIUM = "Premium"
THIRD_PARTY = "Sprzedaż przez stronę trzecią"


def _empty_result() -> dict[str, Any]:
    return {
        "available": None,
        "action": None,
        "price": None,
        "price_type": None,
        "is_aftermarket": False,
        "aftermarket_type": None,
        "verified": False,
        "error": None,
    }


def _normalize(domain: str) -> str:
    """Lowercase ASCII form, so a Unicode domain matches its punycode query."""
    domain = domain.strip().lower().rstrip(".")
    try:
        return domain.encode("idna").decode("ascii")
    except UnicodeError:
        return domain


def checked_domain(url: str) -> str | None:
    """Domain a cart-check request asks about, or None for any other URL."""
    parsed = urlparse(url)
    if not CHECK_PATH.search(parsed.path):
        return None
    values = parse_qs(parsed.query).get("domain")
    return _normalize(values[0]) if values else None


def is_challenge(status: int, payload: Any) -> bool:
    """True for the bot-challenge reply the page answers and then retries."""
    return (
        status == 400
        and isinstance(payload, dict)
        and payload.get("class", "").endswith("ChallengeRequired")
    )


def parse_check_response(status: int, payload: Any) -> dict[str, Any]:
    """
    Map one cart-check response onto the verifier result shape.

    ``pricingMode`` carries the verdict:
        create-default            standard registration
        create-premium            registry premium ("Premium")
        transfer-aftermarket<N>   resale by a third party
        transfer-default/-premium registered, can only be transferred
    """
    result = _empty_result()

    if status != 200:
        message = payload.get("message") if isinstance(payload, dict) else None
        result["error"] = f"OVH cannot check this domain: {message or f'HTTP {status}'}"
        return result

    offer = payload[0] if isinstance(payload, list) and payload else None
    if not isinstance(offer, dict):
        result["error"] = "Unexpected OVH response"
        return result

    mode = offer.get("pricingMode") or ""
    action = offer.get("action")
    prices = {p.get("label"): p.get("price") or {} for p in offer.get("prices", [])}

    result["verified"] = True
    result["price"] = prices.get("PRICE", {}).get("text")

    if mode == "create-premium" or "aftermarket" in mode:
        # Aftermarket domains are NOT available for standard registration
        premium = mode == "create-premium"
        result.update({
            "available": False,
            "action": "buy_aftermarket",
            "price_type": "premium" if premium else "third_party",
            "is_aftermarket": True,
            "aftermarket_type": PREMIUM if premium else THIRD_PARTY,
        })
    elif action == "create":
        orderable = offer.get("orderable") is True
        result.update({
            "available": orderable,
            "action": "create" if orderable else None,
            "price_type": "standard" if orderable else None,
        })
    else:
        result.update({"available": False, "action": action})

    return result


class OVHVerifier:
    """Browser-based domain availability verification using OVH."""

    def __init__(self):
        self._playwright = None
        self._browser: "Browser | None" = None
        self._context: "BrowserContext | None" = None
        self._page: "Page | None" = None
        self._initialized = False
        self._error: str | None = None
        # Every check the page makes lands here, including the ~30 suggestions
        # each search fans out to, so later domains are often already answered
        self._checks: dict[str, tuple[int, Any]] = {}
        self._waiters: dict[str, asyncio.Future] = {}

    async def initialize(self) -> bool:
        """Initialize browser session. Returns True if successful."""
        if not PLAYWRIGHT_AVAILABLE:
            self._error = "Playwright not installed. Run: uv sync && uv run playwright install chromium"
            return False

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._launch()
            self._context = await self._browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )
            self._page = await self._context.new_page()
            self._page.on("response", self._on_response)

            await self._page.goto(ORDER_URL, timeout=30000)
            await self._page.wait_for_load_state("networkidle", timeout=30000)

            # Dismiss the consent dialog ("Kontynuuj bez akceptacji") if shown
            try:
                await self._page.locator("#header_tc_privacy_button").click(timeout=3000)
            except Exception:
                pass

            self._initialized = True
            return True

        except Exception as e:
            self._initialized = False
            self._error = str(e)
            return False

    async def _launch(self) -> "Browser":
        """Bundled Chromium, falling back to an installed Google Chrome."""
        try:
            return await self._playwright.chromium.launch(headless=True)
        except PlaywrightError as e:
            if "Executable doesn't exist" not in str(e):
                raise
        try:
            return await self._playwright.chromium.launch(headless=True, channel="chrome")
        except PlaywrightError as e:
            raise RuntimeError(
                "No browser for OVH verification. Run: uv run playwright install chromium"
            ) from e

    async def _on_response(self, response: "Response"):
        domain = checked_domain(response.url)
        if domain is None or 300 <= response.status < 400:
            return
        try:
            payload = await response.json()
        except Exception:
            payload = None
        if is_challenge(response.status, payload):
            return  # the page solves it and asks again

        self._checks[domain] = (response.status, payload)
        waiter = self._waiters.pop(domain, None)
        if waiter and not waiter.done():
            waiter.set_result(self._checks[domain])

    async def close(self):
        """Close browser session."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._initialized = False
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def verify_domain(self, domain: str) -> dict[str, Any]:
        """
        Verify domain availability using OVH web interface.

        Returns dict with:
            - available: bool | None (True only for standard registration, False for aftermarket/registered)
            - action: 'create' | 'transfer' | 'buy_aftermarket' | None
            - price: str | None (first-year price)
            - price_type: 'standard' | 'premium' | 'third_party' | None
            - is_aftermarket: bool (True if domain is on secondary market)
            - aftermarket_type: str | None ('Premium' or 'Sprzedaż przez stronę trzecią')
            - verified: bool (True if OVH verification succeeded)
            - error: str | None
        """
        if not self._initialized or not self._page:
            result = _empty_result()
            result["error"] = self._error or "OVH verifier not initialized"
            return result

        key = _normalize(domain)
        try:
            if key not in self._checks:
                waiter = asyncio.get_running_loop().create_future()
                self._waiters[key] = waiter
                search_box = self._page.locator("main").get_by_role("textbox").first
                await search_box.fill(domain)
                await search_box.press("Enter")
                await asyncio.wait_for(waiter, CHECK_TIMEOUT_SECONDS)

            return parse_check_response(*self._checks[key])

        except asyncio.TimeoutError:
            result = _empty_result()
            result["error"] = f"OVH verification timed out after {CHECK_TIMEOUT_SECONDS}s"
            return result
        except Exception as e:
            result = _empty_result()
            result["error"] = f"OVH verification error: {str(e)}"
            return result
        finally:
            self._waiters.pop(key, None)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


async def verify_domains_with_ovh(domains: list[str]) -> tuple[dict[str, dict[str, Any]], str | None]:
    """
    Verify multiple domains using OVH.

    Returns:
        (results_dict, initialization_error)
        - results_dict: {domain: verification_result}
        - initialization_error: None if successful, error string if failed to start
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {}, "Playwright not installed"

    async with OVHVerifier() as verifier:
        if not verifier._initialized:
            return {}, verifier._error

        results = {}
        for domain in domains:
            results[domain] = await verifier.verify_domain(domain)
            await asyncio.sleep(0.3)

        return results, None
