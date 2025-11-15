"""
OVH Domain Verification Module

Browser-based verification of domain availability using OVH's web interface.
Used as secondary verification layer when WHOIS reports domain as available.
"""

import asyncio
from typing import Any

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class OVHVerifier:
    """Browser-based domain availability verification using OVH."""

    def __init__(self):
        self._browser: "Browser | None" = None
        self._context: "BrowserContext | None" = None
        self._page: "Page | None" = None
        self._initialized = False
        self._error: str | None = None

    async def initialize(self) -> bool:
        """Initialize browser session. Returns True if successful."""
        if not PLAYWRIGHT_AVAILABLE:
            self._error = "Playwright not installed. Run: pip install playwright && python -m playwright install chromium"
            return False

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._context = await self._browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )
            self._page = await self._context.new_page()

            # Navigate to OVH domain search
            await self._page.goto(
                "https://order.eu.ovhcloud.com/pl/order/webcloud/?#/webCloud/domain/select?selection=~()",
                timeout=30000
            )
            await self._page.wait_for_load_state("networkidle", timeout=30000)

            # Accept cookies if present
            try:
                accept_btn = self._page.get_by_role("button", name="Zaakceptuj")
                if await accept_btn.is_visible(timeout=3000):
                    await accept_btn.click()
                    await self._page.wait_for_timeout(1000)
            except Exception:
                pass

            self._initialized = True
            return True

        except Exception as e:
            self._initialized = False
            self._error = str(e)
            return False

    async def close(self):
        """Close browser session."""
        if self._browser:
            await self._browser.close()
        if hasattr(self, '_playwright') and self._playwright:
            await self._playwright.stop()
        self._initialized = False
        self._browser = None
        self._context = None
        self._page = None

    async def verify_domain(self, domain: str) -> dict[str, Any]:
        """
        Verify domain availability using OVH web interface.

        Returns dict with:
            - available: bool | None
            - action: 'create' | 'transfer' | None
            - price: str | None
            - verified: bool (True if OVH verification succeeded)
            - error: str | None
        """
        if not self._initialized or not self._page:
            return {
                "available": None,
                "action": None,
                "price": None,
                "verified": False,
                "error": self._error or "OVH verifier not initialized"
            }

        try:
            # Clear and fill the search box
            search_box = self._page.get_by_role("textbox", name="mojadomena.pl")
            await search_box.clear()
            await search_box.fill(domain)

            # Click search button
            await self._page.get_by_role("button", name="Szukaj").click()

            # Wait for results to load
            await self._page.wait_for_timeout(2500)

            # Parse results from the page
            result = await self._page.evaluate(f"""
                () => {{
                    const domain = '{domain}';
                    const results = {{
                        available: null,
                        action: null,
                        price: null,
                        verified: false,
                        error: null
                    }};

                    const tables = document.querySelectorAll('table');

                    for (const table of tables) {{
                        const rows = table.querySelectorAll('tbody tr');
                        for (const row of rows) {{
                            const domainCell = row.querySelector('td:first-child');
                            if (!domainCell) continue;

                            const domainText = domainCell.textContent || '';

                            // Match exact domain
                            const domainMatch = domainText.match(/([a-zA-Z0-9-]+\\.[a-zA-Z0-9.]+)/);
                            if (domainMatch && domainMatch[1] === domain) {{
                                results.verified = true;

                                if (domainText.includes('Dostępny')) {{
                                    results.available = true;
                                    results.action = 'create';
                                }} else if (domainText.includes('zarezerwowana') || domainText.includes('Już zarezerwowana')) {{
                                    results.available = false;
                                    results.action = 'transfer';
                                }}

                                const priceCell = row.querySelector('td:nth-child(2) strong, td:nth-child(3) strong');
                                if (priceCell) {{
                                    results.price = priceCell.textContent.trim();
                                }}

                                return results;
                            }}
                        }}
                    }}

                    results.error = 'Domain not found in OVH results';
                    return results;
                }}
            """)

            return result

        except Exception as e:
            return {
                "available": None,
                "action": None,
                "price": None,
                "verified": False,
                "error": f"OVH verification error: {str(e)}"
            }

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
