"""
Unit tests for the OVH verifier.

Response fixtures are real replies from OVH's cart-check endpoint, captured
from the order page, with the long promotions lists and the challenge id
stripped.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import ovh_verifier
from ovh_verifier import (
    OVHVerifier,
    checked_domain,
    is_challenge,
    parse_check_response,
)

FIXTURES = Path(__file__).parent / "fixtures"
RESPONSES = json.loads((FIXTURES / "ovh_cart_domain_responses.json").read_text())
CART = "https://order.eu.ovhcloud.com/engine/api/v1/order/cart/00000000-0000-0000-0000-000000000000"


def parse(case: str) -> dict:
    return parse_check_response(RESPONSES[case]["status"], RESPONSES[case]["body"])


class TestCheckedDomain:
    """Recognising the cart-check endpoint among the page's traffic."""

    def test_cart_check(self):
        assert checked_domain(f"{CART}/domain?domain=erne.dev") == "erne.dev"

    def test_normalises_case(self):
        assert checked_domain(f"{CART}/domain?domain=Erne.DEV") == "erne.dev"

    def test_unicode_matches_punycode(self):
        assert checked_domain(f"{CART}/domain?domain=例子.com") == "xn--fsqu00a.com"

    def test_ignores_suggestions_endpoint(self):
        url = "https://order.eu.ovhcloud.com/engine/api/v1/domain/alternativeNames?label=erne.dev&merchant=PL"
        assert checked_domain(url) is None

    def test_ignores_cart_creation(self):
        assert checked_domain(CART) is None

    def test_missing_domain_param(self):
        assert checked_domain(f"{CART}/domain") is None


class TestIsChallenge:
    def test_challenge(self):
        assert is_challenge(400, RESPONSES["challenge"]["body"])

    def test_check_failed_is_not_a_challenge(self):
        assert not is_challenge(400, RESPONSES["check_failed"]["body"])

    def test_success_is_not_a_challenge(self):
        assert not is_challenge(200, RESPONSES["create_default"]["body"])


class TestParseCheckResponse:
    """pricingMode → verdict, for every mode observed on the live site."""

    def test_standard_registration(self):
        result = parse("create_default")
        assert result["verified"] is True
        assert result["available"] is True
        assert result["action"] == "create"
        assert result["price_type"] == "standard"
        assert result["price"] == "34.08 zł"
        assert result["is_aftermarket"] is False

    def test_registry_premium_is_aftermarket(self):
        result = parse("create_premium")
        assert result["available"] is False
        assert result["is_aftermarket"] is True
        assert result["aftermarket_type"] == "Premium"
        assert result["price_type"] == "premium"
        assert result["action"] == "buy_aftermarket"

    @pytest.mark.parametrize("case", ["transfer_aftermarket1", "transfer_aftermarket2"])
    def test_third_party_resale_is_aftermarket(self, case):
        result = parse(case)
        assert result["available"] is False
        assert result["is_aftermarket"] is True
        assert result["aftermarket_type"] == "Sprzedaż przez stronę trzecią"
        assert result["price_type"] == "third_party"

    @pytest.mark.parametrize("case", ["transfer_default", "transfer_premium"])
    def test_registered(self, case):
        result = parse(case)
        assert result["verified"] is True
        assert result["available"] is False
        assert result["action"] == "transfer"
        assert result["is_aftermarket"] is False

    def test_not_orderable_is_not_available(self):
        body = json.loads(json.dumps(RESPONSES["create_default"]["body"]))
        body[0]["orderable"] = False
        result = parse_check_response(200, body)
        assert result["verified"] is True
        assert result["available"] is False

    def test_unverifiable_extension(self):
        result = parse("check_failed")
        assert result["verified"] is False
        assert result["available"] is None
        assert "check failed" in result["error"]

    def test_empty_list(self):
        result = parse_check_response(200, [])
        assert result["verified"] is False
        assert result["error"] == "Unexpected OVH response"


class FakeResponse:
    def __init__(self, domain: str, status: int, payload):
        self.url = f"{CART}/domain?domain={domain}"
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload


class FakeSearchBox:
    """Stands in for the search box; pressing Enter replays canned responses."""

    def __init__(self, verifier: OVHVerifier, replies: dict[str, list[tuple[int, object]]]):
        self.verifier = verifier
        self.replies = replies
        self.searches: list[str] = []
        self.value = ""

    async def fill(self, value: str):
        self.value = value

    async def press(self, key: str):
        assert key == "Enter"
        self.searches.append(self.value)
        for status, payload in self.replies.get(self.value, []):
            await self.verifier._on_response(FakeResponse(self.value, status, payload))


class FakePage:
    def __init__(self, box: FakeSearchBox):
        self.box = box

    def locator(self, selector: str):
        return self

    def get_by_role(self, role: str):
        return self

    @property
    def first(self):
        return self.box


def ready_verifier(replies: dict) -> tuple[OVHVerifier, FakeSearchBox]:
    verifier = OVHVerifier()
    box = FakeSearchBox(verifier, replies)
    verifier._page = FakePage(box)
    verifier._initialized = True
    return verifier, box


class TestVerifyDomain:
    """Search → wait for the domain's check → parse, without a browser."""

    @pytest.mark.asyncio
    async def test_waits_out_the_challenge(self):
        challenge, ok = RESPONSES["challenge"], RESPONSES["create_default"]
        verifier, box = ready_verifier({
            "fresh.com": [(400, challenge["body"]), (200, ok["body"])],
        })
        result = await verifier.verify_domain("fresh.com")
        assert result["available"] is True
        assert box.searches == ["fresh.com"]

    @pytest.mark.asyncio
    async def test_suggestion_already_seen_skips_search(self):
        verifier, box = ready_verifier({})
        registered = RESPONSES["transfer_default"]
        await verifier._on_response(FakeResponse("erne.dev", 200, registered["body"]))
        result = await verifier.verify_domain("erne.dev")
        assert result["action"] == "transfer"
        assert box.searches == []

    @pytest.mark.asyncio
    async def test_redirects_are_ignored(self):
        verifier, _ = ready_verifier({})
        await verifier._on_response(FakeResponse("erne.dev", 307, None))
        assert verifier._checks == {}

    @pytest.mark.asyncio
    async def test_times_out_when_page_never_checks(self, monkeypatch):
        monkeypatch.setattr(ovh_verifier, "CHECK_TIMEOUT_SECONDS", 0.05)
        verifier, _ = ready_verifier({})
        result = await verifier.verify_domain("silent.com")
        assert result["verified"] is False
        assert "timed out" in result["error"]
        assert verifier._waiters == {}

    @pytest.mark.asyncio
    async def test_uninitialised(self):
        verifier = OVHVerifier()
        verifier._error = "boom"
        result = await verifier.verify_domain("example.com")
        assert result == {
            "available": None,
            "action": None,
            "price": None,
            "price_type": None,
            "is_aftermarket": False,
            "aftermarket_type": None,
            "verified": False,
            "error": "boom",
        }
