"""Unit tests for google_ads_planner.py — no live network, no Google Ads creds.

Everything is mocked: `_post` is monkeypatched for the tool-level tests, and the
few tests that exercise the HTTP layer itself (headers, the Google Ads error
envelope, the 1-QPS throttle) use aioresponses with a fake access token.

Written against the binding contract, not against the implementation.

Live integration tests are marked with @pytest.mark.integration and skipped by
default (they hit googleads.googleapis.com and need real credentials).
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import importlib
import inspect
import json
import re
import time
from pathlib import Path

import pytest
from aioresponses import aioresponses
from fastmcp.exceptions import ToolError
from pydantic import ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

import google_ads_planner


FIXTURES = Path(__file__).parent / "fixtures"

TOOL_NAMES = [
    "google_ads_keyword_metrics",
    "google_ads_keyword_ideas",
    "google_ads_forecast_budget",
    "google_ads_budget_curve",
    "google_ads_seasonality",
]

# A syntactically valid service-account JSON. The private key is nonsense on
# purpose — nothing in the test suite ever mints a real token.
SA_KEY = {
    "type": "service_account",
    "project_id": "marmot-ads-test",
    "private_key_id": "0123456789abcdef",
    "private_key": "-----BEGIN PRIVATE KEY-----\nbm90LWEta2V5\n-----END PRIVATE KEY-----\n",
    "client_email": "ads-test@marmot-ads-test.iam.gserviceaccount.com",
    "client_id": "112233445566778899",
    "token_uri": "https://oauth2.googleapis.com/token",
}
SA_KEY_B64 = base64.b64encode(json.dumps(SA_KEY).encode()).decode()

ENV = {
    "GOOGLE_ADS_DEVELOPER_TOKEN": "test-developer-token",
    "GOOGLE_ADS_SERVICE_ACCOUNT_KEY_B64": SA_KEY_B64,
    "GOOGLE_ADS_LOGIN_CUSTOMER_ID": "1234567890",
    "GOOGLE_ADS_CUSTOMER_ID": "9876543210",
    "GOOGLE_ADS_API_VERSION": "v25",
}

FAKE_TOKEN = "ya29.fake-access-token"


def _fixture(name: str) -> dict:
    with open(FIXTURES / f"google_ads_planner_{name}.json") as f:
        return json.load(f)


IDEAS_RESPONSE = _fixture("keyword_ideas")
HISTORICAL_RESPONSE = _fixture("historical_metrics")
FORECAST_RESPONSE = _fixture("forecast")
ERROR_403 = _fixture("error_403")
SEASONALITY_RESPONSE = _fixture("seasonality")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def gads(monkeypatch):
    """Module reloaded with credentials in the env before every test.

    Reloading also resets module-level state (token cache, throttle clock) so
    tests can't bleed into each other.
    """
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    importlib.reload(google_ads_planner)
    return google_ads_planner


class FakePost:
    """Stand-in for `_post` — records calls, replays canned responses."""

    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls: list[dict] = []

    async def __call__(self, *args, **kwargs):
        endpoint = args[0] if args else kwargs.get("endpoint")
        payload = args[1] if len(args) > 1 else kwargs.get("payload")
        self.calls.append({"endpoint": endpoint, "payload": payload})
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[idx]

    @property
    def payload(self) -> dict:
        return self.calls[-1]["payload"]


def _patch_post(monkeypatch, module, responses) -> FakePost:
    fake = FakePost(responses)
    monkeypatch.setattr(module, "_post", fake)
    return fake


def _patch_token(monkeypatch, module, token: str = FAKE_TOKEN) -> str:
    """Patch `_access_token` whether the implementation made it sync or async."""
    current = getattr(module, "_access_token", None)
    if inspect.iscoroutinefunction(current):
        async def _fake(*args, **kwargs):
            return token
    else:
        def _fake(*args, **kwargs):
            return token
    monkeypatch.setattr(module, "_access_token", _fake)
    return token


async def _run(module, tool_name: str, arguments: dict):
    """Invoke a tool through FastMCP so declared defaults/validation apply."""
    tool = getattr(module, tool_name)
    run = getattr(tool, "run", None)
    if run is None:  # FASTMCP_DECORATOR_MODE=function
        return await tool(**arguments)
    return await run(arguments)


async def _call_fn(module, tool_name: str, ctx=None, **kwargs):
    """Call the raw tool function with an explicit ctx.

    Params the caller didn't supply fall back to their declared Field default,
    because calling the undecorated function leaves FieldInfo objects behind.
    """
    tool = getattr(module, tool_name)
    fn = getattr(tool, "fn", tool)
    call: dict = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name == "ctx":
            call["ctx"] = ctx
        elif name in kwargs:
            call[name] = kwargs[name]
        elif isinstance(param.default, FieldInfo):
            default = param.default.get_default(call_default_factory=True)
            call[name] = None if default is PydanticUndefined else default
        else:
            call[name] = param.default
    return await fn(**call)


def _structured(result) -> dict:
    return result.structured_content


def _prose(result) -> str:
    return "".join(block.text for block in result.content if hasattr(block, "text"))


class RecordingContext:
    """Minimal async Context double — records logs and progress."""

    def __init__(self):
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.progress: list[tuple] = []

    async def info(self, message, logger_name=None, extra=None):
        self.infos.append(message)

    async def warning(self, message, logger_name=None, extra=None):
        self.warnings.append(message)

    async def debug(self, message, logger_name=None, extra=None):
        pass

    async def error(self, message, logger_name=None, extra=None):
        pass

    async def report_progress(self, progress, total=None, message=None):
        self.progress.append((progress, total, message))


ANY_URL = re.compile(r".*")


# ---------------------------------------------------------------------------
# Service-account key decoding
# ---------------------------------------------------------------------------

class TestDecodeSaKey:
    def test_round_trip_with_padding(self, gads):
        assert gads._decode_sa_key(SA_KEY_B64) == SA_KEY

    @pytest.mark.parametrize("filler", ["", "a", "bb", "ccc", "dddd"])
    def test_round_trip_without_padding(self, gads, filler):
        """Padding '=' is routinely stripped by shells/CI vars — must be tolerated.

        Varying the payload length walks every len%4 remainder.
        """
        key = dict(SA_KEY, project_id="marmot-ads-" + filler)
        encoded = base64.b64encode(json.dumps(key).encode()).decode().rstrip("=")
        assert encoded.endswith("=") is False
        assert gads._decode_sa_key(encoded) == key

    def test_returns_a_dict_with_the_expected_sa_fields(self, gads):
        decoded = gads._decode_sa_key(SA_KEY_B64.rstrip("="))
        assert decoded["type"] == "service_account"
        assert decoded["client_email"].endswith(".iam.gserviceaccount.com")

    def test_garbage_raises(self, gads):
        """Contract is silent on the type; ValueError or ToolError both qualify."""
        with pytest.raises((ValueError, ToolError)):
            gads._decode_sa_key("this-is-definitely-not-a-service-account-key")


# ---------------------------------------------------------------------------
# Micros conversion — the presence trap lives here
# ---------------------------------------------------------------------------

class TestMicrosToFloat:
    def test_none_stays_none(self, gads):
        assert gads._micros_to_float(None) is None

    def test_absent_dict_key_is_none_not_zero(self, gads):
        metrics = {"avgMonthlySearches": "590", "competition": "LOW"}
        assert gads._micros_to_float(metrics.get("lowTopOfPageBidMicros")) is None
        assert gads._micros_to_float(metrics.get("highTopOfPageBidMicros")) is None

    @pytest.mark.parametrize("raw,expected", [
        ("420000", 0.42),
        ("2140000", 2.14),
        ("1124435", 1.124435),
        ("65339954340", 65339.95434),
        ("0", 0.0),
    ])
    def test_int64_arrives_as_a_string(self, gads, raw, expected):
        assert gads._micros_to_float(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw,expected", [
        (420000, 0.42),
        (2500000, 2.5),
        (0, 0.0),
        (1, 0.000001),
    ])
    def test_int_input(self, gads, raw, expected):
        assert gads._micros_to_float(raw) == pytest.approx(expected)

    def test_zero_is_zero_not_none(self, gads):
        """Absent -> None, but an explicit 0 is a real zero bid."""
        assert gads._micros_to_float("0") == 0.0
        assert gads._micros_to_float("0") is not None

    def test_returns_float(self, gads):
        assert isinstance(gads._micros_to_float("1000000"), float)


# ---------------------------------------------------------------------------
# Resource paths and constants
# ---------------------------------------------------------------------------

class TestResourcePaths:
    def test_geo_path(self, gads):
        assert gads._geo_path(2840) == "geoTargetConstants/2840"

    def test_lang_path(self, gads):
        assert gads._lang_path(1000) == "languageConstants/1000"

    @pytest.mark.parametrize("geo", [2840, 2826, 2124])
    def test_geo_path_for_each_supported_country(self, gads, geo):
        assert gads._geo_path(geo) == f"geoTargetConstants/{geo}"

    def test_constants(self, gads):
        assert gads.GEO_USA == 2840
        assert gads.GEO_UK == 2826
        assert gads.GEO_CANADA == 2124
        assert gads.LANG_ENGLISH == 1000
        assert gads.LANG_SPANISH == 1003
        assert gads.LANG_FRENCH == 1002

    def test_defaults(self, gads):
        assert gads.DEFAULT_GEO_TARGETS == [gads.GEO_USA]
        assert gads.DEFAULT_LANGUAGE == gads.LANG_ENGLISH


# ---------------------------------------------------------------------------
# Opportunity score
# ---------------------------------------------------------------------------

class TestOpportunityScore:
    @pytest.mark.parametrize("volume,index,expected", [
        (301000, 100, 301000 / 101),
        (165000, 91, 165000 / 92),
        (22000, 54, 22000 / 55),
        (14800, 34, 14800 / 35),
        (9900, 33, 9900 / 34),
        (590, 2, 590 / 3),
    ])
    def test_volume_over_index_plus_one(self, gads, volume, index, expected):
        assert gads._opportunity_score(volume, index) == pytest.approx(expected)

    def test_zero_competition_index_does_not_divide_by_zero(self, gads):
        assert gads._opportunity_score(1000, 0) == pytest.approx(1000.0)

    def test_higher_competition_scores_lower_for_equal_volume(self, gads):
        assert gads._opportunity_score(10000, 10) > gads._opportunity_score(10000, 90)

    def test_higher_volume_scores_higher_for_equal_competition(self, gads):
        assert gads._opportunity_score(50000, 50) > gads._opportunity_score(5000, 50)

    def test_low_volume_high_competition_beaten_by_mid_volume_low_competition(self, gads):
        """The whole point of the metric: 14.8k @ idx 34 beats 22k @ idx 54."""
        assert gads._opportunity_score(14800, 34) > gads._opportunity_score(22000, 54)


# ---------------------------------------------------------------------------
# Competition buckets — boundaries are the contract
# ---------------------------------------------------------------------------

class TestCompetitionBucket:
    @pytest.mark.parametrize("index,bucket", [
        (0, "LOW"),
        (33, "LOW"),
        (34, "MEDIUM"),
        (66, "MEDIUM"),
        (67, "HIGH"),
        (100, "HIGH"),
    ])
    def test_boundaries(self, gads, index, bucket):
        assert gads._competition_bucket(index) == bucket

    @pytest.mark.parametrize("index", [1, 15, 32])
    def test_inside_low(self, gads, index):
        assert gads._competition_bucket(index) == "LOW"

    @pytest.mark.parametrize("index", [35, 50, 65])
    def test_inside_medium(self, gads, index):
        assert gads._competition_bucket(index) == "MEDIUM"

    @pytest.mark.parametrize("index", [68, 85, 99])
    def test_inside_high(self, gads, index):
        assert gads._competition_bucket(index) == "HIGH"


# ---------------------------------------------------------------------------
# Module import / tool registration without credentials
# ---------------------------------------------------------------------------

class TestModuleWithoutCredentials:
    def test_import_succeeds_with_no_env_vars_set(self, monkeypatch):
        for key in ENV:
            monkeypatch.delenv(key, raising=False)
        module = importlib.reload(google_ads_planner)
        assert module.mcp is not None
        for name in TOOL_NAMES:
            assert hasattr(module, name), f"{name} missing after credential-free import"

    @pytest.mark.asyncio
    async def test_all_five_tools_registered(self, gads):
        for name in TOOL_NAMES:
            tool = await gads.mcp.get_tool(name)
            assert tool is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", TOOL_NAMES)
    async def test_tools_are_read_only_idempotent_open_world(self, gads, name):
        tool = await gads.mcp.get_tool(name)
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,params", [
        ("google_ads_keyword_metrics",
         {"keywords", "geo_target_ids", "language_id", "network",
          "start_year_month", "end_year_month"}),
        ("google_ads_keyword_ideas",
         {"seed_keywords", "seed_url", "seed_site", "geo_target_ids",
          "language_id", "network", "limit"}),
        ("google_ads_forecast_budget",
         {"keywords", "match_type", "max_cpc_bid", "daily_budget",
          "start_date", "end_date", "geo_target_ids", "language_id"}),
        ("google_ads_budget_curve",
         {"keywords", "match_type", "bids", "start_date", "end_date",
          "geo_target_ids", "language_id"}),
        ("google_ads_seasonality",
         {"keywords", "months_back", "geo_target_ids", "language_id"}),
    ])
    async def test_declared_parameters(self, gads, name, params):
        tool = await gads.mcp.get_tool(name)
        properties = tool.parameters.get("properties", {})
        assert params <= set(properties), f"{name} missing {params - set(properties)}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", TOOL_NAMES)
    async def test_every_parameter_is_described(self, gads, name):
        tool = await gads.mcp.get_tool(name)
        for param, schema in tool.parameters.get("properties", {}).items():
            assert schema.get("description"), f"{name}.{param} has no description"


# ---------------------------------------------------------------------------
# Tool 1 — google_ads_keyword_metrics
# ---------------------------------------------------------------------------

class TestKeywordMetrics:
    @pytest.mark.asyncio
    async def test_maps_a_fully_populated_keyword(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        result = await _run(gads, "google_ads_keyword_metrics",
                            {"keywords": ["running shoes", "mkbhd merch"]})
        row = _structured(result)["keywords"][0]
        assert row["text"] == "running shoes"
        assert row["avg_monthly_searches"] == 301000
        assert row["competition"] == "HIGH"
        assert row["competition_index"] == 100
        assert row["low_top_of_page_bid"] == pytest.approx(0.42)
        assert row["high_top_of_page_bid"] == pytest.approx(2.14)

    @pytest.mark.asyncio
    async def test_int64_strings_are_cast_to_int(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        result = await _run(gads, "google_ads_keyword_metrics",
                            {"keywords": ["running shoes"]})
        row = _structured(result)["keywords"][0]
        assert isinstance(row["avg_monthly_searches"], int)
        assert isinstance(row["competition_index"], int)

    @pytest.mark.asyncio
    async def test_monthly_volumes_shape(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        result = await _run(gads, "google_ads_keyword_metrics",
                            {"keywords": ["running shoes"]})
        volumes = _structured(result)["keywords"][0]["monthly_volumes"]
        assert len(volumes) == 12
        first = volumes[0]
        assert set(first) == {"year", "month", "searches"}
        assert isinstance(first["year"], int)
        assert first["month"] in {
            "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
            "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
        }
        assert isinstance(first["searches"], int)

    @pytest.mark.asyncio
    async def test_close_variants_are_returned(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        result = await _run(gads, "google_ads_keyword_metrics",
                            {"keywords": ["running shoes", "mkbhd merch"]})
        rows = _structured(result)["keywords"]
        assert rows[0]["close_variants"] == ["running shoe", "shoes for running"]
        assert rows[1]["close_variants"] == []

    @pytest.mark.asyncio
    async def test_echoes_request_context(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        result = await _run(gads, "google_ads_keyword_metrics",
                            {"keywords": ["running shoes", "mkbhd merch"],
                             "geo_target_ids": [2826], "language_id": 1002,
                             "network": "GOOGLE_SEARCH_AND_PARTNERS"})
        structured = _structured(result)
        assert structured["requested"] == 2
        assert structured["returned"] == 2
        assert structured["geo_target_ids"] == [2826]
        assert structured["language_id"] == 1002
        assert structured["network"] == "GOOGLE_SEARCH_AND_PARTNERS"

    @pytest.mark.asyncio
    async def test_returned_counts_what_came_back_not_what_was_asked(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        result = await _run(gads, "google_ads_keyword_metrics",
                            {"keywords": ["running shoes", "mkbhd merch",
                                          "a keyword google has no data for"]})
        structured = _structured(result)
        assert structured["requested"] == 3
        assert structured["returned"] == 2

    @pytest.mark.asyncio
    async def test_defaults_to_usa_english_google_search(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        result = await _run(gads, "google_ads_keyword_metrics",
                            {"keywords": ["running shoes"]})
        structured = _structured(result)
        assert structured["geo_target_ids"] == [2840]
        assert structured["language_id"] == 1000
        assert structured["network"] == "GOOGLE_SEARCH"
        wire = json.dumps(fake.payload)
        assert "geoTargetConstants/2840" in wire
        assert "languageConstants/1000" in wire
        assert "GOOGLE_SEARCH" in wire

    @pytest.mark.asyncio
    async def test_keywords_are_sent_on_the_wire(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        await _run(gads, "google_ads_keyword_metrics",
                   {"keywords": ["running shoes", "mkbhd merch"]})
        wire = json.dumps(fake.payload)
        assert "running shoes" in wire
        assert "mkbhd merch" in wire

    @pytest.mark.asyncio
    async def test_year_month_range_uses_enum_month_names(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        await _run(gads, "google_ads_keyword_metrics",
                   {"keywords": ["running shoes"],
                    "start_year_month": "2025-01", "end_year_month": "2025-12"})
        options = fake.payload["historicalMetricsOptions"]["yearMonthRange"]
        assert str(options["start"]["year"]) == "2025"
        assert options["start"]["month"] == "JANUARY"
        assert str(options["end"]["year"]) == "2025"
        assert options["end"]["month"] == "DECEMBER"

    @pytest.mark.asyncio
    async def test_no_year_month_range_when_dates_omitted(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        await _run(gads, "google_ads_keyword_metrics", {"keywords": ["running shoes"]})
        assert "yearMonthRange" not in json.dumps(fake.payload)

    @pytest.mark.asyncio
    async def test_rejects_more_than_ten_thousand_keywords(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        too_many = [f"keyword {i}" for i in range(10001)]
        with pytest.raises((ToolError, ValidationError)):
            await _run(gads, "google_ads_keyword_metrics", {"keywords": too_many})

    @pytest.mark.asyncio
    async def test_prose_summary_mentions_a_keyword(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        result = await _run(gads, "google_ads_keyword_metrics",
                            {"keywords": ["running shoes"]})
        assert "running shoes" in _prose(result)


# ---------------------------------------------------------------------------
# THE PRESENCE TRAP — absent bid fields must map to None, never 0.0
# ---------------------------------------------------------------------------

class TestAbsentBidFields:
    @pytest.mark.asyncio
    async def test_historical_metrics_absent_bids_are_none(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, HISTORICAL_RESPONSE)
        result = await _run(gads, "google_ads_keyword_metrics",
                            {"keywords": ["running shoes", "mkbhd merch"]})
        row = next(r for r in _structured(result)["keywords"] if r["text"] == "mkbhd merch")
        assert row["avg_monthly_searches"] == 590, "volume is present and must survive"
        assert row["low_top_of_page_bid"] is None
        assert row["high_top_of_page_bid"] is None
        assert row["low_top_of_page_bid"] != 0.0
        assert row["high_top_of_page_bid"] != 0.0

    def test_the_absent_keys_really_are_absent_in_the_fixture(self):
        """Guards the fixture itself: absent, not zero, not null."""
        raw = next(r for r in HISTORICAL_RESPONSE["results"] if r["text"] == "mkbhd merch")
        assert "lowTopOfPageBidMicros" not in raw["keywordMetrics"]
        assert "highTopOfPageBidMicros" not in raw["keywordMetrics"]

    @pytest.mark.asyncio
    async def test_keyword_ideas_absent_bids_are_none(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes"]})
        idea = next(i for i in _structured(result)["ideas"] if i["text"] == "mkbhd merch")
        assert idea["low_top_of_page_bid"] is None
        assert idea["high_top_of_page_bid"] is None
        assert idea["avg_monthly_searches"] == 590
        assert idea["competition"] == "LOW"
        assert idea["competition_index"] == 2

    @pytest.mark.asyncio
    async def test_absent_bids_do_not_break_opportunity_scoring(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes"]})
        idea = next(i for i in _structured(result)["ideas"] if i["text"] == "mkbhd merch")
        assert idea["opportunity_score"] == pytest.approx(590 / 3)

    @pytest.mark.asyncio
    async def test_entry_with_no_metrics_block_at_all_is_all_none(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes"]})
        idea = next(i for i in _structured(result)["ideas"]
                    if i["text"] == "wireless earbuds for running")
        assert idea["avg_monthly_searches"] is None
        assert idea["competition"] is None
        assert idea["competition_index"] is None
        assert idea["low_top_of_page_bid"] is None
        assert idea["high_top_of_page_bid"] is None
        assert idea["opportunity_score"] is None


# ---------------------------------------------------------------------------
# Tool 2 — google_ads_keyword_ideas
# ---------------------------------------------------------------------------

class TestKeywordIdeas:
    @pytest.mark.asyncio
    async def test_sorted_by_volume_desc(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes"]})
        texts = [i["text"] for i in _structured(result)["ideas"]]
        assert texts == [
            "running shoes",
            "nike running shoes",
            "best running shoes for flat feet",
            "trail running shoes women",
            "barefoot running shoes",
            "mkbhd merch",
            "wireless earbuds for running",
        ]

    @pytest.mark.asyncio
    async def test_total_returned(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes"]})
        assert _structured(result)["total_returned"] == 7

    @pytest.mark.asyncio
    async def test_top_by_volume_capped_at_fifteen(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes"]})
        top = _structured(result)["top_by_volume"]
        assert len(top) <= 15
        assert top[0]["text"] == "running shoes"

    @pytest.mark.asyncio
    async def test_top_opportunities_ranked_by_score_not_volume(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes"]})
        top = _structured(result)["top_opportunities"]
        assert len(top) <= 10
        assert [i["text"] for i in top] == [
            "running shoes",
            "nike running shoes",
            "trail running shoes women",       # 14.8k @ 34 beats ...
            "best running shoes for flat feet",  # ... 22k @ 54
            "barefoot running shoes",
            "mkbhd merch",
        ]

    @pytest.mark.asyncio
    async def test_easy_targets_are_low_and_medium_only(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes"]})
        easy = _structured(result)["easy_targets"]
        assert len(easy) <= 10
        assert {i["text"] for i in easy} == {
            "best running shoes for flat feet",
            "trail running shoes women",
            "barefoot running shoes",
            "mkbhd merch",
        }
        assert all(i["competition"] in {"LOW", "MEDIUM"} for i in easy)

    @pytest.mark.asyncio
    async def test_opportunity_scores(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes"]})
        by_text = {i["text"]: i["opportunity_score"] for i in _structured(result)["ideas"]}
        assert by_text["running shoes"] == pytest.approx(301000 / 101)
        assert by_text["nike running shoes"] == pytest.approx(165000 / 92)
        assert by_text["trail running shoes women"] == pytest.approx(14800 / 35)

    @pytest.mark.asyncio
    async def test_limit_caps_the_ideas_returned(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes"], "limit": 3})
        ideas = _structured(result)["ideas"]
        assert len(ideas) == 3
        assert [i["text"] for i in ideas] == [
            "running shoes", "nike running shoes", "best running shoes for flat feet",
        ]

    @pytest.mark.asyncio
    async def test_single_page_only_no_auto_pagination(self, gads, monkeypatch):
        """A nextPageToken in the response must NOT trigger a second request."""
        paged = dict(IDEAS_RESPONSE, nextPageToken="CAESBgiAgICAAg")
        fake = _patch_post(monkeypatch, gads, paged)
        await _run(gads, "google_ads_keyword_ideas", {"seed_keywords": ["running shoes"]})
        assert len(fake.calls) == 1

    @pytest.mark.asyncio
    async def test_seed_echo_for_keywords(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes", "trail shoes"]})
        seed = _structured(result)["seed"]
        assert seed["type"] == "keywords"
        assert seed["value"] == ["running shoes", "trail shoes"]

    @pytest.mark.asyncio
    async def test_seed_echo_for_url(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_url": "https://www.fourthwall.com/pricing"})
        seed = _structured(result)["seed"]
        assert seed["type"] == "url"
        assert seed["value"] == "https://www.fourthwall.com/pricing"

    @pytest.mark.asyncio
    async def test_seed_echo_for_site(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        result = await _run(gads, "google_ads_keyword_ideas",
                            {"seed_site": "fourthwall.com"})
        seed = _structured(result)["seed"]
        assert seed["type"] == "site"
        assert seed["value"] == "fourthwall.com"

    @pytest.mark.asyncio
    async def test_seed_keywords_reach_the_wire(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        await _run(gads, "google_ads_keyword_ideas",
                   {"seed_keywords": ["running shoes", "trail shoes"]})
        wire = json.dumps(fake.payload)
        assert "running shoes" in wire
        assert "trail shoes" in wire
        assert "geoTargetConstants/2840" in wire
        assert "languageConstants/1000" in wire


class TestKeywordIdeasSeedValidation:
    @pytest.mark.asyncio
    async def test_twenty_seeds_is_allowed(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        seeds = [f"running shoes {i}" for i in range(20)]
        result = await _run(gads, "google_ads_keyword_ideas", {"seed_keywords": seeds})
        assert _structured(result)["seed"]["type"] == "keywords"

    @pytest.mark.asyncio
    async def test_twenty_one_seeds_raises(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        seeds = [f"running shoes {i}" for i in range(21)]
        with pytest.raises(ToolError):
            await _run(gads, "google_ads_keyword_ideas", {"seed_keywords": seeds})

    @pytest.mark.asyncio
    async def test_seed_cap_error_mentions_the_limit(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        seeds = [f"running shoes {i}" for i in range(25)]
        with pytest.raises(ToolError) as exc:
            await _run(gads, "google_ads_keyword_ideas", {"seed_keywords": seeds})
        assert "20" in str(exc.value)

    @pytest.mark.asyncio
    async def test_no_request_is_made_when_seeds_are_over_cap(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        seeds = [f"running shoes {i}" for i in range(21)]
        with pytest.raises(ToolError):
            await _run(gads, "google_ads_keyword_ideas", {"seed_keywords": seeds})
        assert fake.calls == []

    @pytest.mark.asyncio
    async def test_no_seed_at_all_raises(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        with pytest.raises(ToolError):
            await _run(gads, "google_ads_keyword_ideas", {})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("arguments", [
        {"seed_keywords": ["running shoes"], "seed_url": "https://example.com/x"},
        {"seed_keywords": ["running shoes"], "seed_site": "example.com"},
        {"seed_url": "https://example.com/x", "seed_site": "example.com"},
        {"seed_keywords": ["running shoes"], "seed_url": "https://example.com/x",
         "seed_site": "example.com"},
    ])
    async def test_seed_kinds_are_mutually_exclusive(self, gads, monkeypatch, arguments):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        with pytest.raises(ToolError):
            await _run(gads, "google_ads_keyword_ideas", arguments)

    @pytest.mark.asyncio
    async def test_empty_seed_keyword_list_is_treated_as_no_seed(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, IDEAS_RESPONSE)
        with pytest.raises(ToolError):
            await _run(gads, "google_ads_keyword_ideas", {"seed_keywords": []})


# ---------------------------------------------------------------------------
# Tool 3 — google_ads_forecast_budget
# ---------------------------------------------------------------------------

def _future(days: int) -> str:
    return (dt.date.today() + dt.timedelta(days=days)).isoformat()


class TestForecastBudget:
    @pytest.mark.asyncio
    async def test_maps_the_v25_forecast_response(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        result = await _run(gads, "google_ads_forecast_budget", {
            "keywords": ["running shoes"], "max_cpc_bid": 2.5,
            "start_date": _future(7), "end_date": _future(37),
        })
        structured = _structured(result)
        assert structured["clicks"] == pytest.approx(58109.14453125)
        assert structured["cost"] == pytest.approx(65339.95434)
        assert structured["average_cpc"] == pytest.approx(1.124435)

    @pytest.mark.asyncio
    async def test_conversion_fields_absent_under_manual_cpc(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        result = await _run(gads, "google_ads_forecast_budget", {
            "keywords": ["running shoes"], "max_cpc_bid": 2.5,
            "start_date": _future(7), "end_date": _future(37),
        })
        structured = _structured(result)
        assert structured["conversions"] is None
        assert structured["average_cpa"] is None

    @pytest.mark.asyncio
    async def test_no_impressions_or_ctr_in_v25(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        result = await _run(gads, "google_ads_forecast_budget", {
            "keywords": ["running shoes"], "max_cpc_bid": 2.5,
            "start_date": _future(7), "end_date": _future(37),
        })
        structured = _structured(result)
        assert "impressions" not in structured
        assert "ctr" not in structured
        assert "v25" in structured["note"]
        assert structured["note"] == (
            "Google Ads API v25 does not return impression or CTR forecasts."
        )

    @pytest.mark.asyncio
    async def test_cost_per_click_is_derived(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        result = await _run(gads, "google_ads_forecast_budget", {
            "keywords": ["running shoes"], "max_cpc_bid": 2.5,
            "start_date": _future(7), "end_date": _future(37),
        })
        structured = _structured(result)
        assert structured["cost_per_click"] == pytest.approx(
            65339.95434 / 58109.14453125
        )

    @pytest.mark.asyncio
    async def test_echoes_the_request(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        result = await _run(gads, "google_ads_forecast_budget", {
            "keywords": ["running shoes", "trail running shoes"],
            "match_type": "PHRASE", "max_cpc_bid": 2.5, "daily_budget": 150.0,
            "start_date": _future(7), "end_date": _future(37),
        })
        structured = _structured(result)
        assert structured["keywords"] == ["running shoes", "trail running shoes"]
        assert structured["match_type"] == "PHRASE"
        assert structured["max_cpc_bid"] == pytest.approx(2.5)
        assert structured["daily_budget"] == pytest.approx(150.0)
        assert structured["period"] == {"start_date": _future(7), "end_date": _future(37)}

    @pytest.mark.asyncio
    async def test_daily_budget_none_when_omitted(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        result = await _run(gads, "google_ads_forecast_budget", {
            "keywords": ["running shoes"], "max_cpc_bid": 2.5,
            "start_date": _future(7), "end_date": _future(37),
        })
        assert _structured(result)["daily_budget"] is None

    @pytest.mark.asyncio
    async def test_payload_is_the_v25_camel_case_shape(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        await _run(gads, "google_ads_forecast_budget", {
            "keywords": ["running shoes", "trail running shoes"],
            "match_type": "EXACT", "max_cpc_bid": 2.5, "daily_budget": 150.0,
            "start_date": _future(7), "end_date": _future(37),
        })
        campaign = fake.payload["campaign"]
        bidding = campaign["biddingStrategy"]["manualCpcBiddingStrategy"]
        assert str(bidding["maxCpcBidMicros"]) == "2500000"
        assert str(bidding["dailyBudgetMicros"]) == "150000000"
        assert campaign["geoTargetConstants"] == ["geoTargetConstants/2840"]
        assert campaign["languageConstants"] == ["languageConstants/1000"]
        keywords = campaign["adGroups"][0]["keywords"]
        assert [k["text"] for k in keywords] == ["running shoes", "trail running shoes"]
        assert {k["matchType"] for k in keywords} == {"EXACT"}
        assert fake.payload["forecastPeriod"] == {
            "startDate": _future(7), "endDate": _future(37),
        }

    @pytest.mark.asyncio
    async def test_snake_case_never_appears_on_the_wire(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        await _run(gads, "google_ads_forecast_budget", {
            "keywords": ["running shoes"], "max_cpc_bid": 1.0,
            "start_date": _future(7), "end_date": _future(37),
        })
        wire = json.dumps(fake.payload)
        for snake in ("max_cpc_bid_micros", "forecast_period", "start_date",
                      "geo_target_constants", "ad_groups", "match_type"):
            assert snake not in wire, f"{snake} is not accepted by v25"


class TestForecastDateValidation:
    @pytest.mark.asyncio
    async def test_default_period_is_next_monday_plus_thirty_days(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        result = await _run(gads, "google_ads_forecast_budget",
                            {"keywords": ["running shoes"], "max_cpc_bid": 2.5})
        period = _structured(result)["period"]
        start = dt.date.fromisoformat(period["start_date"])
        end = dt.date.fromisoformat(period["end_date"])
        today = dt.date.today()
        assert start.weekday() == 0, "default start is the next Monday"
        assert start > today
        assert start <= today + dt.timedelta(days=7)
        assert (end - start).days == 30

    @pytest.mark.asyncio
    async def test_start_date_in_the_past_raises(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        with pytest.raises(ToolError):
            await _run(gads, "google_ads_forecast_budget", {
                "keywords": ["running shoes"], "max_cpc_bid": 2.5,
                "start_date": _future(-1), "end_date": _future(30),
            })
        assert fake.calls == []

    @pytest.mark.asyncio
    async def test_start_date_far_in_the_past_raises(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        with pytest.raises(ToolError):
            await _run(gads, "google_ads_forecast_budget", {
                "keywords": ["running shoes"], "max_cpc_bid": 2.5,
                "start_date": "2020-01-01", "end_date": "2020-02-01",
            })

    @pytest.mark.asyncio
    async def test_end_date_more_than_a_year_out_raises(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        with pytest.raises(ToolError):
            await _run(gads, "google_ads_forecast_budget", {
                "keywords": ["running shoes"], "max_cpc_bid": 2.5,
                "start_date": _future(7), "end_date": _future(400),
            })
        assert fake.calls == []

    @pytest.mark.asyncio
    async def test_end_date_just_inside_a_year_is_accepted(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        result = await _run(gads, "google_ads_forecast_budget", {
            "keywords": ["running shoes"], "max_cpc_bid": 2.5,
            "start_date": _future(1), "end_date": _future(360),
        })
        assert _structured(result)["period"]["end_date"] == _future(360)

    @pytest.mark.asyncio
    async def test_error_message_explains_which_date_is_wrong(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, FORECAST_RESPONSE)
        with pytest.raises(ToolError) as exc:
            await _run(gads, "google_ads_forecast_budget", {
                "keywords": ["running shoes"], "max_cpc_bid": 2.5,
                "start_date": _future(-30), "end_date": _future(30),
            })
        assert "start" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Tool 4 — google_ads_budget_curve
# ---------------------------------------------------------------------------

def _forecast(clicks: float, cost_dollars: float, avg_cpc_dollars: float) -> dict:
    return {"campaignForecastMetrics": {
        "clicks": clicks,
        "costMicros": str(int(round(cost_dollars * 1_000_000))),
        "averageCpcMicros": str(int(round(avg_cpc_dollars * 1_000_000))),
    }}


CURVE_RESPONSES = [
    _forecast(100.0, 50.0, 0.50),    # bid 0.25
    _forecast(180.0, 108.0, 0.60),   # bid 0.50  -> +80 clicks for +$58  => 0.725/click
    _forecast(200.0, 160.0, 0.80),   # bid 1.00  -> +20 clicks for +$52  => 2.600/click
]


class TestBudgetCurve:
    @pytest.mark.asyncio
    async def test_one_request_per_bid_in_order(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, CURVE_RESPONSES)
        await _run(gads, "google_ads_budget_curve", {
            "keywords": ["running shoes"], "bids": [0.25, 0.5, 1.0],
            "start_date": _future(7), "end_date": _future(37),
        })
        assert len(fake.calls) == 3
        sent = [
            str(call["payload"]["campaign"]["biddingStrategy"]
                    ["manualCpcBiddingStrategy"]["maxCpcBidMicros"])
            for call in fake.calls
        ]
        assert sent == ["250000", "500000", "1000000"]

    @pytest.mark.asyncio
    async def test_curve_rows_carry_the_forecast(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, CURVE_RESPONSES)
        result = await _run(gads, "google_ads_budget_curve", {
            "keywords": ["running shoes"], "bids": [0.25, 0.5, 1.0],
            "start_date": _future(7), "end_date": _future(37),
        })
        curve = _structured(result)["curve"]
        assert [row["max_cpc_bid"] for row in curve] == [0.25, 0.5, 1.0]
        assert [row["clicks"] for row in curve] == [100.0, 180.0, 200.0]
        assert curve[0]["cost"] == pytest.approx(50.0)
        assert curve[1]["average_cpc"] == pytest.approx(0.60)

    @pytest.mark.asyncio
    async def test_first_row_has_no_marginals(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, CURVE_RESPONSES)
        result = await _run(gads, "google_ads_budget_curve", {
            "keywords": ["running shoes"], "bids": [0.25, 0.5, 1.0],
            "start_date": _future(7), "end_date": _future(37),
        })
        first = _structured(result)["curve"][0]
        assert first["marginal_clicks"] is None
        assert first["marginal_cost"] is None
        assert first["marginal_cost_per_click"] is None

    @pytest.mark.asyncio
    async def test_marginal_math(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, CURVE_RESPONSES)
        result = await _run(gads, "google_ads_budget_curve", {
            "keywords": ["running shoes"], "bids": [0.25, 0.5, 1.0],
            "start_date": _future(7), "end_date": _future(37),
        })
        curve = _structured(result)["curve"]
        assert curve[1]["marginal_clicks"] == pytest.approx(80.0)
        assert curve[1]["marginal_cost"] == pytest.approx(58.0)
        assert curve[1]["marginal_cost_per_click"] == pytest.approx(58.0 / 80.0)
        assert curve[2]["marginal_clicks"] == pytest.approx(20.0)
        assert curve[2]["marginal_cost"] == pytest.approx(52.0)
        assert curve[2]["marginal_cost_per_click"] == pytest.approx(52.0 / 20.0)

    @pytest.mark.asyncio
    async def test_best_efficiency_bid_is_lowest_marginal_cost_per_click(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, CURVE_RESPONSES)
        result = await _run(gads, "google_ads_budget_curve", {
            "keywords": ["running shoes"], "bids": [0.25, 0.5, 1.0],
            "start_date": _future(7), "end_date": _future(37),
        })
        assert _structured(result)["best_efficiency_bid"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_flat_step_does_not_divide_by_zero(self, gads, monkeypatch):
        """Clicks plateau: marginal_clicks == 0 must not blow up."""
        responses = CURVE_RESPONSES + [_forecast(200.0, 200.0, 1.0)]
        _patch_post(monkeypatch, gads, responses)
        result = await _run(gads, "google_ads_budget_curve", {
            "keywords": ["running shoes"], "bids": [0.25, 0.5, 1.0, 2.0],
            "start_date": _future(7), "end_date": _future(37),
        })
        last = _structured(result)["curve"][-1]
        assert last["marginal_clicks"] == pytest.approx(0.0)
        assert last["marginal_cost"] == pytest.approx(40.0)
        assert last["marginal_cost_per_click"] is None

    @pytest.mark.asyncio
    async def test_single_bid_has_no_best_efficiency_bid(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, CURVE_RESPONSES[:1])
        result = await _run(gads, "google_ads_budget_curve", {
            "keywords": ["running shoes"], "bids": [0.25],
            "start_date": _future(7), "end_date": _future(37),
        })
        structured = _structured(result)
        assert len(structured["curve"]) == 1
        assert structured["best_efficiency_bid"] is None

    @pytest.mark.asyncio
    async def test_default_bid_ladder(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, CURVE_RESPONSES)
        result = await _run(gads, "google_ads_budget_curve", {
            "keywords": ["running shoes"],
            "start_date": _future(7), "end_date": _future(37),
        })
        bids = [row["max_cpc_bid"] for row in _structured(result)["curve"]]
        assert bids == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        assert len(fake.calls) == 6

    @pytest.mark.asyncio
    async def test_more_than_ten_bids_raises(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, CURVE_RESPONSES)
        with pytest.raises((ToolError, ValidationError)):
            await _run(gads, "google_ads_budget_curve", {
                "keywords": ["running shoes"],
                "bids": [0.1 * i for i in range(1, 12)],
                "start_date": _future(7), "end_date": _future(37),
            })
        assert fake.calls == []

    @pytest.mark.asyncio
    async def test_ten_bids_is_allowed(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, CURVE_RESPONSES)
        result = await _run(gads, "google_ads_budget_curve", {
            "keywords": ["running shoes"],
            "bids": [round(0.25 * i, 2) for i in range(1, 11)],
            "start_date": _future(7), "end_date": _future(37),
        })
        assert len(_structured(result)["curve"]) == 10

    @pytest.mark.asyncio
    async def test_echoes_period_keywords_match_type(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, CURVE_RESPONSES)
        result = await _run(gads, "google_ads_budget_curve", {
            "keywords": ["running shoes"], "match_type": "PHRASE",
            "bids": [0.25, 0.5, 1.0],
            "start_date": _future(7), "end_date": _future(37),
        })
        structured = _structured(result)
        assert structured["keywords"] == ["running shoes"]
        assert structured["match_type"] == "PHRASE"
        assert structured["period"] == {"start_date": _future(7), "end_date": _future(37)}

    @pytest.mark.asyncio
    async def test_reports_progress_once_per_bid(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, CURVE_RESPONSES)
        ctx = RecordingContext()
        await _call_fn(gads, "google_ads_budget_curve", ctx=ctx,
                       keywords=["running shoes"], bids=[0.25, 0.5, 1.0],
                       start_date=_future(7), end_date=_future(37))
        assert len(ctx.progress) == 3
        assert all(step[1] == 3 for step in ctx.progress)

    @pytest.mark.asyncio
    async def test_past_start_date_raises_before_any_request(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, CURVE_RESPONSES)
        with pytest.raises(ToolError):
            await _run(gads, "google_ads_budget_curve", {
                "keywords": ["running shoes"], "bids": [0.25, 0.5],
                "start_date": _future(-5), "end_date": _future(30),
            })
        assert fake.calls == []


# ---------------------------------------------------------------------------
# Tool 5 — google_ads_seasonality
# ---------------------------------------------------------------------------

# 2024: 60,70,...,170 (sum 1380) / 2025: 120,140,...,340 (sum 2760)
# total 4140 over 24 months -> mean 172.5
MEAN_MONTHLY = 172.5


class TestSeasonality:
    @pytest.mark.asyncio
    async def test_totals_and_mean(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, SEASONALITY_RESPONSE)
        result = await _run(gads, "google_ads_seasonality",
                            {"keywords": ["running shoes"]})
        row = _structured(result)["keywords"][0]
        assert row["text"] == "running shoes"
        assert row["total_searches"] == 4140
        assert row["mean_monthly"] == pytest.approx(MEAN_MONTHLY)

    @pytest.mark.asyncio
    async def test_monthly_index_is_searches_over_mean(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, SEASONALITY_RESPONSE)
        result = await _run(gads, "google_ads_seasonality",
                            {"keywords": ["running shoes"]})
        monthly = _structured(result)["keywords"][0]["monthly"]
        assert len(monthly) == 24
        first = monthly[0]
        assert first["year"] == 2024
        assert first["month"] == "JANUARY"
        assert first["searches"] == 60
        assert first["index"] == pytest.approx(60 / MEAN_MONTHLY)
        last = monthly[-1]
        assert last["year"] == 2025
        assert last["month"] == "DECEMBER"
        assert last["index"] == pytest.approx(340 / MEAN_MONTHLY)

    @pytest.mark.asyncio
    async def test_index_of_one_is_an_average_month(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, SEASONALITY_RESPONSE)
        result = await _run(gads, "google_ads_seasonality",
                            {"keywords": ["running shoes"]})
        monthly = _structured(result)["keywords"][0]["monthly"]
        above = [m for m in monthly if m["index"] > 1.0]
        below = [m for m in monthly if m["index"] < 1.0]
        assert above and below
        assert sum(m["index"] for m in monthly) == pytest.approx(len(monthly))

    @pytest.mark.asyncio
    async def test_peak_and_trough(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, SEASONALITY_RESPONSE)
        result = await _run(gads, "google_ads_seasonality",
                            {"keywords": ["running shoes"]})
        row = _structured(result)["keywords"][0]
        assert row["peak_month"] == "DECEMBER"
        assert row["trough_month"] == "JANUARY"

    @pytest.mark.asyncio
    async def test_yoy_change_is_a_fraction(self, gads, monkeypatch):
        """Last 12mo = 2760, prior 12mo = 1380 -> +100% -> 1.0."""
        _patch_post(monkeypatch, gads, SEASONALITY_RESPONSE)
        result = await _run(gads, "google_ads_seasonality",
                            {"keywords": ["running shoes"]})
        assert _structured(result)["keywords"][0]["yoy_change"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_aggregate_monthly_index_covers_all_twelve_months(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, SEASONALITY_RESPONSE)
        result = await _run(gads, "google_ads_seasonality",
                            {"keywords": ["running shoes"]})
        index = _structured(result)["aggregate"]["monthly_index"]
        assert set(index) == {
            "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
            "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
        }

    @pytest.mark.asyncio
    async def test_aggregate_index_math(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, SEASONALITY_RESPONSE)
        result = await _run(gads, "google_ads_seasonality",
                            {"keywords": ["running shoes"]})
        index = _structured(result)["aggregate"]["monthly_index"]
        # JANUARY across both years: (60 + 120) / 2 = 90 -> 90 / 172.5
        assert index["JANUARY"] == pytest.approx(90 / MEAN_MONTHLY)
        # DECEMBER: (170 + 340) / 2 = 255 -> 255 / 172.5
        assert index["DECEMBER"] == pytest.approx(255 / MEAN_MONTHLY)
        assert sum(index.values()) == pytest.approx(12.0)

    @pytest.mark.asyncio
    async def test_aggregate_peak_and_trough(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, SEASONALITY_RESPONSE)
        result = await _run(gads, "google_ads_seasonality",
                            {"keywords": ["running shoes"]})
        aggregate = _structured(result)["aggregate"]
        assert aggregate["peak_month"] == "DECEMBER"
        assert aggregate["trough_month"] == "JANUARY"

    @pytest.mark.asyncio
    async def test_default_window_is_forty_eight_months(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, SEASONALITY_RESPONSE)
        result = await _run(gads, "google_ads_seasonality",
                            {"keywords": ["running shoes"]})
        structured = _structured(result)
        assert structured["months_requested"] == 48
        assert structured["geo_target_ids"] == [2840]
        assert structured["language_id"] == 1000

    @pytest.mark.asyncio
    async def test_months_back_is_capped_at_forty_eight(self, gads, monkeypatch):
        _patch_post(monkeypatch, gads, SEASONALITY_RESPONSE)
        result = await _run(gads, "google_ads_seasonality",
                            {"keywords": ["running shoes"], "months_back": 60})
        assert _structured(result)["months_requested"] == 48

    @pytest.mark.asyncio
    async def test_requests_an_explicit_year_month_range(self, gads, monkeypatch):
        fake = _patch_post(monkeypatch, gads, SEASONALITY_RESPONSE)
        await _run(gads, "google_ads_seasonality",
                   {"keywords": ["running shoes"], "months_back": 24})
        assert "yearMonthRange" in json.dumps(fake.payload)
        assert "running shoes" in json.dumps(fake.payload)


# ---------------------------------------------------------------------------
# HTTP layer: headers, error envelope, throttle
# ---------------------------------------------------------------------------

class TestRequestHeaders:
    @pytest.mark.asyncio
    async def test_sends_auth_developer_token_and_login_customer_id(self, gads, monkeypatch):
        _patch_token(monkeypatch, gads)
        with aioresponses() as mocked:
            mocked.post(ANY_URL, status=200, payload=IDEAS_RESPONSE, repeat=True)
            await _run(gads, "google_ads_keyword_ideas", {"seed_keywords": ["running shoes"]})
            request = next(iter(mocked.requests.values()))[0]
        headers = {k.lower(): v for k, v in (request.kwargs.get("headers") or {}).items()}
        assert headers["authorization"] == f"Bearer {FAKE_TOKEN}"
        assert headers["developer-token"] == ENV["GOOGLE_ADS_DEVELOPER_TOKEN"]
        assert headers["login-customer-id"] == ENV["GOOGLE_ADS_LOGIN_CUSTOMER_ID"]
        assert headers["content-type"] == "application/json"

    @pytest.mark.asyncio
    async def test_url_targets_the_client_customer_id_and_api_version(self, gads, monkeypatch):
        _patch_token(monkeypatch, gads)
        with aioresponses() as mocked:
            mocked.post(ANY_URL, status=200, payload=IDEAS_RESPONSE, repeat=True)
            await _run(gads, "google_ads_keyword_ideas", {"seed_keywords": ["running shoes"]})
            url = str(next(iter(mocked.requests))[1])
        assert "/v25/" in url
        assert ENV["GOOGLE_ADS_CUSTOMER_ID"] in url
        assert "generateKeywordIdeas" in url


class TestGoogleAdsErrorEnvelope:
    @pytest.mark.asyncio
    async def test_403_raises_tool_error_with_the_real_message(self, gads, monkeypatch):
        _patch_token(monkeypatch, gads)
        with aioresponses() as mocked:
            mocked.post(ANY_URL, status=403, payload=ERROR_403, repeat=True)
            with pytest.raises(ToolError) as exc:
                await _run(gads, "google_ads_keyword_ideas",
                           {"seed_keywords": ["running shoes"]})
        assert "developer token is not approved" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_error_code_is_surfaced(self, gads, monkeypatch):
        _patch_token(monkeypatch, gads)
        with aioresponses() as mocked:
            mocked.post(ANY_URL, status=403, payload=ERROR_403, repeat=True)
            with pytest.raises(ToolError) as exc:
                await _run(gads, "google_ads_keyword_ideas",
                           {"seed_keywords": ["running shoes"]})
        assert "DEVELOPER_TOKEN_NOT_APPROVED" in str(exc.value)

    @pytest.mark.asyncio
    async def test_error_is_not_a_generic_status_message(self, gads, monkeypatch):
        _patch_token(monkeypatch, gads)
        with aioresponses() as mocked:
            mocked.post(ANY_URL, status=403, payload=ERROR_403, repeat=True)
            with pytest.raises(ToolError) as exc:
                await _run(gads, "google_ads_keyword_ideas",
                           {"seed_keywords": ["running shoes"]})
        message = str(exc.value)
        assert message.strip() not in {"403", "Forbidden", "403 Forbidden"}
        assert len(message) > 30

    @pytest.mark.asyncio
    async def test_error_envelope_on_every_tool(self, gads, monkeypatch):
        _patch_token(monkeypatch, gads)
        with aioresponses() as mocked:
            mocked.post(ANY_URL, status=403, payload=ERROR_403, repeat=True)
            with pytest.raises(ToolError):
                await _run(gads, "google_ads_keyword_metrics",
                           {"keywords": ["running shoes"]})

    @pytest.mark.asyncio
    async def test_non_google_error_body_still_raises_tool_error(self, gads, monkeypatch):
        """A 500 with an HTML body (LB/proxy) must not leak a raw aiohttp error."""
        _patch_token(monkeypatch, gads)
        with aioresponses() as mocked:
            mocked.post(ANY_URL, status=500, body="<html>upstream boom</html>", repeat=True)
            with pytest.raises(ToolError):
                await _run(gads, "google_ads_keyword_ideas",
                           {"seed_keywords": ["running shoes"]})


class TestThrottle:
    @pytest.mark.asyncio
    async def test_consecutive_calls_are_spaced_at_least_1_1s(self, gads, monkeypatch):
        """Keyword planning is capped at 1 QPS per customer id."""
        _patch_token(monkeypatch, gads)
        with aioresponses() as mocked:
            mocked.post(ANY_URL, status=200, payload=IDEAS_RESPONSE, repeat=True)
            started = time.monotonic()
            await _run(gads, "google_ads_keyword_ideas", {"seed_keywords": ["a"]})
            after_first = time.monotonic()
            await _run(gads, "google_ads_keyword_ideas", {"seed_keywords": ["b"]})
            after_second = time.monotonic()
        assert after_first - started < 1.0, "the first call must not be delayed"
        assert after_second - after_first >= 1.1

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_serialised(self, gads, monkeypatch):
        """Two coroutines racing must still be >=1.1s apart, not simultaneous."""
        _patch_token(monkeypatch, gads)
        with aioresponses() as mocked:
            mocked.post(ANY_URL, status=200, payload=IDEAS_RESPONSE, repeat=True)
            started = time.monotonic()
            await asyncio.gather(
                _run(gads, "google_ads_keyword_ideas", {"seed_keywords": ["a"]}),
                _run(gads, "google_ads_keyword_ideas", {"seed_keywords": ["b"]}),
            )
            elapsed = time.monotonic() - started
        assert elapsed >= 1.1

    @pytest.mark.asyncio
    async def test_budget_curve_sweeps_sequentially(self, gads, monkeypatch):
        """Sequential, not gathered — 3 bids means 2 throttle gaps."""
        _patch_token(monkeypatch, gads)
        with aioresponses() as mocked:
            mocked.post(ANY_URL, status=200, payload=FORECAST_RESPONSE, repeat=True)
            started = time.monotonic()
            await _run(gads, "google_ads_budget_curve", {
                "keywords": ["running shoes"], "bids": [0.25, 0.5, 1.0],
                "start_date": _future(7), "end_date": _future(37),
            })
            elapsed = time.monotonic() - started
        assert elapsed >= 2.2


# ---------------------------------------------------------------------------
# Live integration (skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLiveAPI:
    @pytest.mark.asyncio
    async def test_historical_metrics_for_a_known_keyword(self):
        module = importlib.reload(google_ads_planner)
        result = await _run(module, "google_ads_keyword_metrics",
                            {"keywords": ["running shoes"]})
        structured = _structured(result)
        assert structured["returned"] >= 1
        row = structured["keywords"][0]
        assert row["text"] == "running shoes"
        assert row["avg_monthly_searches"] > 10000
        assert row["competition"] in {"LOW", "MEDIUM", "HIGH"}

    @pytest.mark.asyncio
    async def test_keyword_ideas_from_a_seed(self):
        module = importlib.reload(google_ads_planner)
        result = await _run(module, "google_ads_keyword_ideas",
                            {"seed_keywords": ["running shoes"], "limit": 25})
        structured = _structured(result)
        assert 0 < structured["total_returned"] <= 25
        assert structured["seed"] == {"type": "keywords", "value": ["running shoes"]}

    @pytest.mark.asyncio
    async def test_forecast_returns_no_impressions_in_v25(self):
        module = importlib.reload(google_ads_planner)
        result = await _run(module, "google_ads_forecast_budget",
                            {"keywords": ["running shoes"], "max_cpc_bid": 1.5})
        structured = _structured(result)
        assert "impressions" not in structured
        assert "ctr" not in structured
        assert structured["clicks"] is None or structured["clicks"] >= 0
