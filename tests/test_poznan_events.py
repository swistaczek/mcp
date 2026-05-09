"""Unit tests for poznan_events.py — fixture-driven (no live network).

Live integrations are marked with @pytest.mark.integration and skipped by
default (run with `-m integration` to include them).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import poznan_events as pe

FIXTURES = Path(__file__).parent / "fixtures"
LIST_FIXTURE = FIXTURES / "poznan_events_list.html"
DETAIL_FIXTURE = FIXTURES / "poznan_event_detail.html"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestPolishDateConversion:
    def test_valid_date(self):
        assert pe._polish_date_to_iso("09.05.2026") == "2026-05-09"

    def test_first_of_month(self):
        assert pe._polish_date_to_iso("01.01.2025") == "2025-01-01"

    def test_invalid_date_returns_none(self):
        assert pe._polish_date_to_iso("not a date") is None

    def test_impossible_date_returns_none(self):
        assert pe._polish_date_to_iso("32.13.2026") is None

    def test_strips_whitespace(self):
        assert pe._polish_date_to_iso("  09.05.2026  ") == "2026-05-09"


class TestTimeNormalization:
    def test_already_padded(self):
        assert pe._normalize_time("09:00") == "09:00"

    def test_pads_single_digit_hour(self):
        assert pe._normalize_time("9:30") == "09:30"

    def test_invalid_returns_none(self):
        assert pe._normalize_time("morning") is None

    def test_strips_whitespace(self):
        assert pe._normalize_time("  09:00 ") == "09:00"


class TestExtractEventId:
    def test_extracts_id_from_canonical_path(self):
        href = "/mim/events/parkrun-poznan,179433.html"
        assert pe._extract_event_id(href) == "179433"

    def test_extracts_id_from_full_url(self):
        href = "https://www.poznan.pl/mim/events/spektakl-x,182776.html"
        assert pe._extract_event_id(href) == "182776"

    def test_returns_none_for_non_event_path(self):
        assert pe._extract_event_id("/mim/events/sport,c,214/") is None
        assert pe._extract_event_id("/mim/events/") is None


class TestNormalizeCategory:
    def test_none_returns_empty(self):
        assert pe._normalize_category(None) == ""

    def test_empty_returns_empty(self):
        assert pe._normalize_category("") == ""

    def test_int_stringifies(self):
        assert pe._normalize_category(214) == "214"

    def test_digit_string_passthrough(self):
        assert pe._normalize_category("214") == "214"

    def test_strips_whitespace(self):
        assert pe._normalize_category(" 214 ") == "214"

    def test_non_digit_raises(self):
        with pytest.raises(ValueError, match="numeric id"):
            pe._normalize_category("Sport")


class TestResolveEventUrl:
    def test_full_url_passthrough(self):
        url = "https://www.poznan.pl/mim/events/x,123.html"
        assert pe._resolve_event_url(url) == url

    def test_absolute_path(self):
        path = "/mim/events/x,123.html"
        assert pe._resolve_event_url(path) == "https://www.poznan.pl/mim/events/x,123.html"

    def test_numeric_id_only(self):
        # Source rewrites unknown slugs to canonical, so any slug is fine.
        out = pe._resolve_event_url("179433")
        assert out.startswith("https://www.poznan.pl/mim/events/")
        assert ",179433.html" in out

    def test_slug_id_pair(self):
        out = pe._resolve_event_url("parkrun-poznan,179433")
        assert out == "https://www.poznan.pl/mim/events/parkrun-poznan,179433.html"

    def test_slug_id_pair_with_html(self):
        out = pe._resolve_event_url("parkrun-poznan,179433.html")
        assert out == "https://www.poznan.pl/mim/events/parkrun-poznan,179433.html"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pe._resolve_event_url("   ")

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            pe._resolve_event_url("not-a-real-ref")


# ---------------------------------------------------------------------------
# List parser (fixture-driven)
# ---------------------------------------------------------------------------

class TestListParser:
    def setup_method(self):
        self.html = LIST_FIXTURE.read_text(encoding="utf-8")

    def test_parses_twenty_events(self):
        events = pe._parse_event_list(self.html)
        assert len(events) == 20

    def test_event_ids_are_unique(self):
        events = pe._parse_event_list(self.html)
        ids = [e["event_id"] for e in events if e["event_id"]]
        assert len(ids) == len(set(ids))
        assert all(i.isdigit() for i in ids)

    def test_first_event_shape(self):
        events = pe._parse_event_list(self.html)
        ev = events[0]
        # Every parsed event must have these populated.
        assert ev["event_id"] and ev["event_id"].isdigit()
        assert ev["title"]
        assert ev["detail_url"].startswith("https://www.poznan.pl/mim/events/")
        # Date must be in ISO form when source date is parseable.
        assert ev["date"] is None or ev["date"].count("-") == 2

    def test_date_is_iso(self):
        events = pe._parse_event_list(self.html)
        for ev in events:
            if ev["date"]:
                # YYYY-MM-DD
                y, m, d = ev["date"].split("-")
                assert len(y) == 4 and len(m) == 2 and len(d) == 2

    def test_time_is_padded(self):
        events = pe._parse_event_list(self.html)
        for ev in events:
            if ev["time"]:
                assert len(ev["time"]) == 5
                assert ev["time"][2] == ":"

    def test_categories_have_ids(self):
        events = pe._parse_event_list(self.html)
        all_cats = [c for ev in events for c in ev["categories"]]
        assert all_cats, "expected at least one category across events"
        for c in all_cats:
            assert c["id"] and c["id"].isdigit()
            assert c["name"]

    def test_thumbnails_absolute(self):
        events = pe._parse_event_list(self.html)
        thumbs = [e["thumbnail_url"] for e in events if e["thumbnail_url"]]
        assert thumbs, "expected at least one thumbnail"
        for t in thumbs:
            assert t.startswith("https://www.poznan.pl/")

    def test_returns_empty_for_empty_html(self):
        assert pe._parse_event_list("") == []

    def test_handles_card_without_title_link(self):
        broken = '<article class="event-box"></article>'
        assert pe._parse_event_list(broken) == []


# ---------------------------------------------------------------------------
# Categories parser
# ---------------------------------------------------------------------------

class TestCategoriesParser:
    def setup_method(self):
        self.html = LIST_FIXTURE.read_text(encoding="utf-8")

    def test_parses_categories(self):
        cats = pe._parse_categories(self.html)
        assert len(cats) >= 10

    def test_each_category_has_id_slug_name(self):
        cats = pe._parse_categories(self.html)
        for c in cats:
            assert c["id"].isdigit()
            assert c["slug"]
            assert c["name"]

    def test_known_category_present(self):
        cats = pe._parse_categories(self.html)
        sport = next((c for c in cats if c["id"] == "214"), None)
        assert sport is not None
        assert sport["slug"] == "sport"

    def test_ids_are_unique(self):
        cats = pe._parse_categories(self.html)
        ids = [c["id"] for c in cats]
        assert len(ids) == len(set(ids))

    def test_sorted_alphabetically(self):
        cats = pe._parse_categories(self.html)
        names_lower = [c["name"].lower() for c in cats]
        assert names_lower == sorted(names_lower)


# ---------------------------------------------------------------------------
# Detail parser (fixture-driven)
# ---------------------------------------------------------------------------

class TestDetailParser:
    def setup_method(self):
        self.html = DETAIL_FIXTURE.read_text(encoding="utf-8")
        self.url = (
            "https://www.poznan.pl/mim/events/"
            "parkrun-poznan-parkrun-lasek-marcelinski-parkrun-las-debinski,179433.html"
        )

    def test_extracts_title(self):
        ev = pe._parse_event_detail(self.html, self.url)
        assert ev["title"] == (
            "Parkrun Poznań, Parkrun Lasek Marceliński, Parkrun Las Dębiński"
        )

    def test_extracts_event_id_from_url(self):
        ev = pe._parse_event_detail(self.html, self.url)
        assert ev["event_id"] == "179433"

    def test_extracts_date_iso(self):
        ev = pe._parse_event_detail(self.html, self.url)
        assert ev["date"] == "2026-05-09"
        assert ev["date_raw"] == "09.05.2026"

    def test_extracts_time(self):
        ev = pe._parse_event_detail(self.html, self.url)
        assert ev["time"] == "09:00"

    def test_extracts_place(self):
        ev = pe._parse_event_detail(self.html, self.url)
        assert "Park Cytadela" in ev["place"]
        assert "Lasek Marceliński" in ev["place"]

    def test_extracts_categories(self):
        ev = pe._parse_event_detail(self.html, self.url)
        names = [c["name"] for c in ev["categories"]]
        assert "Sport" in names
        sport = next(c for c in ev["categories"] if c["name"] == "Sport")
        assert sport["id"] == "214"
        assert sport["slug"] == "sport"

    def test_short_description_from_og(self):
        ev = pe._parse_event_detail(self.html, self.url)
        assert ev["short_description"]
        assert "9 maja" in ev["short_description"]

    def test_long_description_strips_ui_noise(self):
        ev = pe._parse_event_detail(self.html, self.url)
        # The print block contains banner/share UI; we strip it.
        assert ev["description"]
        # No "Pobierz baner" / "Drukuj" / "Pokaż QR" UI controls in the
        # cleaned long description.
        assert "facebook" not in ev["description"].lower()

    def test_image_url_extracted(self):
        ev = pe._parse_event_detail(self.html, self.url)
        # The detail page always carries a hero image — we should find it.
        assert ev["image_url"] is not None
        assert ev["image_url"].startswith("https://www.poznan.pl/")
        # Prefer the high-res variant (show2.jpg) over the thumbnail.
        assert "show2" in ev["image_url"] or "with-dims" in ev["image_url"]

    def test_detail_url_uses_canonical(self):
        # Detail pages declare their own canonical URL. We prefer it over the
        # caller-supplied URL so round-tripped URLs are stable.
        ev = pe._parse_event_detail(self.html, self.url)
        assert ev["detail_url"] == "https://www.poznan.pl/mim/events/-,179433.html"

    def test_detail_url_falls_back_to_source(self):
        # If the canonical link is absent, we keep the URL the caller used.
        html_no_canonical = self.html.replace(
            '<link rel="canonical" href="https://www.poznan.pl/mim/events/-,179433.html" />',
            "",
        )
        ev = pe._parse_event_detail(html_no_canonical, self.url)
        # Falls back through og:url first (also points at the dash-slug),
        # then to the source URL only if both meta tags are missing.
        assert ev["detail_url"].endswith(",179433.html")


# ---------------------------------------------------------------------------
# Live integration tests (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
class TestLiveEndpoint:
    async def test_list_events_first_page(self):
        async with __import__("aiohttp").ClientSession() as s:
            events = await pe._fetch_events_page(s, "", 0)
        assert len(events) > 0
        ev = events[0]
        assert ev["title"]
        assert ev["detail_url"].startswith("https://www.poznan.pl/")

    async def test_list_events_with_category(self):
        async with __import__("aiohttp").ClientSession() as s:
            events = await pe._fetch_events_page(s, "214", 0)
        # Sport category — every event should list "Sport" among categories.
        for ev in events:
            assert any(c["id"] == "214" for c in ev["categories"]), ev["title"]
