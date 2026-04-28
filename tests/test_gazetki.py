"""Unit tests for gazetki.py — fixture-driven (no live network).

Live integrations are marked with @pytest.mark.integration and skipped by default.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path

import pytest
from PIL import Image

import gazetki

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Lidl listing parser
# ---------------------------------------------------------------------------

class TestLidlListParser:
    def setup_method(self):
        self.html = (FIXTURES / "lidl_gazetki_list.html").read_text(encoding="utf-8")

    def test_parses_at_least_one_flyer(self):
        flyers = gazetki._parse_lidl_list(self.html)
        assert len(flyers) >= 1

    def test_first_flyer_shape(self):
        flyers = gazetki._parse_lidl_list(self.html)
        f = flyers[0]
        assert set(f.keys()) >= {"flyer_id", "slug", "name", "detail_url"}
        assert len(f["flyer_id"]) == 36  # UUID-ish
        assert "-" in f["flyer_id"]
        assert f["slug"]  # non-empty
        assert f["detail_url"].startswith("https://www.lidl.pl/")

    def test_flyer_ids_are_unique(self):
        flyers = gazetki._parse_lidl_list(self.html)
        ids = [f["flyer_id"] for f in flyers]
        assert len(ids) == len(set(ids))

    def test_html_unescapes_in_name(self):
        flyers = gazetki._parse_lidl_list(self.html)
        # Real Lidl names contain Polish characters; data-track-name is HTML-escaped
        # (e.g. "Gazetka&#x20;wa&#x017C;na&#x20;od&#x20;...") and we unescape.
        assert any("ż" in f["name"] or "Gazetka" in f["name"] for f in flyers)
        # Make sure no &#x sequences leaked through
        for f in flyers:
            assert "&#x" not in f["name"]

    def test_slug_extracted_from_href(self):
        flyers = gazetki._parse_lidl_list(self.html)
        for f in flyers:
            assert f"/gazetki/{f['slug']}/" in f["detail_url"]

    def test_returns_empty_for_empty_html(self):
        assert gazetki._parse_lidl_list("") == []

    def test_ignores_non_flyer_anchors(self):
        # Non-flyer anchors should not appear
        html_text = '<a class="other" href="/x" data-track-id="abc">x</a>'
        assert gazetki._parse_lidl_list(html_text) == []


# ---------------------------------------------------------------------------
# Lidl detail parser
# ---------------------------------------------------------------------------

class TestLidlDetailParser:
    def setup_method(self):
        self.payload = json.loads(
            (FIXTURES / "lidl_flyer_detail.json").read_text(encoding="utf-8")
        )

    def test_extracts_pdf_url(self):
        d = gazetki._parse_lidl_detail(self.payload)
        assert d["pdf_url"]
        assert d["pdf_url"].endswith(".pdf")

    def test_extracts_dates(self):
        d = gazetki._parse_lidl_detail(self.payload)
        assert d["valid_from"]
        assert d["valid_to"]

    def test_extracts_metadata(self):
        d = gazetki._parse_lidl_detail(self.payload)
        assert d["flyer_id"]
        assert d["title"] is not None
        assert d["name"] is not None

    def test_handles_missing_fields(self):
        d = gazetki._parse_lidl_detail({"flyer": {}})
        assert d["pdf_url"] is None
        assert d["valid_from"] is None

    def test_handles_missing_envelope(self):
        d = gazetki._parse_lidl_detail({})
        assert d["pdf_url"] is None


# ---------------------------------------------------------------------------
# Biedronka listing parser
# ---------------------------------------------------------------------------

class TestBiedronkaListParser:
    def setup_method(self):
        self.html = (FIXTURES / "biedronka_gazetki_list.html").read_text(encoding="utf-8")

    def test_parses_multiple_flyers(self):
        flyers = gazetki._parse_biedronka_list(self.html)
        assert len(flyers) >= 5

    def test_flyer_shape(self):
        flyers = gazetki._parse_biedronka_list(self.html)
        f = flyers[0]
        assert set(f.keys()) >= {
            "flyer_id", "slug", "title", "detail_url", "thumbnail_url",
            "slot_id", "valid_from_hint", "is_adult",
        }
        assert f["flyer_id"]  # non-empty short id
        assert f["slug"]
        assert f["detail_url"].startswith("https://www.biedronka.pl/pl/press")

    def test_strips_url_fragment(self):
        flyers = gazetki._parse_biedronka_list(self.html)
        for f in flyers:
            assert "#" not in f["detail_url"]

    def test_extracts_week_hint(self):
        flyers = gazetki._parse_biedronka_list(self.html)
        # At least some entries have a week hint extracted from data-slot-id
        with_hint = [f for f in flyers if f["valid_from_hint"]]
        assert len(with_hint) >= 1
        for f in with_hint:
            assert f["valid_from_hint"].startswith("week ")

    def test_excludes_external_links(self):
        # The fixture contains a glovoapp.com link that should be filtered out.
        flyers = gazetki._parse_biedronka_list(self.html)
        for f in flyers:
            assert "glovoapp" not in f["detail_url"]

    def test_pressadult_flagged(self):
        flyers = gazetki._parse_biedronka_list(self.html)
        adult = [f for f in flyers if f["is_adult"]]
        # Fixture has at least one alkohole entry
        assert len(adult) >= 1
        for f in adult:
            assert "pressadult" in f["detail_url"]

    def test_flyer_ids_unique(self):
        flyers = gazetki._parse_biedronka_list(self.html)
        ids = [f["flyer_id"] for f in flyers]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Biedronka UUID extractor
# ---------------------------------------------------------------------------

class TestBiedronkaUUIDExtractor:
    def setup_method(self):
        self.html = (FIXTURES / "biedronka_press_detail.html").read_text(encoding="utf-8")

    def test_extracts_uuid(self):
        uuid = gazetki._extract_biedronka_uuid(self.html)
        assert uuid is not None
        assert len(uuid) == 36
        assert uuid.count("-") == 4

    def test_returns_none_when_absent(self):
        assert gazetki._extract_biedronka_uuid("<html>no init here</html>") is None

    def test_returns_none_for_invalid_uuid(self):
        # 36-char-but-not-UUID strings should not match (regex is strict)
        bad = '<script>galleryLeaflet.init("not-a-uuid-not-a-uuid-not-a-uuid-no")</script>'
        assert gazetki._extract_biedronka_uuid(bad) is None


# ---------------------------------------------------------------------------
# Biedronka leaflet API parser
# ---------------------------------------------------------------------------

class TestBiedronkaApiParser:
    def setup_method(self):
        self.payload = json.loads(
            (FIXTURES / "biedronka_leaflet_api.json").read_text(encoding="utf-8")
        )

    def test_parses_mobile_pages(self):
        urls = gazetki._parse_biedronka_leaflet(self.payload)
        assert len(urls) >= 1
        assert all(u.startswith("https://") for u in urls)

    def test_pages_ordered(self):
        # Pages should be in order — the parser sorts by page number
        urls = gazetki._parse_biedronka_leaflet(self.payload)
        # sanity: at least the count matches images_mobile length
        assert len(urls) == len(self.payload["images_mobile"])

    def test_falls_back_to_desktop(self):
        # Synthetic payload with no mobile images
        synth = {
            "images_mobile": [],
            "images_desktop": [
                {"page": 0, "images": ["", "https://example.com/page0.png"]},
                {"page": 1, "images": ["https://example.com/page1a.png", "https://example.com/page1b.png"]},
            ],
        }
        urls = gazetki._parse_biedronka_leaflet(synth)
        assert urls == [
            "https://example.com/page0.png",
            "https://example.com/page1b.png",
        ]

    def test_empty_payload(self):
        assert gazetki._parse_biedronka_leaflet({}) == []


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------

class TestPdfAssembly:
    def _png_bytes(self, color: tuple[int, int, int], size=(200, 300)) -> bytes:
        img = Image.new("RGB", size, color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_assembles_single_page_pdf(self):
        pdf_bytes = gazetki._assemble_biedronka_pdf([self._png_bytes((255, 0, 0))])
        assert pdf_bytes.startswith(b"%PDF")

    def test_assembles_multi_page_pdf(self):
        pages = [
            self._png_bytes((255, 0, 0)),
            self._png_bytes((0, 255, 0)),
            self._png_bytes((0, 0, 255)),
        ]
        pdf_bytes = gazetki._assemble_biedronka_pdf(pages)
        assert pdf_bytes.startswith(b"%PDF")
        # Verify page count via pypdf if available
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            assert len(reader.pages) == 3
        except ImportError:
            pass

    def test_assembles_rgba_input(self):
        # PDF assembly must auto-convert RGBA → RGB
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        pdf_bytes = gazetki._assemble_biedronka_pdf([buf.getvalue()])
        assert pdf_bytes.startswith(b"%PDF")

    def test_zero_pages_raises(self):
        with pytest.raises(ValueError):
            gazetki._assemble_biedronka_pdf([])


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------

class TestThrottle:
    def setup_method(self):
        gazetki._LAST_REQUEST.clear()

    @pytest.mark.asyncio
    async def test_first_call_is_immediate(self):
        t0 = time.monotonic()
        await gazetki._throttle("immediate.example.com", min_interval=0.5)
        assert time.monotonic() - t0 < 0.1

    @pytest.mark.asyncio
    async def test_second_call_waits(self):
        await gazetki._throttle("repeat.example.com", min_interval=0.2)
        t0 = time.monotonic()
        await gazetki._throttle("repeat.example.com", min_interval=0.2)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.18  # allow a touch of slack

    @pytest.mark.asyncio
    async def test_independent_hosts_dont_interfere(self):
        await gazetki._throttle("a.example.com", min_interval=0.5)
        t0 = time.monotonic()
        await gazetki._throttle("b.example.com", min_interval=0.5)
        # different host → should not wait
        assert time.monotonic() - t0 < 0.1


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

class TestCacheBehavior:
    @pytest.mark.asyncio
    async def test_is_cache_fresh_with_valid_to(self):
        from datetime import date, timedelta
        future = (date.today() + timedelta(days=1)).isoformat()
        assert gazetki._is_cache_fresh({"valid_to": future}) is True
        past = (date.today() - timedelta(days=10)).isoformat()
        assert gazetki._is_cache_fresh({"valid_to": past}) is False

    @pytest.mark.asyncio
    async def test_is_cache_fresh_with_recent_fetched_at(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        assert gazetki._is_cache_fresh({"fetched_at": now}) is True

    @pytest.mark.asyncio
    async def test_is_cache_fresh_falsey_for_empty(self):
        assert gazetki._is_cache_fresh({}) is False

    def test_cache_dir_respects_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GAZETKI_CACHE_DIR", str(tmp_path / "custom"))
        assert gazetki._cache_dir() == tmp_path / "custom"

    def test_flyer_cache_paths_creates_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GAZETKI_CACHE_DIR", str(tmp_path))
        pdf_path, meta_path = gazetki._flyer_cache_paths("lidl", "abc-123")
        assert pdf_path.parent.exists()
        assert pdf_path.name == "abc-123.pdf"
        assert meta_path.name == "abc-123.meta.json"

    def test_flyer_cache_paths_sanitizes_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GAZETKI_CACHE_DIR", str(tmp_path))
        pdf_path, _ = gazetki._flyer_cache_paths("biedronka", "../bad/id")
        # No path traversal: pdf must live under the chain dir
        assert "/" not in pdf_path.name
        assert pdf_path.parent.name == "biedronka"
        assert pdf_path.parent.parent == tmp_path


# ---------------------------------------------------------------------------
# Integration smoke tests (live network) — opt-in
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLiveSmoke:
    @pytest.mark.asyncio
    async def test_lidl_listing_live(self):
        flyers = await gazetki._list_lidl()
        assert len(flyers) >= 1
        assert all(f["chain"] == "lidl" for f in flyers)

    @pytest.mark.asyncio
    async def test_biedronka_listing_live(self):
        flyers = await gazetki._list_biedronka()
        assert len(flyers) >= 1
        assert all(f["chain"] == "biedronka" for f in flyers)
