"""Unit tests for krs.py — no live network.

Live integration tests are marked with @pytest.mark.integration and skipped by
default (they hit api-krs.ms.gov.pl).
"""

from __future__ import annotations

import pytest
from aioresponses import aioresponses

import krs


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_krs(self):
        assert krs._validate_krs("0001236495") == "0001236495"

    @pytest.mark.parametrize("bad", [
        "1234567890",   # doesn't start with 0
        "001236495",    # 9 digits
        "00012364950",  # 11 digits
        "000123649A",   # contains letter
        "",
    ])
    def test_rejects_bad_krs(self, bad):
        with pytest.raises(ValueError):
            krs._validate_krs(bad)

    @pytest.mark.parametrize("inp,expected", [("p", "P"), ("P", "P"), ("s", "S"), ("S", "S")])
    def test_normalize_rejestr(self, inp, expected):
        assert krs._normalize_rejestr(inp) == expected

    @pytest.mark.parametrize("bad", ["X", "PS", "", "1"])
    def test_rejects_bad_rejestr(self, bad):
        with pytest.raises(ValueError):
            krs._normalize_rejestr(bad)


# ---------------------------------------------------------------------------
# URL building
# ---------------------------------------------------------------------------

class TestBuildExtractUrls:
    def test_query_param_form_first(self):
        urls = krs._build_extract_urls("OdpisAktualny", "P", "0000635012")
        assert urls[0] == (
            "https://api-krs.ms.gov.pl/api/krs/OdpisAktualny/0000635012"
            "?rejestr=P&format=json"
        )

    def test_path_form_fallback(self):
        urls = krs._build_extract_urls("OdpisPelny", "S", "0001236495")
        assert urls[1] == (
            "https://api-krs.ms.gov.pl/api/krs/OdpisPelny/S/0001236495?format=json"
        )

    def test_two_urls_returned(self):
        urls = krs._build_extract_urls("OdpisAktualny", "P", "0000635012")
        assert len(urls) == 2


# ---------------------------------------------------------------------------
# Headline formatting
# ---------------------------------------------------------------------------

class TestFormatHeadline:
    def test_full_data(self):
        extract = {
            "odpis": {
                "naglowekA": {"numerKRS": "0000635012"},
                "dane": {
                    "dzial1": {
                        "danePodmiotu": {
                            "nazwa": "ALLEGRO SP. Z O.O.",
                            "formaPrawna": "SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ",
                            "identyfikatory": {"nip": "5252674798", "regon": "36533155300000"},
                        }
                    }
                },
            }
        }
        headline = krs._format_headline(extract)
        assert "ALLEGRO" in headline
        assert "0000635012" in headline
        assert "NIP 5252674798" in headline
        assert "REGON 36533155300000" in headline
        assert "SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ" in headline

    def test_missing_fields_falls_back(self):
        headline = krs._format_headline({})
        assert "Nieznana nazwa" in headline
        assert "KRS ?" in headline

    def test_missing_optional_extras(self):
        extract = {
            "odpis": {
                "naglowekA": {"numerKRS": "0001236495"},
                "dane": {"dzial1": {"danePodmiotu": {"nazwa": "FIRMA TESTOWA"}}},
            }
        }
        # No NIP/REGON brackets when identifiers absent
        headline = krs._format_headline(extract)
        assert "[" not in headline
        assert "FIRMA TESTOWA" in headline


# ---------------------------------------------------------------------------
# HTTP fetch (mocked)
# ---------------------------------------------------------------------------

class TestFetchExtract:
    @pytest.mark.asyncio
    async def test_returns_200_body(self):
        sample = {
            "odpis": {
                "naglowekA": {"numerKRS": "0001236495"},
                "dane": {"dzial1": {"danePodmiotu": {"nazwa": "PATRON DEVELOPMENT"}}},
            }
        }
        url = krs._build_extract_urls("OdpisAktualny", "P", "0001236495")[0]
        with aioresponses() as m:
            m.get(url, status=200, payload=sample)
            status, body = await krs._fetch_extract("OdpisAktualny", "P", "0001236495")
        assert status == 200
        assert body == sample

    @pytest.mark.asyncio
    async def test_204_falls_through_to_path_form_then_returns_204(self):
        urls = krs._build_extract_urls("OdpisAktualny", "P", "0001236495")
        with aioresponses() as m:
            m.get(urls[0], status=204)
            m.get(urls[1], status=204)
            status, body = await krs._fetch_extract("OdpisAktualny", "P", "0001236495")
        assert status == 204
        assert body is None

    @pytest.mark.asyncio
    async def test_404_returns_404(self):
        urls = krs._build_extract_urls("OdpisAktualny", "P", "0001236495")
        with aioresponses() as m:
            m.get(urls[0], status=404)
            m.get(urls[1], status=404)
            status, body = await krs._fetch_extract("OdpisAktualny", "P", "0001236495")
        assert status == 404
        assert body is None

    @pytest.mark.asyncio
    async def test_path_form_used_when_query_form_404s(self):
        sample = {"odpis": {"naglowekA": {"numerKRS": "0001236495"}, "dane": {}}}
        urls = krs._build_extract_urls("OdpisAktualny", "P", "0001236495")
        with aioresponses() as m:
            m.get(urls[0], status=404)
            m.get(urls[1], status=200, payload=sample)
            status, body = await krs._fetch_extract("OdpisAktualny", "P", "0001236495")
        assert status == 200
        assert body == sample


# ---------------------------------------------------------------------------
# Empty-state messages
# ---------------------------------------------------------------------------

class TestEmptyMessage:
    def test_204_hints_other_register(self):
        msg = krs._empty_message("odpis aktualny", "0001236495", "P", 204)
        assert "rejestru S" in msg

    def test_204_for_S_hints_P(self):
        msg = krs._empty_message("odpis aktualny", "0001236495", "S", 204)
        assert "rejestru P" in msg

    def test_404_says_not_found(self):
        msg = krs._empty_message("odpis aktualny", "0001236495", "P", 404)
        assert "Nie znaleziono" in msg

    def test_other_status_includes_status_code(self):
        msg = krs._empty_message("odpis aktualny", "0001236495", "P", 500)
        assert "500" in msg


# ---------------------------------------------------------------------------
# Live integration (skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLiveAPI:
    @pytest.mark.asyncio
    async def test_known_krs_returns_data(self):
        # Allegro sp. z o.o.
        status, body = await krs._fetch_extract("OdpisAktualny", "P", "0000635012")
        assert status == 200
        assert isinstance(body, dict)
        assert "odpis" in body
        nazwa = body["odpis"]["dane"]["dzial1"]["danePodmiotu"]["nazwa"]
        assert "ALLEGRO" in nazwa.upper()
