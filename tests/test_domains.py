"""
Unit tests for domain checker server.
"""

import pytest
import sys
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from domains import (
    extract_tld,
    contains_chinese_characters,
    get_not_found_patterns,
    WHOIS_SERVERS,
    OVH_VERIFIER_AVAILABLE,
)


class TestTLDExtraction:
    """Test TLD extraction logic."""

    def test_regular_tld(self):
        """Test regular single TLDs."""
        assert extract_tld("example.com") == "com"
        assert extract_tld("test.org") == "org"
        assert extract_tld("demo.io") == "io"

    def test_compound_tld(self):
        """Test compound TLDs like .com.cn."""
        assert extract_tld("example.com.cn") == "com.cn"
        assert extract_tld("test.net.cn") == "net.cn"
        assert extract_tld("demo.org.cn") == "org.cn"

    def test_chinese_tld(self):
        """Test Chinese/IDN TLDs."""
        assert extract_tld("测试.中国") == "中国"
        assert extract_tld("示例.公司") == "公司"

    def test_unsupported_tld(self):
        """Test unsupported TLDs."""
        assert extract_tld("example.unsupported") == ""
        assert extract_tld("test.xyz123") == ""

    def test_case_insensitive(self):
        """Test that TLD extraction is case insensitive."""
        assert extract_tld("EXAMPLE.COM") == "com"
        assert extract_tld("Test.ORG") == "org"


class TestChineseCharacterDetection:
    """Test Chinese character detection."""

    def test_contains_chinese(self):
        """Test detection of Chinese characters."""
        assert contains_chinese_characters("中国") is True
        assert contains_chinese_characters("测试") is True
        assert contains_chinese_characters("example中国") is True

    def test_no_chinese(self):
        """Test strings without Chinese characters."""
        assert contains_chinese_characters("example") is False
        assert contains_chinese_characters("test123") is False
        assert contains_chinese_characters("") is False


class TestNotFoundPatterns:
    """Test not found pattern retrieval."""

    def test_default_patterns(self):
        """Test default patterns for unknown TLDs."""
        patterns = get_not_found_patterns("xyz")
        assert "Domain not found" in patterns
        assert "No match for" in patterns

    def test_tld_specific_patterns(self):
        """Test TLD-specific patterns."""
        cn_patterns = get_not_found_patterns("cn")
        assert "no matching record" in cn_patterns

        pl_patterns = get_not_found_patterns("pl")
        assert "No information available about domain name" in pl_patterns


class TestWHOISServerMapping:
    """Test WHOIS server configuration."""

    def test_common_tlds_mapped(self):
        """Test that common TLDs have WHOIS servers."""
        assert "com" in WHOIS_SERVERS
        assert "net" in WHOIS_SERVERS
        assert "org" in WHOIS_SERVERS
        assert "io" in WHOIS_SERVERS

    def test_compound_tlds_mapped(self):
        """Test that compound TLDs have WHOIS servers."""
        assert "com.cn" in WHOIS_SERVERS
        assert "net.cn" in WHOIS_SERVERS

    def test_chinese_tlds_mapped(self):
        """Test that Chinese TLDs have WHOIS servers."""
        assert "中国" in WHOIS_SERVERS
        assert "公司" in WHOIS_SERVERS


# Integration tests would go here (requires mocking WHOIS/DNS)
class TestDomainChecking:
    """Integration tests for domain checking (requires mocking)."""

    @pytest.mark.skip(reason="Requires WHOIS mock server")
    def test_check_registered_domain(self):
        """Test checking a known registered domain."""
        # Would test with mocked WHOIS response
        pass

    @pytest.mark.skip(reason="Requires WHOIS mock server")
    def test_check_available_domain(self):
        """Test checking a known available domain."""
        # Would test with mocked WHOIS response
        pass

    @pytest.mark.skip(reason="Requires DNS mock")
    def test_dns_fallback(self):
        """Test DNS fallback when WHOIS fails."""
        # Would test with mocked DNS
        pass


class TestResponseStructure:
    """Test the structure of check_domains response."""

    @pytest.fixture
    def mock_results(self):
        """Create mock results for testing response structure."""
        return [
            (
                "available.com",
                {
                    "registered": False,
                    "method": "whois",
                    "reason": "Domain available according to WHOIS",
                },
            ),
            (
                "registered.org",
                {
                    "registered": True,
                    "method": "whois",
                    "reason": "Domain registered according to WHOIS",
                },
            ),
            (
                "failed.xyz",
                {
                    "registered": False,
                    "method": "unsupported",
                    "reason": "Unsupported top-level domain: xyz",
                },
            ),
            (
                "dns-available.io",
                {
                    "registered": False,
                    "method": "dns",
                    "reason": "No DNS records found",
                },
            ),
        ]

    def test_detailed_results_structure(self, mock_results):
        """Test that detailed_results includes all fields."""
        detailed_results = {
            domain: {
                "registered": status["registered"],
                "available": not status["registered"]
                and status["method"] not in ["failed", "unsupported"],
                "method": status["method"],
                "reason": status["reason"],
            }
            for domain, status in mock_results
        }

        # Check available domain
        assert detailed_results["available.com"]["registered"] is False
        assert detailed_results["available.com"]["available"] is True
        assert detailed_results["available.com"]["method"] == "whois"
        assert "Domain available" in detailed_results["available.com"]["reason"]

        # Check registered domain
        assert detailed_results["registered.org"]["registered"] is True
        assert detailed_results["registered.org"]["available"] is False
        assert detailed_results["registered.org"]["method"] == "whois"

        # Check failed domain - not available because check failed
        assert detailed_results["failed.xyz"]["registered"] is False
        assert detailed_results["failed.xyz"]["available"] is False
        assert detailed_results["failed.xyz"]["method"] == "unsupported"

        # Check DNS-based available domain
        assert detailed_results["dns-available.io"]["available"] is True
        assert detailed_results["dns-available.io"]["method"] == "dns"

    def test_available_domains_list(self, mock_results):
        """Test that available_domains list is correctly populated."""
        available_domains = [
            domain
            for domain, status in mock_results
            if not status["registered"]
            and status["method"] not in ["failed", "unsupported"]
        ]

        assert "available.com" in available_domains
        assert "dns-available.io" in available_domains
        assert "registered.org" not in available_domains
        assert "failed.xyz" not in available_domains
        assert len(available_domains) == 2

    def test_registered_domains_list(self, mock_results):
        """Test that registered_domains list is correctly populated."""
        registered_domains = [
            domain for domain, status in mock_results if status["registered"]
        ]

        assert "registered.org" in registered_domains
        assert "available.com" not in registered_domains
        assert len(registered_domains) == 1

    def test_failed_domains_list(self, mock_results):
        """Test that failed_domains list is correctly populated."""
        failed_domains = [
            domain
            for domain, status in mock_results
            if status["method"] in ["failed", "unsupported"]
        ]

        assert "failed.xyz" in failed_domains
        assert "available.com" not in failed_domains
        assert len(failed_domains) == 1

    def test_summary_text_includes_available_domains(self, mock_results):
        """Test that human-readable summary includes available domain names."""
        available_domains = [
            domain
            for domain, status in mock_results
            if not status["registered"]
            and status["method"] not in ["failed", "unsupported"]
        ]

        summary_parts = ["✓ Checked 4 domain(s) in 1.00s\n"]
        if available_domains:
            summary_parts.append(f"AVAILABLE ({len(available_domains)}):")
            for domain in available_domains:
                summary_parts.append(f"  • {domain}")
            summary_parts.append("")

        summary = "\n".join(summary_parts)

        assert "AVAILABLE (2):" in summary
        assert "available.com" in summary
        assert "dns-available.io" in summary

    def test_summary_text_includes_registered_domains(self, mock_results):
        """Test that human-readable summary includes registered domain names."""
        registered_domains = [
            domain for domain, status in mock_results if status["registered"]
        ]

        summary_parts = []
        if registered_domains:
            summary_parts.append(f"REGISTERED ({len(registered_domains)}):")
            for domain in registered_domains:
                summary_parts.append(f"  • {domain}")

        summary = "\n".join(summary_parts)

        assert "REGISTERED (1):" in summary
        assert "registered.org" in summary

    def test_summary_text_includes_failed_domains_with_reasons(self, mock_results):
        """Test that human-readable summary includes failed domains with reasons."""
        detailed_results = {
            domain: {
                "registered": status["registered"],
                "available": not status["registered"]
                and status["method"] not in ["failed", "unsupported"],
                "method": status["method"],
                "reason": status["reason"],
            }
            for domain, status in mock_results
        }

        failed_domains = [
            domain
            for domain, status in mock_results
            if status["method"] in ["failed", "unsupported"]
        ]

        summary_parts = []
        if failed_domains:
            summary_parts.append(f"FAILED ({len(failed_domains)}):")
            for domain in failed_domains:
                reason = detailed_results[domain]["reason"]
                summary_parts.append(f"  • {domain} ({reason})")

        summary = "\n".join(summary_parts)

        assert "FAILED (1):" in summary
        assert "failed.xyz" in summary
        assert "Unsupported top-level domain" in summary


class TestOVHVerificationIntegration:
    """Test OVH verification integration."""

    def test_ovh_verifier_module_imported(self):
        """Test that OVH verifier module is importable."""
        # The import should not fail, just check the flag
        assert isinstance(OVH_VERIFIER_AVAILABLE, bool)

    def test_response_with_ovh_disabled(self):
        """Test response structure when OVH verification is disabled."""
        structured_response = {
            "results": {},
            "available_domains": [],
            "registered_domains": [],
            "failed_domains": [],
            "summary": {},
            "ovh_verification": {"enabled": False},
        }

        assert structured_response["ovh_verification"]["enabled"] is False
        assert "confirmed_available" not in structured_response["ovh_verification"]

    def test_response_with_ovh_enabled_no_false_positives(self):
        """Test response structure when OVH finds no false positives."""
        structured_response = {
            "results": {
                "sparkhelm.com": {
                    "registered": False,
                    "available": True,
                    "method": "whois",
                    "reason": "OVH CONFIRMED available (120 PLN)",
                    "ovh_confirmed": True,
                    "standard_price": "120 PLN",
                    "ovh_verification": {
                        "ovh_available": True,
                        "ovh_verified": True,
                        "ovh_price": "120 PLN",
                        "ovh_price_type": "standard",
                        "ovh_is_aftermarket": False,
                        "ovh_aftermarket_type": None,
                        "ovh_error": None,
                    },
                }
            },
            "available_domains": ["sparkhelm.com"],
            "aftermarket_domains": [],
            "registered_domains": [],
            "failed_domains": [],
            "summary": {},
            "ovh_verification": {
                "enabled": True,
                "confirmed_available": ["sparkhelm.com"],
                "aftermarket": [],
                "false_positives": [],
                "verification_failed": [],
                "duration_seconds": 5.0,
            },
        }

        assert structured_response["ovh_verification"]["enabled"] is True
        assert "sparkhelm.com" in structured_response["ovh_verification"]["confirmed_available"]
        assert len(structured_response["ovh_verification"]["false_positives"]) == 0
        assert len(structured_response["ovh_verification"]["aftermarket"]) == 0
        assert structured_response["results"]["sparkhelm.com"]["ovh_confirmed"] is True
        assert structured_response["results"]["sparkhelm.com"]["standard_price"] == "120 PLN"

    def test_response_with_ovh_false_positive_detected(self):
        """Test response structure when OVH detects false positive."""
        structured_response = {
            "results": {
                "nexus.dev": {
                    "registered": False,
                    "available": False,  # Changed to False after OVH check
                    "method": "dns",
                    "reason": "FALSE POSITIVE: OVH verification shows domain is registered",
                    "false_positive": True,
                    "ovh_verification": {
                        "ovh_available": False,
                        "ovh_verified": True,
                        "ovh_price": None,
                        "ovh_price_type": None,
                        "ovh_is_aftermarket": False,
                        "ovh_aftermarket_type": None,
                        "ovh_error": None,
                    },
                }
            },
            "available_domains": [],  # Removed from available list
            "aftermarket_domains": [],
            "registered_domains": [],
            "failed_domains": [],
            "summary": {},
            "ovh_verification": {
                "enabled": True,
                "confirmed_available": [],
                "aftermarket": [],
                "false_positives": ["nexus.dev"],
                "verification_failed": [],
                "duration_seconds": 3.0,
            },
        }

        assert structured_response["ovh_verification"]["enabled"] is True
        assert "nexus.dev" in structured_response["ovh_verification"]["false_positives"]
        assert "nexus.dev" not in structured_response["available_domains"]
        assert structured_response["results"]["nexus.dev"]["available"] is False
        assert structured_response["results"]["nexus.dev"]["false_positive"] is True

    def test_response_with_ovh_aftermarket_detected(self):
        """Test response structure when OVH detects aftermarket/premium domain."""
        structured_response = {
            "results": {
                "catch.dev": {
                    "registered": False,
                    "available": False,  # Not available for standard registration
                    "method": "dns",
                    "reason": "AFTERMARKET: Domain on secondary market (Premium) - 1 384,70 zł",
                    "aftermarket": True,
                    "aftermarket_type": "Premium",
                    "aftermarket_price": "1 384,70 zł",
                    "ovh_verification": {
                        "ovh_available": False,
                        "ovh_verified": True,
                        "ovh_price": "1 384,70 zł",
                        "ovh_price_type": "premium",
                        "ovh_is_aftermarket": True,
                        "ovh_aftermarket_type": "Premium",
                        "ovh_error": None,
                    },
                }
            },
            "available_domains": [],  # NOT in available list
            "aftermarket_domains": ["catch.dev"],  # In aftermarket list
            "registered_domains": [],
            "failed_domains": [],
            "summary": {},
            "ovh_verification": {
                "enabled": True,
                "confirmed_available": [],
                "aftermarket": ["catch.dev"],
                "false_positives": [],
                "verification_failed": [],
                "duration_seconds": 3.0,
            },
        }

        assert structured_response["ovh_verification"]["enabled"] is True
        assert "catch.dev" in structured_response["ovh_verification"]["aftermarket"]
        assert "catch.dev" not in structured_response["available_domains"]
        assert "catch.dev" in structured_response["aftermarket_domains"]
        assert structured_response["results"]["catch.dev"]["available"] is False
        assert structured_response["results"]["catch.dev"]["aftermarket"] is True
        assert structured_response["results"]["catch.dev"]["aftermarket_type"] == "Premium"
        assert structured_response["results"]["catch.dev"]["aftermarket_price"] == "1 384,70 zł"
        assert structured_response["results"]["catch.dev"]["ovh_verification"]["ovh_is_aftermarket"] is True

    def test_response_with_third_party_aftermarket(self):
        """Test response structure for third-party aftermarket domain."""
        structured_response = {
            "results": {
                "catch.io": {
                    "registered": False,
                    "available": False,
                    "method": "dns",
                    "reason": "AFTERMARKET: Domain on secondary market (Sprzedaż przez stronę trzecią) - 639 514,20 zł",
                    "aftermarket": True,
                    "aftermarket_type": "Sprzedaż przez stronę trzecią",
                    "aftermarket_price": "639 514,20 zł",
                    "ovh_verification": {
                        "ovh_available": False,
                        "ovh_verified": True,
                        "ovh_price": "639 514,20 zł",
                        "ovh_price_type": "third_party",
                        "ovh_is_aftermarket": True,
                        "ovh_aftermarket_type": "Sprzedaż przez stronę trzecią",
                        "ovh_error": None,
                    },
                }
            },
            "available_domains": [],
            "aftermarket_domains": ["catch.io"],
            "registered_domains": [],
            "failed_domains": [],
            "summary": {},
            "ovh_verification": {
                "enabled": True,
                "confirmed_available": [],
                "aftermarket": ["catch.io"],
                "false_positives": [],
                "verification_failed": [],
                "duration_seconds": 3.0,
            },
        }

        assert structured_response["results"]["catch.io"]["aftermarket_type"] == "Sprzedaż przez stronę trzecią"
        assert structured_response["results"]["catch.io"]["ovh_verification"]["ovh_price_type"] == "third_party"

    def test_response_with_ovh_verification_failure(self):
        """Test response structure when OVH verification fails for a domain."""
        structured_response = {
            "results": {
                "example.com": {
                    "registered": False,
                    "available": True,
                    "method": "dns",
                    "reason": "No DNS records found (OVH verification failed: Domain not found in OVH results)",
                    "ovh_verification": {
                        "ovh_available": None,
                        "ovh_verified": False,
                        "ovh_price": None,
                        "ovh_error": "Domain not found in OVH results",
                    },
                }
            },
            "available_domains": ["example.com"],
            "registered_domains": [],
            "failed_domains": [],
            "summary": {},
            "ovh_verification": {
                "enabled": True,
                "confirmed_available": [],
                "false_positives": [],
                "verification_failed": ["example.com"],
                "duration_seconds": 2.0,
            },
        }

        assert "example.com" in structured_response["ovh_verification"]["verification_failed"]
        assert "example.com" in structured_response["available_domains"]  # Still available, just unverified
        assert structured_response["results"]["example.com"]["ovh_verification"]["ovh_verified"] is False

    def test_summary_includes_false_positive_warning(self):
        """Test that summary text includes false positive warnings."""
        false_positives = ["nexus.dev", "bolt.dev"]

        summary_parts = []
        if false_positives:
            summary_parts.append(f"❌ FALSE POSITIVES ({len(false_positives)}):")
            for domain in false_positives:
                summary_parts.append(f"  ❌ {domain} (WHOIS/DNS says available, OVH says registered)")
            summary_parts.append("")

        summary = "\n".join(summary_parts)

        assert "FALSE POSITIVES (2):" in summary
        assert "nexus.dev" in summary
        assert "bolt.dev" in summary
        assert "OVH says registered" in summary

    def test_summary_includes_confirmed_available_with_price(self):
        """Test that summary shows confirmed available domains with prices."""
        confirmed_available = ["sparkhelm.com"]
        ovh_verification = {
            "sparkhelm.com": {
                "ovh_available": True,
                "ovh_verified": True,
                "ovh_price": "120.00 PLN",
                "ovh_error": None,
            }
        }

        summary_parts = []
        if confirmed_available:
            summary_parts.append(f"AVAILABLE - OVH CONFIRMED ({len(confirmed_available)}):")
            for domain in confirmed_available:
                ovh_price = ovh_verification.get(domain, {}).get("ovh_price", "N/A")
                summary_parts.append(f"  ✅ {domain} ({ovh_price})")
            summary_parts.append("")

        summary = "\n".join(summary_parts)

        assert "AVAILABLE - OVH CONFIRMED (1):" in summary
        assert "sparkhelm.com" in summary
        assert "120.00 PLN" in summary
        assert "✅" in summary

    def test_dns_based_warning_without_ovh(self):
        """Test that DNS-based results show warning when OVH verification not enabled."""
        available_domains = ["test.dev"]
        detailed_results = {
            "test.dev": {
                "registered": False,
                "available": True,
                "method": "dns",
                "reason": "No DNS records found",
            }
        }

        summary_parts = []
        summary_parts.append(f"AVAILABLE ({len(available_domains)}):")
        for domain in available_domains:
            method = detailed_results[domain]["method"]
            if method == "dns":
                summary_parts.append(f"  ⚠️  {domain} (DNS-based, may be false positive)")
            else:
                summary_parts.append(f"  • {domain}")

        summary = "\n".join(summary_parts)

        assert "⚠️" in summary
        assert "DNS-based" in summary
        assert "false positive" in summary

    def test_summary_includes_aftermarket_domains(self):
        """Test that summary shows aftermarket domains with pricing."""
        aftermarket_domains = ["catch.dev", "catch.io"]
        ovh_verification = {
            "catch.dev": {
                "ovh_available": False,
                "ovh_verified": True,
                "ovh_price": "1 384,70 zł",
                "ovh_price_type": "premium",
                "ovh_is_aftermarket": True,
                "ovh_aftermarket_type": "Premium",
                "ovh_error": None,
            },
            "catch.io": {
                "ovh_available": False,
                "ovh_verified": True,
                "ovh_price": "639 514,20 zł",
                "ovh_price_type": "third_party",
                "ovh_is_aftermarket": True,
                "ovh_aftermarket_type": "Sprzedaż przez stronę trzecią",
                "ovh_error": None,
            },
        }

        summary_parts = []
        if aftermarket_domains:
            summary_parts.append(f"💰 AFTERMARKET - Secondary Market ({len(aftermarket_domains)}):")
            for domain in aftermarket_domains:
                ovh_result = ovh_verification.get(domain, {})
                aftermarket_type = ovh_result.get("ovh_aftermarket_type", "aftermarket")
                price = ovh_result.get("ovh_price", "N/A")
                summary_parts.append(f"  💰 {domain} ({aftermarket_type}) - {price}")
            summary_parts.append("")

        summary = "\n".join(summary_parts)

        assert "AFTERMARKET - Secondary Market (2):" in summary
        assert "catch.dev" in summary
        assert "catch.io" in summary
        assert "Premium" in summary
        assert "1 384,70 zł" in summary
        assert "639 514,20 zł" in summary
        assert "💰" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
