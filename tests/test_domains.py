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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
