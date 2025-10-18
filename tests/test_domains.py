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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
