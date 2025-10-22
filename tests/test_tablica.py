"""
Tests for the Tablica Rejestracyjna PL MCP server.
"""

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from tablica import (
    validate_polish_plate,
    optimize_image,
    parse_comments_html,
    HEIC_SUPPORT,
)


class TestPlateValidation:
    """Tests for Polish license plate validation."""

    def test_valid_plates(self):
        """Test various valid Polish plate formats."""
        valid_plates = [
            "WW 12345",
            "WW12345",
            "KR1234",
            "KR 1234",
            "DW AB123",
            "DWAB123",
            "PO5JR15",
            "TST65412",
            "WA 12AB",
            "GD12345",
        ]
        for plate in valid_plates:
            assert validate_polish_plate(plate), f"Plate {plate} should be valid"

    def test_invalid_plates(self):
        """Test invalid plate formats."""
        invalid_plates = [
            "123",           # Too short
            "ABCDEFGHI",     # Too long
            "12345",         # Only numbers
            "ABCDE",         # Only letters
            "",              # Empty
            "W",             # Single character
            "1234567890",    # Too many numbers
        ]
        for plate in invalid_plates:
            assert not validate_polish_plate(plate), f"Plate {plate} should be invalid"

    def test_plate_normalization(self):
        """Test that plates with different spacing are still valid."""
        plates = [
            ("WW 12345", "WW12345"),
            ("WW-12345", "WW12345"),
            ("ww 12345", "WW12345"),
            ("Kr 1234", "KR1234"),
        ]
        for plate, _ in plates:
            assert validate_polish_plate(plate)


class TestImageOptimization:
    """Tests for image optimization and downscaling."""

    def test_optimize_small_image(self):
        """Test that small images are not resized."""
        # Create a small test image
        img = Image.new('RGB', (800, 600), color='red')

        image_bytes, mime_type = optimize_image(img, max_dimension=1920)

        assert mime_type == 'image/jpeg'
        assert len(image_bytes) > 0

        # Verify image is still valid
        result_img = Image.open(io.BytesIO(image_bytes))
        assert result_img.size[0] == 800
        assert result_img.size[1] == 600

    def test_optimize_large_image(self):
        """Test that large images are downscaled."""
        # Create a large test image
        img = Image.new('RGB', (4000, 3000), color='blue')

        image_bytes, mime_type = optimize_image(img, max_dimension=1920)

        assert mime_type == 'image/jpeg'
        assert len(image_bytes) > 0

        # Verify image was resized
        result_img = Image.open(io.BytesIO(image_bytes))
        assert result_img.size[0] <= 1920
        assert result_img.size[1] <= 1920

        # Verify aspect ratio is maintained
        original_ratio = 4000 / 3000
        result_ratio = result_img.size[0] / result_img.size[1]
        assert abs(original_ratio - result_ratio) < 0.01

    def test_optimize_rgba_image(self):
        """Test that RGBA images are converted to RGB."""
        # Create an RGBA image with transparency
        img = Image.new('RGBA', (800, 600), color=(255, 0, 0, 128))

        image_bytes, mime_type = optimize_image(img)

        assert mime_type == 'image/jpeg'

        # Verify conversion to RGB
        result_img = Image.open(io.BytesIO(image_bytes))
        assert result_img.mode == 'RGB'

    def test_optimize_maintains_quality(self):
        """Test that optimization maintains reasonable quality."""
        # Create a detailed image
        img = Image.new('RGB', (1000, 800), color='green')

        image_bytes_high = optimize_image(img, quality=95)[0]
        image_bytes_low = optimize_image(img, quality=50)[0]

        # Higher quality should produce larger files
        assert len(image_bytes_high) > len(image_bytes_low)


class TestHTMLParsing:
    """Tests for HTML comment parsing."""

    def test_parse_empty_html(self):
        """Test parsing HTML with no comments."""
        html = "<html><body><div>No comments here</div></body></html>"
        comments = parse_comments_html(html)

        # Should handle gracefully, might return empty or extract some text
        assert isinstance(comments, list)

    def test_parse_comments_with_structure(self):
        """Test parsing HTML with structured comments."""
        html = """
        <html>
        <body>
            <div class="komentarz">
                <p>Samochód zaparkowany na chodniku</p>
                <span class="user">Jan Kowalski</span>
                <time>2025-10-20</time>
                <div class="votes">
                    <span class="plus">5</span>
                    <span class="minus">1</span>
                </div>
            </div>
            <div class="komentarz">
                <p>Blokuje przejście dla pieszych</p>
                <span class="user">Anna Nowak</span>
                <time>2025-10-21</time>
            </div>
        </body>
        </html>
        """

        comments = parse_comments_html(html)

        assert len(comments) >= 1

        # Check first comment has expected structure
        if len(comments) > 0:
            first_comment = comments[0]
            assert 'text' in first_comment
            assert 'chodniku' in first_comment['text'].lower() or 'przejście' in first_comment['text'].lower()

    def test_parse_comments_with_images(self):
        """Test parsing comments that include images."""
        html = """
        <html>
        <body>
            <article>
                <p>Naruszenie przepisów</p>
                <img src="/uploads/photo123.jpg" alt="Evidence">
            </article>
        </body>
        </html>
        """

        comments = parse_comments_html(html)

        assert isinstance(comments, list)

        # Check if image was captured
        if len(comments) > 0 and 'image' in comments[0]:
            assert 'photo123.jpg' in comments[0]['image']

    def test_parse_malformed_html(self):
        """Test that malformed HTML doesn't crash parser."""
        html = "<html><body><div class='komentarz'><p>Unclosed paragraph</body>"

        # Should not raise exception
        comments = parse_comments_html(html)
        assert isinstance(comments, list)

    def test_parse_real_tablica_html(self):
        """Test parsing actual HTML structure from tablica-rejestracyjna.pl."""
        html = """
        <html><body>
        <div id='c1855132' comment-id='1855132' itemprop='comment' class='comment'>
            <span class='plate'><a href='/PO5JR15'>PO 5JR15</a></span>
            <span style='float: right; margin-top: -10px; font-size: x-small; font-weight: bold;'>
            <span class='name name-not-verified' itemprop='author'>ernest</span>
            <span class='date' itemprop='dateCreated'>2025-10-22 11:08:08</span>
            </span>
            <br/><br/>
            <span class='text' itemprop='text'>parkowanie na chodniku
            <a href='//tablica-rejestracyjna.pl/images/photos/20251022110806.jpg'>
            <img src="//tablica-rejestracyjna.pl/images/komentarze/253295_min.jpg"/>
            </a></span>
            <div class="commentControls">
                <meta itemprop="upvoteCount" content="0"/>
                <meta itemprop="downvoteCount" content="0"/>
                <div style="float: right">
                    <a title="Oceń komentarz" href="#"><div class="plusMinus plus">+</div></a>
                    <div class="plusMinus voteCount">0</div>
                    <a title="Oceń komentarz" href="#"><div class="plusMinus minus">-</div></a>
                </div>
            </div>
        </div>
        </body></html>
        """

        comments = parse_comments_html(html)

        assert len(comments) == 1
        comment = comments[0]

        # Check text is extracted
        assert 'text' in comment
        assert 'parkowanie na chodniku' in comment['text']

        # Check user is extracted
        assert 'user' in comment
        assert comment['user'] == 'ernest'

        # Check timestamp is extracted
        assert 'timestamp' in comment
        assert '2025-10-22 11:08:08' in comment['timestamp']

        # Check image is extracted
        assert 'image' in comment
        assert '253295_min.jpg' in comment['image']

        # Check rating is extracted
        assert 'rating' in comment
        assert comment['rating']['score'] == 0


class TestHEICSupport:
    """Tests for HEIC image support."""

    def test_heic_support_flag(self):
        """Test that HEIC support flag is set correctly."""
        assert isinstance(HEIC_SUPPORT, bool)

        # If pillow-heif is installed, it should be True
        try:
            import pillow_heif
            assert HEIC_SUPPORT is True
        except ImportError:
            # It's OK if not installed for testing
            pass


class TestImageFormats:
    """Tests for various image format handling."""

    def test_png_format(self):
        """Test PNG image optimization."""
        img = Image.new('RGB', (800, 600), color='yellow')

        # Save as PNG first
        png_buffer = io.BytesIO()
        img.save(png_buffer, format='PNG')
        png_img = Image.open(io.BytesIO(png_buffer.getvalue()))

        # Optimize (should convert to JPEG)
        image_bytes, mime_type = optimize_image(png_img)

        assert mime_type == 'image/jpeg'
        assert len(image_bytes) > 0

    def test_grayscale_image(self):
        """Test grayscale image handling."""
        img = Image.new('L', (800, 600), color=128)

        image_bytes, mime_type = optimize_image(img)

        assert mime_type == 'image/jpeg'

        # Verify conversion to RGB
        result_img = Image.open(io.BytesIO(image_bytes))
        assert result_img.mode == 'RGB'


# Integration tests (with mocking)

class MockContext:
    """Mock FastMCP context for testing."""

    def __init__(self):
        self.logs = []

    async def info(self, msg):
        self.logs.append(('info', msg))

    async def warning(self, msg):
        self.logs.append(('warning', msg))

    async def error(self, msg):
        self.logs.append(('error', msg))


@pytest.mark.integration
class TestFetchComments:
    """Integration tests for fetching comments with aioresponses mocking."""

    @pytest.mark.asyncio
    async def test_fetch_with_403_response(self):
        """Test that fetch handles 403 Forbidden gracefully."""
        from aioresponses import aioresponses
        from tablica import fetch_comments

        ctx = MockContext()

        with aioresponses() as m:
            url = "https://tablica-rejestracyjna.pl/PY49772"
            m.get(url, status=403, body='<html>Access Denied</html>')

            fetch_fn = fetch_comments.fn if hasattr(fetch_comments, 'fn') else fetch_comments
            result = await fetch_fn(plate_number="PY49772", ctx=ctx)

            assert result.structured_content.get('error') is not None

    @pytest.mark.asyncio
    async def test_fetch_nonexistent_plate(self):
        """Test fetching comments for non-existent plate."""
        from aioresponses import aioresponses
        from tablica import fetch_comments

        ctx = MockContext()

        with aioresponses() as m:
            url = "https://tablica-rejestracyjna.pl/PY49772"
            m.get(url, status=404)

            fetch_fn = fetch_comments.fn if hasattr(fetch_comments, 'fn') else fetch_comments
            result = await fetch_fn(plate_number="PY49772", ctx=ctx)

            assert result.structured_content['found'] is False


@pytest.mark.integration
class TestSubmitComplaint:
    """Integration tests for submitting complaints with aioresponses mocking."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Automated POST submission disabled to avoid IP bans - function now returns Playwright instructions")
    async def test_submit_establishes_session_before_post(self):
        """Test that submit makes GET request before POST."""
        from aioresponses import aioresponses
        from tablica import submit_complaint

        ctx = MockContext()

        # Create a test image
        test_img = Image.new('RGB', (100, 100), color='red')
        test_img.save('/tmp/test_complaint.jpg')

        with aioresponses() as m:
            # Mock GET and POST requests
            url = "https://tablica-rejestracyjna.pl/WW12345"
            m.get(url, status=200, body='<html><body></body></html>')
            m.post(url, status=200, body='<html><body>Dodano komentarz</body></html>')

            submit_fn = submit_complaint.fn if hasattr(submit_complaint, 'fn') else submit_complaint
            result = await submit_fn(
                plate_number="WW12345",
                violation_description="Test violation description in Polish",
                image_path="/tmp/test_complaint.jpg",
                ctx=ctx
            )

            # Verify both GET and POST were called (2 requests total)
            assert len(m.requests) == 2
            # Verify success
            assert result.structured_content['success'] is True

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Automated POST submission disabled to avoid IP bans - function now returns Playwright instructions")
    async def test_submit_extracts_csrf_token(self):
        """Test that submit detects and uses CSRF tokens."""
        from aioresponses import aioresponses
        from tablica import submit_complaint

        ctx = MockContext()

        # Mock GET response with CSRF token
        html_with_csrf = '''<html><body>
            <form>
                <input type="hidden" name="csrf_token" value="test_csrf_token_12345">
            </form>
        </body></html>'''

        # Create a test image
        test_img = Image.new('RGB', (100, 100), color='blue')
        test_img.save('/tmp/test_csrf.jpg')

        with aioresponses() as m:
            url = "https://tablica-rejestracyjna.pl/KR1234"
            m.get(url, status=200, body=html_with_csrf)
            m.post(url, status=200, body='<html>Sukces</html>')

            submit_fn = submit_complaint.fn if hasattr(submit_complaint, 'fn') else submit_complaint
            result = await submit_fn(
                plate_number="KR1234",
                violation_description="Another test violation",
                image_path="/tmp/test_csrf.jpg",
                ctx=ctx
            )

            # Check that CSRF token was detected
            assert result.structured_content.get('csrf_token_found') is True

            # Verify info log mentions CSRF token
            csrf_log = [log for log in ctx.logs if 'CSRF token' in log[1]]
            assert len(csrf_log) > 0

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Automated POST submission disabled to avoid IP bans - function now returns Playwright instructions")
    async def test_submit_with_403_provides_debug_info(self):
        """Test that 403 errors provide comprehensive debug information."""
        from aioresponses import aioresponses
        from tablica import submit_complaint

        ctx = MockContext()

        # Create a test image
        test_img = Image.new('RGB', (100, 100), color='green')
        test_img.save('/tmp/test_403.jpg')

        with aioresponses() as m:
            url = "https://tablica-rejestracyjna.pl/GD5678"
            # Mock 403 responses for both GET and POST
            m.get(url, status=403, body='<html>Access Denied</html>')
            m.post(url, status=403, body='<html>Forbidden</html>', headers={'Content-Type': 'text/html'})

            submit_fn = submit_complaint.fn if hasattr(submit_complaint, 'fn') else submit_complaint
            result = await submit_fn(
                plate_number="GD5678",
                violation_description="Test for 403 handling",
                image_path="/tmp/test_403.jpg",
                ctx=ctx
            )

            # Verify enhanced error reporting
            structured = result.structured_content
            assert structured['success'] is False
            assert structured['status_code'] == 403
            assert 'cookies_received' in structured
            assert 'response_html_preview' in structured
            assert 'response_headers' in structured

            # Verify user-facing error message is helpful
            assert 'bot protection' in structured['error'].lower()

            # Check text content has HTTP status
            text_content = result.content[0].text
            assert '403' in text_content
            assert 'HTTP Status:' in text_content

    @pytest.mark.asyncio
    async def test_submit_with_heic_image(self):
        """Test submission with HEIC image (if supported)."""
        if not HEIC_SUPPORT:
            pytest.skip("HEIC support not available")

        from aioresponses import aioresponses
        from tablica import submit_complaint

        ctx = MockContext()

        # Use real HEIC fixture
        heic_path = Path(__file__).parent / "fixtures" / "IMG_5134.heic"
        if not heic_path.exists():
            pytest.skip(f"HEIC fixture not found: {heic_path}")

        with aioresponses() as m:
            url = "https://tablica-rejestracyjna.pl/PY49772"
            m.get(url, status=200, body='<html></html>')
            m.post(url, status=200, body='<html>Dodano</html>')

            submit_fn = submit_complaint.fn if hasattr(submit_complaint, 'fn') else submit_complaint
            result = await submit_fn(
                plate_number="PY49772",
                violation_description="Parkowanie na chodniku",
                image_path=str(heic_path),
                ctx=ctx
            )

            # Check that HEIC conversion was logged
            heic_logs = [log for log in ctx.logs if 'HEIC' in log[1] or 'Converting' in log[1]]
            assert len(heic_logs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])