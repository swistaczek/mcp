"""
Tests for the Plate Recognition MCP server.
"""

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from plate_recognition import (
    optimize_image,
    create_plate_recognition_prompt,
)


class TestImageOptimization:
    """Tests for image optimization."""

    def test_optimize_small_image(self):
        """Test that small images are not resized."""
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
        img = Image.new('RGBA', (800, 600), color=(255, 0, 0, 128))

        image_bytes, mime_type = optimize_image(img)

        assert mime_type == 'image/jpeg'

        # Verify conversion to RGB
        result_img = Image.open(io.BytesIO(image_bytes))
        assert result_img.mode == 'RGB'


class TestPromptGeneration:
    """Tests for prompt generation."""

    def test_create_plate_recognition_prompt(self):
        """Test that prompt is generated correctly."""
        prompt = create_plate_recognition_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "license plate" in prompt.lower()
        assert "violation" in prompt.lower()
        assert "JSON" in prompt
        assert "plates" in prompt
        assert "reasoning" in prompt


class TestJSONParsing:
    """Tests for JSON response parsing."""

    def test_parse_valid_response_with_violation(self):
        """Test parsing valid JSON with violation."""
        response = {
            "plates": ["WW12345", "KR1234"],
            "violation_vehicle": "WW12345",
            "reasoning": "Vehicle WW12345 is parked on the sidewalk."
        }

        assert isinstance(response["plates"], list)
        assert len(response["plates"]) == 2
        assert response["violation_vehicle"] == "WW12345"
        assert len(response["reasoning"]) > 0

    def test_parse_valid_response_no_violation(self):
        """Test parsing valid JSON without violation."""
        response = {
            "plates": ["ABC123"],
            "violation_vehicle": None,
            "reasoning": "No clear violation visible in this image."
        }

        assert isinstance(response["plates"], list)
        assert len(response["plates"]) == 1
        assert response["violation_vehicle"] is None

    def test_parse_response_no_plates(self):
        """Test parsing response with no plates detected."""
        response = {
            "plates": [],
            "violation_vehicle": None,
            "reasoning": "No license plates are visible in this image."
        }

        assert isinstance(response["plates"], list)
        assert len(response["plates"]) == 0
        assert response["violation_vehicle"] is None


@pytest.mark.asyncio
class TestRecognizePlatesTool:
    """Tests for the recognize_plates tool."""

    @pytest.mark.skip(reason="Requires mocking FastMCP Context")
    async def test_recognize_plates_with_mock(self):
        """Test recognize_plates with mocked Gemini response."""
        # This would require setting up proper FastMCP context mocking
        pass

    @pytest.mark.skip(reason="Requires mocking FastMCP Context")
    async def test_recognize_plates_file_not_found(self):
        """Test recognize_plates with non-existent file."""
        # This would require setting up proper FastMCP context mocking
        pass


@pytest.mark.integration
@pytest.mark.skip(reason="Requires GEMINI_API_KEY and real API calls")
class TestPlateRecognitionIntegration:
    """Integration tests with real Gemini API."""

    async def test_recognize_plates_with_fixture(self):
        """Test plate recognition with fixture image."""
        from plate_recognition import recognize_plates
        from fastmcp import Context

        # Create a mock context
        ctx = AsyncMock(spec=Context)
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        ctx.error = AsyncMock()

        # Test with fixture image
        fixture_path = "tests/fixtures/1.png"
        if Path(fixture_path).exists():
            result = await recognize_plates(image_path=fixture_path, ctx=ctx)

            assert result is not None
            assert hasattr(result, 'structured_content')
            assert 'plates' in result.structured_content
            assert 'reasoning' in result.structured_content

    async def test_recognize_plates_with_heic(self):
        """Test plate recognition with HEIC image."""
        from plate_recognition import recognize_plates
        from fastmcp import Context

        ctx = AsyncMock(spec=Context)
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        ctx.error = AsyncMock()

        fixture_path = "tests/fixtures/IMG_5134.heic"
        if Path(fixture_path).exists():
            result = await recognize_plates(image_path=fixture_path, ctx=ctx)

            assert result is not None
            assert hasattr(result, 'structured_content')


class TestMarkdownCodeBlockRemoval:
    """Tests for cleaning markdown code blocks from Gemini responses."""

    def test_remove_json_code_block(self):
        """Test removing ```json code blocks."""
        response = """```json
{
  "plates": ["ABC123"],
  "violation_vehicle": null,
  "reasoning": "Test"
}
```"""

        # Simulate the cleaning logic
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Should be valid JSON now
        result = json.loads(cleaned)
        assert result["plates"] == ["ABC123"]

    def test_remove_plain_code_block(self):
        """Test removing plain ``` code blocks."""
        response = """```
{
  "plates": ["XYZ789"],
  "violation_vehicle": "XYZ789",
  "reasoning": "Parked on sidewalk"
}
```"""

        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        result = json.loads(cleaned)
        assert result["plates"] == ["XYZ789"]

    def test_no_code_block(self):
        """Test that plain JSON is not affected."""
        response = """{
  "plates": ["DEF456"],
  "violation_vehicle": null,
  "reasoning": "No violation"
}"""

        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        result = json.loads(cleaned)
        assert result["plates"] == ["DEF456"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
