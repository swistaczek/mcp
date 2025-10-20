"""
Tests for the Gemini Alt Tag Generator MCP server.
"""

import base64
import io
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for GEMINI_API_KEY and provide helpful error message
if not os.getenv("GEMINI_API_KEY"):
    pytest.skip(
        "GEMINI_API_KEY not found in environment or .env file. "
        "Please set GEMINI_API_KEY in your .env file to run tests.",
        allow_module_level=True
    )

import gemini_alt


class TestImageOptimization:
    """Test image optimization utilities."""

    def test_is_text_heavy_image_document(self):
        """Test detection of document-like images."""
        # Portrait document (A4-like)
        image = Image.new('RGB', (210, 297))
        assert gemini_alt.is_text_heavy_image(image) is True

        # Landscape document
        image = Image.new('RGB', (297, 210))
        assert gemini_alt.is_text_heavy_image(image) is True

        # Wide screenshot
        image = Image.new('RGB', (1920, 1080))
        assert gemini_alt.is_text_heavy_image(image) is True

    def test_is_text_heavy_image_regular(self):
        """Test detection of regular images."""
        # Square image
        image = Image.new('RGB', (500, 500))
        assert gemini_alt.is_text_heavy_image(image) is False

        # Regular photo aspect ratio
        image = Image.new('RGB', (400, 300))
        assert gemini_alt.is_text_heavy_image(image) is False

    def test_optimize_image_resize(self):
        """Test image resizing functionality."""
        # Create a large image
        image = Image.new('RGB', (3000, 2000), color='red')

        # Optimize it
        optimized_bytes, mime_type = gemini_alt.optimize_image(image, max_dimension=1024)

        # Check the result
        assert mime_type == 'image/jpeg'

        # Load optimized image and check dimensions
        optimized_image = Image.open(io.BytesIO(optimized_bytes))
        assert optimized_image.size[0] <= 1024
        assert optimized_image.size[1] <= 1024

        # Check aspect ratio is maintained
        original_ratio = 3000 / 2000
        new_ratio = optimized_image.size[0] / optimized_image.size[1]
        assert abs(original_ratio - new_ratio) < 0.01

    def test_optimize_image_no_resize_needed(self):
        """Test that small images are not resized."""
        # Create a small image
        image = Image.new('RGB', (500, 400), color='blue')

        # Optimize it
        optimized_bytes, mime_type = gemini_alt.optimize_image(image, max_dimension=1024)

        # Load optimized image
        optimized_image = Image.open(io.BytesIO(optimized_bytes))

        # Size should remain the same
        assert optimized_image.size == (500, 400)

    def test_optimize_image_with_transparency(self):
        """Test optimization of images with transparency."""
        # Create RGBA image with transparency
        image = Image.new('RGBA', (500, 400), (255, 0, 0, 128))

        # Optimize it
        optimized_bytes, mime_type = gemini_alt.optimize_image(image)

        # Load optimized image
        optimized_image = Image.open(io.BytesIO(optimized_bytes))

        # Should be converted to RGB
        assert optimized_image.mode == 'RGB'
        assert mime_type == 'image/jpeg'


class TestImageLoading:
    """Test image loading from various sources."""

    @pytest.mark.asyncio
    async def test_load_image_from_file(self, tmp_path):
        """Test loading image from file path."""
        # Create a test image file
        image = Image.new('RGB', (100, 100), color='green')
        image_path = tmp_path / "test.png"
        image.save(image_path)

        # Mock context
        ctx = AsyncMock()

        # Load the image
        image_bytes, mime_type = await gemini_alt.load_image(str(image_path), ctx)

        # Check result
        assert isinstance(image_bytes, bytes)
        assert mime_type == 'image/jpeg'
        ctx.debug.assert_called()

    @pytest.mark.asyncio
    async def test_load_image_from_data_url(self):
        """Test loading image from data URL."""
        # Create a small test image
        image = Image.new('RGB', (50, 50), color='yellow')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        # Create data URL
        data_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"

        # Mock context
        ctx = AsyncMock()

        # Load the image
        loaded_bytes, mime_type = await gemini_alt.load_image(data_url, ctx)

        # Check result
        assert isinstance(loaded_bytes, bytes)
        assert mime_type == 'image/jpeg'

    @pytest.mark.asyncio
    async def test_load_image_invalid_input(self):
        """Test error handling for invalid image input."""
        ctx = AsyncMock()

        # Test with invalid path
        with pytest.raises(ValueError, match="Invalid image input"):
            await gemini_alt.load_image("nonexistent.png", ctx)

        # Test with URL (not implemented)
        with pytest.raises(ValueError, match="URL image loading not implemented"):
            await gemini_alt.load_image("http://example.com/image.png", ctx)


class TestContextLoading:
    """Test context loading from files and raw text."""

    @pytest.mark.asyncio
    async def test_load_context_from_file(self, tmp_path):
        """Test loading context from a file path."""
        # Create a test context file
        context_file = tmp_path / "context.md"
        test_content = "# Test Document\n\nThis is test context for alt tag generation."
        context_file.write_text(test_content)

        # Mock context
        ctx = AsyncMock()

        # Load context from file path
        result = await gemini_alt.load_context(str(context_file), ctx)

        # Check result
        assert result == test_content
        ctx.debug.assert_called_with(f"Loaded context from file: context.md ({len(test_content)} characters)")

    @pytest.mark.asyncio
    async def test_load_context_from_raw_text(self):
        """Test using raw text as context."""
        test_context = "This is raw context text"

        # Mock context
        ctx = AsyncMock()

        # Load context as raw text
        result = await gemini_alt.load_context(test_context, ctx)

        # Check result
        assert result == test_context
        ctx.debug.assert_called_with(f"Using context as raw text ({len(test_context)} characters)")

    @pytest.mark.asyncio
    async def test_load_context_nonexistent_file(self):
        """Test that non-existent file paths are treated as raw text."""
        fake_path = "/nonexistent/path/to/file.md"

        # Mock context
        ctx = AsyncMock()

        # Should treat as raw text since file doesn't exist
        result = await gemini_alt.load_context(fake_path, ctx)

        # Check result
        assert result == fake_path
        ctx.debug.assert_called_with(f"Using context as raw text ({len(fake_path)} characters)")

    @pytest.mark.asyncio
    async def test_load_context_none(self):
        """Test loading None context."""
        # Mock context
        ctx = AsyncMock()

        # Load None context
        result = await gemini_alt.load_context(None, ctx)

        # Check result
        assert result is None
        ctx.debug.assert_not_called()


class TestPromptGeneration:
    """Test prompt generation for Gemini."""

    def test_create_alt_generation_prompt_basic(self):
        """Test basic prompt generation."""
        prompt = gemini_alt.create_alt_generation_prompt()

        assert "expert at creating accessible alt text" in prompt
        assert "Concise but descriptive" in prompt
        assert "50-125 characters" in prompt

    def test_create_alt_generation_prompt_with_context(self):
        """Test prompt generation with document context."""
        context = "This is a tutorial about setting up Facebook Pixel"
        prompt = gemini_alt.create_alt_generation_prompt(context=context)

        assert "Facebook Pixel" in prompt
        assert "document context" in prompt

    def test_create_alt_generation_prompt_batch_mode(self):
        """Test prompt generation for batch mode."""
        prompt = gemini_alt.create_alt_generation_prompt(is_batch=True)

        assert "JSON format" in prompt
        assert "image_1" in prompt


class TestGeminiIntegration:
    """Test Gemini API integration."""

    @pytest.mark.asyncio
    @patch('gemini_alt.model')
    async def test_generate_alt_for_batch_success(self, mock_model):
        """Test successful batch alt tag generation."""
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = '{"image_1": "Connect data sources button", "image_2": "Web selection dialog"}'
        mock_model.generate_content.return_value = mock_response

        # Create test images
        images = [
            (b"fake_image_1", "image/jpeg"),
            (b"fake_image_2", "image/jpeg")
        ]

        ctx = AsyncMock()

        # Generate alt tags
        result = await gemini_alt.generate_alt_for_batch(images, None, ctx)

        # Check results
        assert result["image_1"] == "Connect data sources button"
        assert result["image_2"] == "Web selection dialog"
        mock_model.generate_content.assert_called_once()

    @pytest.mark.asyncio
    @patch('gemini_alt.model')
    async def test_generate_alt_for_batch_json_error(self, mock_model):
        """Test handling of JSON parsing errors."""
        # Mock Gemini response with invalid JSON
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON"
        mock_model.generate_content.return_value = mock_response

        # Create test images
        images = [(b"fake_image", "image/jpeg")]

        ctx = AsyncMock()

        # Generate alt tags
        result = await gemini_alt.generate_alt_for_batch(images, None, ctx)

        # Should fallback to generic alt text
        assert "image_1" in result
        ctx.warning.assert_called_with("Failed to parse batch response as JSON, using fallback parsing")


class TestGenerateAltTagsTool:
    """Test the main generate_alt_tags tool."""

    @pytest.mark.asyncio
    @patch('gemini_alt.model')
    @patch('gemini_alt.load_image')
    async def test_generate_alt_tags_single_image(self, mock_load_image, mock_model):
        """Test generating alt tag for a single image."""
        # Mock image loading
        mock_load_image.return_value = (b"fake_image_data", "image/jpeg")

        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = "Facebook Pixel setup dialog"
        mock_model.generate_content.return_value = mock_response

        # Mock context
        ctx = AsyncMock()

        # Call the tool (access underlying function via .fn)
        result = await gemini_alt.generate_alt_tags.fn(
            images=["test.png"],
            context=None,
            batch_size=5,
            ctx=ctx
        )

        # Check result
        assert result.content[0].type == "text"
        assert "Generated alt tags for 1/1 image(s)" in result.content[0].text
        assert result.structured_content["alt_tags"]["test.png"] == "Facebook Pixel setup dialog"
        assert result.structured_content["stats"]["successful"] == 1

    @pytest.mark.asyncio
    @patch('gemini_alt.model')
    @patch('gemini_alt.load_image')
    async def test_generate_alt_tags_multiple_images(self, mock_load_image, mock_model):
        """Test generating alt tags for multiple images."""
        # Mock image loading
        mock_load_image.side_effect = [
            (b"image1", "image/jpeg"),
            (b"image2", "image/jpeg"),
            (b"image3", "image/jpeg")
        ]

        # Mock Gemini response for batch
        mock_response = MagicMock()
        mock_response.text = '{"image_1": "First image", "image_2": "Second image", "image_3": "Third image"}'
        mock_model.generate_content.return_value = mock_response

        # Mock context
        ctx = AsyncMock()

        # Call the tool (access underlying function via .fn)
        result = await gemini_alt.generate_alt_tags.fn(
            images=["1.png", "2.png", "3.png"],
            context="Test document context",
            batch_size=5,
            ctx=ctx
        )

        # Check result
        assert result.structured_content["stats"]["successful"] == 3
        assert result.structured_content["alt_tags"]["1.png"] == "First image"
        assert result.structured_content["alt_tags"]["2.png"] == "Second image"
        assert result.structured_content["alt_tags"]["3.png"] == "Third image"

    @pytest.mark.asyncio
    @patch('gemini_alt.model')
    @patch('gemini_alt.load_image')
    async def test_generate_alt_tags_with_context_file(self, mock_load_image, mock_model, tmp_path):
        """Test generating alt tags with context loaded from a file."""
        # Create a context file
        context_file = tmp_path / "context.md"
        context_content = "# Product Documentation\n\nThis image shows our main product interface."
        context_file.write_text(context_content)

        # Mock image loading
        mock_load_image.return_value = (b"fake_image_data", "image/jpeg")

        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = "Product interface screenshot"
        mock_model.generate_content.return_value = mock_response

        # Mock context
        ctx = AsyncMock()

        # Call the tool with file path as context
        result = await gemini_alt.generate_alt_tags.fn(
            images=["test.png"],
            context=str(context_file),  # Pass file path instead of content
            batch_size=5,
            ctx=ctx
        )

        # Verify context was loaded from file
        ctx.debug.assert_any_call(f"Loaded context from file: context.md ({len(context_content)} characters)")

        # Check result
        assert result.structured_content["alt_tags"]["test.png"] == "Product interface screenshot"
        assert result.structured_content["metadata"]["context_provided"] is True

    @pytest.mark.asyncio
    @patch('gemini_alt.load_image')
    async def test_generate_alt_tags_with_failed_images(self, mock_load_image):
        """Test handling of failed image loading."""
        # Mock image loading to fail
        mock_load_image.side_effect = Exception("Failed to load")

        # Mock context
        ctx = AsyncMock()

        # Call the tool (access underlying function via .fn)
        result = await gemini_alt.generate_alt_tags.fn(
            images=["bad_image.png"],
            context=None,
            batch_size=5,
            ctx=ctx
        )

        # Check result
        assert "Failed to load any images" in result.content[0].text
        assert result.structured_content["error"] == "No images could be processed"


@pytest.mark.integration
class TestIntegrationWithFixtures:
    """Integration tests using actual test fixtures."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.getenv("GEMINI_API_KEY", "test-api-key") == "test-api-key",
        reason="Real Gemini API key required for integration test"
    )
    async def test_generate_alt_tag_for_fixture_image(self):
        """Test with actual fixture image and markdown context."""
        # Read the markdown context
        fixtures_path = Path(__file__).parent / "fixtures"
        with open(fixtures_path / "meta_pixel_setup.md", "r") as f:
            context = f.read()

        # Mock context
        ctx = AsyncMock()

        # Generate alt tag for the fixture image (access underlying function via .fn)
        result = await gemini_alt.generate_alt_tags.fn(
            images=[str(fixtures_path / "1.png")],
            context=context,
            batch_size=5,
            ctx=ctx
        )

        # Check that we got a meaningful result
        assert result.structured_content["stats"]["successful"] == 1
        alt_text = result.structured_content["alt_tags"][str(fixtures_path / "1.png")]

        # The alt text should be relevant to Facebook/Meta Pixel setup
        assert len(alt_text) > 0
        assert len(alt_text) <= 200  # Should be concise

        # Print for manual verification
        print(f"\nGenerated alt text: {alt_text}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])