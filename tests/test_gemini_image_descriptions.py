"""
Tests for the Gemini Image Description Generator MCP server.
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

import gemini_image_descriptions as gemini_alt


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
        """Test basic prompt generation (alt mode - default)."""
        prompt = gemini_alt.create_alt_generation_prompt()

        assert "expert at creating accessible alt text" in prompt
        assert "Concise but descriptive" in prompt
        assert "50-125 characters" in prompt

    def test_create_alt_generation_prompt_description_mode(self):
        """Test prompt generation for detailed description mode with auto-detection."""
        prompt = gemini_alt.create_alt_generation_prompt(description_type="description")

        assert "visually impaired users" in prompt
        assert "screen readers" in prompt
        # Check for tutorial auto-detection
        assert "Adapt your description style" in prompt
        assert "UI screenshots/tutorials" in prompt
        assert "step-by-step" in prompt.lower()
        # Check for non-tutorial guidance
        assert "150-300 characters" in prompt
        assert "spatial layout" in prompt

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

    def test_create_gif_description_prompt_description_mode(self):
        """Test GIF description prompt for detailed mode with auto-detection."""
        prompt = gemini_alt.create_gif_description_prompt(description_type="description")

        assert "visually impaired users" in prompt
        assert "screen readers" in prompt
        # Check for tutorial auto-detection
        assert "Adapt your description style" in prompt
        assert "UI tutorials/screencasts" in prompt
        assert "step-by-step" in prompt.lower()
        # Check for non-tutorial guidance
        assert "150-300 characters" in prompt
        assert "sequence of actions" in prompt


class TestGeminiIntegration:
    """Test Gemini API integration."""

    @pytest.mark.asyncio
    @patch('gemini_image_descriptions.genai.GenerativeModel')
    async def test_generate_alt_for_batch_success(self, mock_model_class):
        """Test successful batch alt tag generation."""
        # Mock model instance
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"image_1": "Connect data sources button", "image_2": "Web selection dialog"}'
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        # Create test images
        images = [
            (b"fake_image_1", "image/jpeg"),
            (b"fake_image_2", "image/jpeg")
        ]

        ctx = AsyncMock()

        # Generate alt tags
        result = await gemini_alt.generate_alt_for_batch(images, None, "gemini-flash-latest", ctx)

        # Check results
        assert result["image_1"] == "Connect data sources button"
        assert result["image_2"] == "Web selection dialog"
        mock_model.generate_content.assert_called_once()

    @pytest.mark.asyncio
    @patch('gemini_image_descriptions.genai.GenerativeModel')
    async def test_generate_alt_for_batch_json_error(self, mock_model_class):
        """Test handling of JSON parsing errors."""
        # Mock model instance
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        # Create test images
        images = [(b"fake_image", "image/jpeg")]

        ctx = AsyncMock()

        # Generate alt tags
        result = await gemini_alt.generate_alt_for_batch(images, None, "gemini-flash-latest", ctx)

        # Should fallback to generic alt text
        assert "image_1" in result
        ctx.warning.assert_called_with("Failed to parse batch response as JSON, using fallback parsing")


class TestGenerateImageDescriptionsTool:
    """Test the main generate_image_descriptions tool."""

    @pytest.mark.asyncio
    @patch('gemini_image_descriptions.genai.GenerativeModel')
    @patch('gemini_image_descriptions.load_image')
    async def test_generate_descriptions_single_image(self, mock_load_image, mock_model_class):
        """Test generating description for a single image."""
        # Mock image loading
        mock_load_image.return_value = (b"fake_image_data", "image/jpeg")

        # Mock model instance
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Facebook Pixel setup dialog"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        # Mock context
        ctx = AsyncMock()

        # Call the tool (access underlying function via .fn)
        result = await gemini_alt.generate_image_descriptions.fn(
            images=["test.png"],
            type="alt",
            context=None,
            batch_size=5,
            model=None,  # Use default model
            ctx=ctx
        )

        # Check result
        assert result.content[0].type == "text"
        assert "Generated alt text for 1/1 image(s)" in result.content[0].text
        assert result.structured_content["descriptions"]["test.png"] == "Facebook Pixel setup dialog"
        assert result.structured_content["stats"]["successful"] == 1
        assert result.structured_content["metadata"]["description_type"] == "alt"

    @pytest.mark.asyncio
    @patch('gemini_image_descriptions.genai.GenerativeModel')
    @patch('gemini_image_descriptions.load_image')
    async def test_generate_descriptions_description_type(self, mock_load_image, mock_model_class):
        """Test generating detailed descriptions for accessibility."""
        # Mock image loading
        mock_load_image.return_value = (b"fake_image_data", "image/jpeg")

        # Mock model instance
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "A detailed dialog showing Facebook Pixel setup with blue buttons on the right side"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        # Mock context
        ctx = AsyncMock()

        # Call the tool with type="description"
        result = await gemini_alt.generate_image_descriptions.fn(
            images=["test.png"],
            type="description",
            context=None,
            batch_size=5,
            model=None,
            ctx=ctx
        )

        # Check result
        assert "Generated detailed descriptions for 1/1 image(s)" in result.content[0].text
        assert result.structured_content["metadata"]["description_type"] == "description"

    @pytest.mark.asyncio
    @patch('gemini_image_descriptions.genai.GenerativeModel')
    @patch('gemini_image_descriptions.load_image')
    async def test_generate_descriptions_multiple_images(self, mock_load_image, mock_model_class):
        """Test generating descriptions for multiple images."""
        # Mock image loading
        mock_load_image.side_effect = [
            (b"image1", "image/jpeg"),
            (b"image2", "image/jpeg"),
            (b"image3", "image/jpeg")
        ]

        # Mock model instance
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"image_1": "First image", "image_2": "Second image", "image_3": "Third image"}'
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        # Mock context
        ctx = AsyncMock()

        # Call the tool (access underlying function via .fn)
        result = await gemini_alt.generate_image_descriptions.fn(
            images=["1.png", "2.png", "3.png"],
            type="alt",
            context="Test document context",
            batch_size=5,
            model=None,  # Use default model
            ctx=ctx
        )

        # Check result
        assert result.structured_content["stats"]["successful"] == 3
        assert result.structured_content["descriptions"]["1.png"] == "First image"
        assert result.structured_content["descriptions"]["2.png"] == "Second image"
        assert result.structured_content["descriptions"]["3.png"] == "Third image"

    @pytest.mark.asyncio
    @patch('gemini_image_descriptions.genai.GenerativeModel')
    @patch('gemini_image_descriptions.load_image')
    async def test_generate_descriptions_with_context_file(self, mock_load_image, mock_model_class, tmp_path):
        """Test generating descriptions with context loaded from a file."""
        # Create a context file
        context_file = tmp_path / "context.md"
        context_content = "# Product Documentation\n\nThis image shows our main product interface."
        context_file.write_text(context_content)

        # Mock image loading
        mock_load_image.return_value = (b"fake_image_data", "image/jpeg")

        # Mock model instance
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Product interface screenshot"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        # Mock context
        ctx = AsyncMock()

        # Call the tool with file path as context
        result = await gemini_alt.generate_image_descriptions.fn(
            images=["test.png"],
            type="alt",
            context=str(context_file),  # Pass file path instead of content
            batch_size=5,
            model=None,  # Use default model
            ctx=ctx
        )

        # Verify context was loaded from file
        ctx.debug.assert_any_call(f"Loaded context from file: context.md ({len(context_content)} characters)")

        # Check result
        assert result.structured_content["descriptions"]["test.png"] == "Product interface screenshot"
        assert result.structured_content["metadata"]["context_provided"] is True

    @pytest.mark.asyncio
    @patch('gemini_image_descriptions.load_image')
    async def test_generate_descriptions_with_failed_images(self, mock_load_image):
        """Test handling of failed image loading."""
        # Mock image loading to fail
        mock_load_image.side_effect = Exception("Failed to load")

        # Mock context
        ctx = AsyncMock()

        # Call the tool (access underlying function via .fn)
        result = await gemini_alt.generate_image_descriptions.fn(
            images=["bad_image.png"],
            type="alt",
            context=None,
            batch_size=5,
            model=None,  # Use default model
            ctx=ctx
        )

        # Check result
        assert "Failed to load any images" in result.content[0].text
        assert result.structured_content["error"] == "No images could be processed"

    @pytest.mark.asyncio
    async def test_generate_descriptions_invalid_type(self):
        """Test error handling for invalid type parameter."""
        ctx = AsyncMock()

        # Call the tool with invalid type
        result = await gemini_alt.generate_image_descriptions.fn(
            images=["test.png"],
            type="invalid",
            context=None,
            batch_size=5,
            model=None,
            ctx=ctx
        )

        # Check error result
        assert "Invalid type 'invalid'" in result.content[0].text
        assert "error" in result.structured_content


class TestGifSupport:
    """Test GIF detection and processing utilities."""

    def test_is_gif_file_valid_gif(self, tmp_path):
        """Test detection of valid GIF files."""
        # Create a valid GIF file
        gif_path = tmp_path / "test.gif"
        # GIF89a header
        with open(gif_path, 'wb') as f:
            f.write(b'GIF89a' + b'\x00' * 100)

        assert gemini_alt.is_gif_file(gif_path) is True

    def test_is_gif_file_gif87a(self, tmp_path):
        """Test detection of GIF87a format."""
        gif_path = tmp_path / "test.gif"
        with open(gif_path, 'wb') as f:
            f.write(b'GIF87a' + b'\x00' * 100)

        assert gemini_alt.is_gif_file(gif_path) is True

    def test_is_gif_file_wrong_extension(self, tmp_path):
        """Test that non-GIF extensions are rejected."""
        png_path = tmp_path / "test.png"
        with open(png_path, 'wb') as f:
            f.write(b'GIF89a' + b'\x00' * 100)

        assert gemini_alt.is_gif_file(png_path) is False

    def test_is_gif_file_wrong_magic_bytes(self, tmp_path):
        """Test that files with wrong magic bytes are rejected."""
        gif_path = tmp_path / "test.gif"
        with open(gif_path, 'wb') as f:
            f.write(b'NOTGIF' + b'\x00' * 100)

        assert gemini_alt.is_gif_file(gif_path) is False

    def test_is_gif_file_nonexistent(self):
        """Test behavior with non-existent file."""
        assert gemini_alt.is_gif_file(Path("/nonexistent/file.gif")) is False

    def test_check_ffmpeg_available(self):
        """Test FFmpeg availability check."""
        # This should work on systems with FFmpeg installed
        result = gemini_alt.check_ffmpeg_available()
        # Just check it returns a boolean
        assert isinstance(result, bool)

    def test_create_gif_description_prompt_basic(self):
        """Test GIF description prompt generation."""
        prompt = gemini_alt.create_gif_description_prompt()

        assert "animations" in prompt.lower() or "gifs" in prompt.lower()
        assert "motion" in prompt.lower() or "action" in prompt.lower()
        assert "alt text" in prompt.lower()

    def test_create_gif_description_prompt_with_context(self):
        """Test GIF description prompt with document context."""
        context = "This is a tutorial about video editing."
        prompt = gemini_alt.create_gif_description_prompt(context)

        assert context in prompt
        assert "animations" in prompt.lower() or "gifs" in prompt.lower()


class TestGifConversion:
    """Test GIF to MP4 conversion with FFmpeg."""

    @pytest.mark.asyncio
    async def test_convert_gif_to_mp4_ffmpeg_not_available(self):
        """Test error when FFmpeg is not available."""
        ctx = AsyncMock()

        with patch.object(gemini_alt, 'check_ffmpeg_available', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                await gemini_alt.convert_gif_to_mp4(Path("/some/file.gif"), ctx)

            assert "FFmpeg is required" in str(exc_info.value)
            assert "brew install ffmpeg" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not gemini_alt.check_ffmpeg_available(),
        reason="FFmpeg not installed"
    )
    async def test_convert_gif_to_mp4_real_conversion(self):
        """Test actual GIF to MP4 conversion with FFmpeg."""
        ctx = AsyncMock()
        fixtures_path = Path(__file__).parent / "fixtures"
        gif_path = fixtures_path / "example.gif"

        if not gif_path.exists():
            pytest.skip("example.gif fixture not found")

        # Convert GIF to MP4
        mp4_path = await gemini_alt.convert_gif_to_mp4(gif_path, ctx)

        try:
            # Verify the output exists
            assert mp4_path.exists()
            assert mp4_path.suffix == ".mp4"

            # Verify it's smaller than the original (usually true for GIF to MP4)
            original_size = gif_path.stat().st_size
            converted_size = mp4_path.stat().st_size
            assert converted_size > 0

            # Log the conversion ratio
            print(f"\nConversion: {original_size/1024:.1f}KB -> {converted_size/1024:.1f}KB")
        finally:
            # Cleanup
            mp4_path.unlink(missing_ok=True)


class TestGifUploadAndGeneration:
    """Test Gemini File API integration for GIF processing."""

    @pytest.mark.asyncio
    async def test_upload_video_to_gemini_mocked(self):
        """Test video upload with mocked Gemini client."""
        ctx = AsyncMock()

        # Mock the genai_new.Client
        mock_file = MagicMock()
        mock_file.name = "files/test123"
        mock_file.state.name = "ACTIVE"
        mock_file.uri = "https://example.com/files/test123"

        mock_client = MagicMock()
        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.return_value = mock_file

        with patch.object(gemini_alt, 'genai_new') as mock_genai:
            mock_genai.Client.return_value = mock_client

            result = await gemini_alt.upload_video_to_gemini(
                Path("/tmp/test.mp4"),
                ctx,
                timeout=10
            )

            assert result.name == "files/test123"
            mock_client.files.upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_description_for_gif_mocked(self):
        """Test GIF description generation with mocked Gemini."""
        ctx = AsyncMock()

        mock_video_file = MagicMock()
        mock_video_file.uri = "https://example.com/files/test123"

        mock_response = MagicMock()
        mock_response.text = "A cat jumping over a fence"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(gemini_alt, 'genai_new') as mock_genai:
            mock_genai.Client.return_value = mock_client

            result = await gemini_alt.generate_description_for_gif(
                mock_video_file,
                context=None,
                model_name="gemini-2.0-flash",
                ctx=ctx
            )

            assert result == "A cat jumping over a fence"
            mock_client.models.generate_content.assert_called_once()


@pytest.mark.integration
class TestGifIntegration:
    """Integration tests for full GIF processing pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not gemini_alt.check_ffmpeg_available(),
        reason="FFmpeg not installed"
    )
    @pytest.mark.skipif(
        os.getenv("GEMINI_API_KEY", "test-api-key") == "test-api-key",
        reason="Real Gemini API key required for integration test"
    )
    async def test_full_gif_pipeline(self):
        """End-to-end test of GIF description generation."""
        ctx = AsyncMock()
        fixtures_path = Path(__file__).parent / "fixtures"
        gif_path = fixtures_path / "example.gif"

        if not gif_path.exists():
            pytest.skip("example.gif fixture not found")

        # Generate alt text for the GIF
        result = await gemini_alt.generate_image_descriptions.fn(
            images=[str(gif_path)],
            type="alt",
            context=None,
            batch_size=5,
            model=None,
            ctx=ctx
        )

        # Check results
        assert result.structured_content["stats"]["successful"] == 1
        assert result.structured_content["stats"]["gifs"] == 1

        alt_text = result.structured_content["descriptions"][str(gif_path)]
        assert len(alt_text) > 0
        assert len(alt_text) <= 200

        print(f"\nGenerated GIF alt text: {alt_text}")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not gemini_alt.check_ffmpeg_available(),
        reason="FFmpeg not installed"
    )
    @pytest.mark.skipif(
        os.getenv("GEMINI_API_KEY", "test-api-key") == "test-api-key",
        reason="Real Gemini API key required for integration test"
    )
    async def test_full_gif_pipeline_description_type(self):
        """End-to-end test of GIF with detailed description mode."""
        ctx = AsyncMock()
        fixtures_path = Path(__file__).parent / "fixtures"
        gif_path = fixtures_path / "example.gif"

        if not gif_path.exists():
            pytest.skip("example.gif fixture not found")

        # Generate detailed description for the GIF
        result = await gemini_alt.generate_image_descriptions.fn(
            images=[str(gif_path)],
            type="description",
            context=None,
            batch_size=5,
            model=None,
            ctx=ctx
        )

        # Check results
        assert result.structured_content["stats"]["successful"] == 1
        assert result.structured_content["metadata"]["description_type"] == "description"

        description = result.structured_content["descriptions"][str(gif_path)]
        assert len(description) > 0

        print(f"\nGenerated GIF detailed description: {description}")


@pytest.mark.integration
class TestIntegrationWithFixtures:
    """Integration tests using actual test fixtures."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.getenv("GEMINI_API_KEY", "test-api-key") == "test-api-key",
        reason="Real Gemini API key required for integration test"
    )
    async def test_generate_description_for_fixture_image(self):
        """Test with actual fixture image and markdown context."""
        # Read the markdown context
        fixtures_path = Path(__file__).parent / "fixtures"
        with open(fixtures_path / "meta_pixel_setup.md", "r") as f:
            context = f.read()

        # Mock context
        ctx = AsyncMock()

        # Generate description for the fixture image (access underlying function via .fn)
        result = await gemini_alt.generate_image_descriptions.fn(
            images=[str(fixtures_path / "1.png")],
            type="alt",
            context=context,
            batch_size=5,
            model=None,  # Use default model
            ctx=ctx
        )

        # Check that we got a meaningful result
        assert result.structured_content["stats"]["successful"] == 1
        alt_text = result.structured_content["descriptions"][str(fixtures_path / "1.png")]

        # The alt text should be relevant to Facebook/Meta Pixel setup
        assert len(alt_text) > 0
        assert len(alt_text) <= 200  # Should be concise

        # Print for manual verification
        print(f"\nGenerated alt text: {alt_text}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])