"""
Tests for the EXIF Metadata Extractor MCP server.
"""

import io
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

import exif_extractor

# Get the underlying function from the wrapped tool
_analyze_image_metadata = exif_extractor.analyze_image_metadata.fn


class TestConvertToDegrees:
    """Test GPS coordinate conversion."""

    def test_convert_to_degrees_basic(self):
        """Test conversion of GPS coordinates to decimal degrees."""
        # 52 degrees, 24 minutes, 30.41 seconds = 52.408447°
        result = exif_extractor.convert_to_degrees((52.0, 24.0, 30.41))
        assert abs(result - 52.408447) < 0.000001

    def test_convert_to_degrees_zero(self):
        """Test conversion with zero minutes and seconds."""
        result = exif_extractor.convert_to_degrees((45.0, 0.0, 0.0))
        assert result == 45.0

    def test_convert_to_degrees_precision(self):
        """Test conversion with high precision."""
        # 16 degrees, 52 minutes, 4.14 seconds = 16.867817°
        result = exif_extractor.convert_to_degrees((16.0, 52.0, 4.14))
        assert abs(result - 16.867817) < 0.000001


class TestExtractExifData:
    """Test EXIF data extraction from images."""

    def test_extract_exif_from_heic_with_gps(self):
        """Test EXIF extraction from HEIC file with GPS data."""
        fixture_path = Path(__file__).parent / "fixtures" / "IMG_5134.heic"

        if not fixture_path.exists():
            pytest.skip("Test fixture IMG_5134.heic not found")

        exif_dict, image = exif_extractor.extract_exif_data(fixture_path)

        # Verify basic metadata
        assert image.format == "HEIF"
        assert image.size == (4032, 3024)

        # Verify EXIF data extracted
        assert "Make" in exif_dict
        assert exif_dict["Make"] == "Apple"
        assert "Model" in exif_dict
        assert "iPhone" in exif_dict["Model"]

        # Verify GPS tag present
        assert "GPSInfo" in exif_dict

    def test_extract_exif_from_heic_without_gps(self):
        """Test EXIF extraction from HEIC file without GPS data."""
        fixture_path = Path(__file__).parent / "fixtures" / "IMG_2852.heic"

        if not fixture_path.exists():
            pytest.skip("Test fixture IMG_2852.heic not found")

        exif_dict, image = exif_extractor.extract_exif_data(fixture_path)

        # Verify basic metadata
        assert image.format == "HEIF"
        assert "Make" in exif_dict
        assert exif_dict["Make"] == "Apple"

        # Verify NO GPS tag
        assert "GPSInfo" not in exif_dict

    def test_extract_exif_from_png(self):
        """Test EXIF extraction from PNG file."""
        fixture_path = Path(__file__).parent / "fixtures" / "1.png"

        if not fixture_path.exists():
            pytest.skip("Test fixture 1.png not found")

        exif_dict, image = exif_extractor.extract_exif_data(fixture_path)

        # PNG files typically have minimal EXIF
        assert image.format == "PNG"
        # May have resolution data
        assert len(exif_dict) >= 0  # May be empty or have basic tags


class TestParseGpsData:
    """Test GPS data parsing from EXIF."""

    def test_parse_gps_data_with_coordinates(self):
        """Test parsing GPS data from image with coordinates."""
        fixture_path = Path(__file__).parent / "fixtures" / "IMG_5134.heic"

        if not fixture_path.exists():
            pytest.skip("Test fixture IMG_5134.heic not found")

        exif_dict, image = exif_extractor.extract_exif_data(fixture_path)
        gps_data = exif_extractor.parse_gps_data(exif_dict, image)

        # Verify GPS data extracted
        assert gps_data is not None
        assert "latitude" in gps_data
        assert "longitude" in gps_data

        # Verify known coordinates for IMG_5134.heic (Poznań, Poland)
        assert abs(gps_data["latitude"] - 52.408447) < 0.0001
        assert abs(gps_data["longitude"] - 16.867817) < 0.0001

        # Verify additional GPS metadata
        assert "altitude_meters" in gps_data
        assert gps_data["altitude_meters"] > 0

        assert "speed_kmh" in gps_data
        assert "direction_degrees" in gps_data
        assert "timestamp_utc" in gps_data
        assert "accuracy_meters" in gps_data

    def test_parse_gps_data_without_coordinates(self):
        """Test parsing GPS data from image without GPS."""
        fixture_path = Path(__file__).parent / "fixtures" / "IMG_2852.heic"

        if not fixture_path.exists():
            pytest.skip("Test fixture IMG_2852.heic not found")

        exif_dict, image = exif_extractor.extract_exif_data(fixture_path)
        gps_data = exif_extractor.parse_gps_data(exif_dict, image)

        # Should return None for images without GPS
        assert gps_data is None


class TestReverseGeocode:
    """Test reverse geocoding functionality."""

    @pytest.mark.asyncio
    async def test_reverse_geocode_poznań(self):
        """Test reverse geocoding with known coordinates (Poznań, Poland)."""
        # Mock context
        ctx = MagicMock()
        ctx.warning = AsyncMock()

        # Mock aiohttp response
        mock_response_data = {
            "address": {
                "road": "Konstancji Łubieńskiej",
                "neighbourhood": "Marcelin",
                "suburb": "Ławica",
                "city": "Poznań",
                "state": "województwo wielkopolskie",
                "postcode": "60-378",
                "country": "Polska",
                "country_code": "pl",
            },
            "display_name": "Konstancji Łubieńskiej, Marcelin, Ławica, Poznań, województwo wielkopolskie, 60-378, Polska"
        }

        # Properly mock the async context manager chain
        mock_get_cm = AsyncMock()
        mock_get_cm.__aenter__.return_value.status = 200
        mock_get_cm.__aenter__.return_value.json = AsyncMock(return_value=mock_response_data)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value.get = MagicMock(return_value=mock_get_cm)

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            result = await exif_extractor.reverse_geocode(52.408447, 16.867817, 18, ctx)

            assert result is not None
            assert result["road"] == "Konstancji Łubieńskiej"
            assert result["city"] == "Poznań"
            assert result["country"] == "Polska"
            assert result["postcode"] == "60-378"

    @pytest.mark.asyncio
    async def test_reverse_geocode_failure(self):
        """Test reverse geocoding handles API failures gracefully."""
        ctx = MagicMock()
        ctx.warning = AsyncMock()

        # Properly mock the async context manager chain for failure case
        mock_get_cm = AsyncMock()
        mock_get_cm.__aenter__.return_value.status = 500

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value.get = MagicMock(return_value=mock_get_cm)

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            result = await exif_extractor.reverse_geocode(52.408447, 16.867817, 18, ctx)

            assert result is None
            ctx.warning.assert_called_once()


class TestAnalyzeImageMetadataTool:
    """Test the main analyze_image_metadata tool."""

    @pytest.mark.asyncio
    async def test_analyze_single_image_with_gps(self):
        """Test analyzing a single image with GPS data."""
        fixture_path = Path(__file__).parent / "fixtures" / "IMG_5134.heic"

        if not fixture_path.exists():
            pytest.skip("Test fixture IMG_5134.heic not found")

        # Mock context
        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        ctx.warning = AsyncMock()
        ctx.error = AsyncMock()
        ctx.report_progress = AsyncMock()

        # Mock geocoding to avoid hitting real API
        with patch.object(exif_extractor, 'reverse_geocode', return_value={
            "road": "Test Street",
            "city": "Test City",
            "country": "Test Country",
            "display_name": "Test Address"
        }):
            result = await _analyze_image_metadata(
                images=[str(fixture_path)],
                include_address=True,
                zoom_level=18,
                batch_size=10,
                ctx=ctx
            )

        # Verify ToolResult structure
        assert isinstance(result, exif_extractor.ToolResult)
        assert len(result.content) == 1
        assert result.content[0].type == "text"

        # Verify structured content
        metadata = result.structured_content["metadata"]
        assert str(fixture_path) in metadata

        image_data = metadata[str(fixture_path)]
        assert image_data["format"] == "HEIF"
        assert "gps" in image_data
        assert "address" in image_data

        # Verify stats
        stats = result.structured_content["stats"]
        assert stats["total"] == 1
        assert stats["successful"] == 1
        assert stats["with_gps"] == 1
        assert stats["geocoded"] == 1

    @pytest.mark.asyncio
    async def test_analyze_single_image_without_gps(self):
        """Test analyzing a single image without GPS data."""
        fixture_path = Path(__file__).parent / "fixtures" / "IMG_2852.heic"

        if not fixture_path.exists():
            pytest.skip("Test fixture IMG_2852.heic not found")

        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        ctx.warning = AsyncMock()
        ctx.error = AsyncMock()
        ctx.report_progress = AsyncMock()

        result = await _analyze_image_metadata(
            images=[str(fixture_path)],
            include_address=False,
            zoom_level=18,
            batch_size=10,
            ctx=ctx
        )

        # Verify result
        metadata = result.structured_content["metadata"]
        image_data = metadata[str(fixture_path)]

        assert image_data["format"] == "HEIF"
        assert "gps" not in image_data
        assert "address" not in image_data

        # Verify stats
        stats = result.structured_content["stats"]
        assert stats["with_gps"] == 0
        assert stats["geocoded"] == 0

    @pytest.mark.asyncio
    async def test_analyze_multiple_images(self):
        """Test analyzing multiple images in batch."""
        fixture_dir = Path(__file__).parent / "fixtures"
        image_paths = [
            fixture_dir / "IMG_5134.heic",
            fixture_dir / "IMG_2852.heic",
        ]

        # Filter to only existing files
        existing_paths = [str(p) for p in image_paths if p.exists()]

        if len(existing_paths) < 2:
            pytest.skip("Multiple test fixtures not available")

        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        ctx.warning = AsyncMock()
        ctx.error = AsyncMock()
        ctx.report_progress = AsyncMock()

        # Mock geocoding
        with patch.object(exif_extractor, 'reverse_geocode', return_value=None):
            result = await _analyze_image_metadata(
                images=existing_paths,
                include_address=True,
                zoom_level=18,
                batch_size=10,
                ctx=ctx
            )

        # Verify all images processed
        metadata = result.structured_content["metadata"]
        assert len(metadata) == len(existing_paths)

        stats = result.structured_content["stats"]
        assert stats["total"] == len(existing_paths)
        assert stats["successful"] == len(existing_paths)

    @pytest.mark.asyncio
    async def test_analyze_nonexistent_file(self):
        """Test handling of non-existent file."""
        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        ctx.warning = AsyncMock()
        ctx.error = AsyncMock()
        ctx.report_progress = AsyncMock()

        result = await _analyze_image_metadata(
            images=["/nonexistent/path/image.jpg"],
            include_address=False,
            zoom_level=18,
            batch_size=10,
            ctx=ctx
        )

        # Verify error handling
        metadata = result.structured_content["metadata"]
        assert "/nonexistent/path/image.jpg" in metadata
        assert "error" in metadata["/nonexistent/path/image.jpg"]

        stats = result.structured_content["stats"]
        assert stats["failed"] == 1


class TestIntegration:
    """Integration tests with real fixtures."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_workflow_with_real_geocoding(self):
        """Test complete workflow with real Nominatim API."""
        fixture_path = Path(__file__).parent / "fixtures" / "IMG_5134.heic"

        if not fixture_path.exists():
            pytest.skip("Test fixture IMG_5134.heic not found")

        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        ctx.warning = AsyncMock()
        ctx.error = AsyncMock()
        ctx.report_progress = AsyncMock()

        # Run with real geocoding (respects rate limits)
        result = await _analyze_image_metadata(
            images=[str(fixture_path)],
            include_address=True,
            zoom_level=18,
            batch_size=1,
            ctx=ctx
        )

        # Verify complete result
        metadata = result.structured_content["metadata"]
        image_data = metadata[str(fixture_path)]

        assert "gps" in image_data
        assert image_data["gps"]["latitude"] > 0
        assert image_data["gps"]["longitude"] > 0

        # If geocoding succeeded, verify address
        if "address" in image_data:
            assert "country" in image_data["address"]
            # Should be Poland based on coordinates
            assert image_data["address"]["country"] in ["Polska", "Poland"]
