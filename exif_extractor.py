"""
FastMCP EXIF Metadata Extractor Server

Extracts EXIF metadata from images (PNG, JPEG, HEIC) with focus on GPS data.
Automatically reverse geocodes GPS coordinates to street addresses using Nominatim.
"""

import asyncio
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp
import pillow_heif
from fastmcp import FastMCP, Context
from fastmcp.tools.tool import ToolResult
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pydantic import Field


# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

mcp = FastMCP("EXIF Metadata Extractor")

# Nominatim rate limiting (1 request per second)
NOMINATIM_RATE_LIMIT = 1.0  # seconds


def extract_exif_data(image_path: Path) -> tuple[dict[str, Any], Image.Image]:
    """
    Extract all EXIF data from an image file.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (exif_dict, PIL_Image)
    """
    image = Image.open(image_path)
    exif_data = image.getexif()

    if not exif_data:
        return {}, image

    # Convert EXIF tags to readable names
    exif_dict = {}
    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, f"Unknown_{tag_id}")

        # Convert bytes to string or skip binary data
        if isinstance(value, bytes):
            try:
                value = value.decode('utf-8', errors='ignore').strip('\x00')
            except:
                value = f"<binary data: {len(value)} bytes>"

        exif_dict[tag_name] = value

    return exif_dict, image


def parse_gps_data(exif_data: dict[str, Any], image: Image.Image) -> Optional[dict[str, Any]]:
    """
    Extract and parse GPS data from EXIF.

    Args:
        exif_data: EXIF data dictionary from image.getexif()
        image: PIL Image object

    Returns:
        Dictionary with GPS metadata or None if no GPS data
    """
    try:
        # Get GPS IFD (tag 0x8825)
        raw_exif = image.getexif()
        gps_ifd = raw_exif.get_ifd(0x8825)

        if not gps_ifd:
            return None

        # Extract GPS components
        lat = gps_ifd.get(2)  # GPSLatitude
        lat_ref = gps_ifd.get(1)  # GPSLatitudeRef (N/S)
        lon = gps_ifd.get(4)  # GPSLongitude
        lon_ref = gps_ifd.get(3)  # GPSLongitudeRef (E/W)

        if not (lat and lon):
            return None

        # Convert to decimal degrees
        lat_decimal = convert_to_degrees(lat)
        lon_decimal = convert_to_degrees(lon)

        # Apply hemisphere references
        if lat_ref == 'S':
            lat_decimal = -lat_decimal
        if lon_ref == 'W':
            lon_decimal = -lon_decimal

        # Extract additional GPS metadata
        gps_data = {
            "latitude": round(lat_decimal, 6),
            "longitude": round(lon_decimal, 6),
        }

        # Altitude
        altitude = gps_ifd.get(6)  # GPSAltitude
        if altitude is not None:
            gps_data["altitude_meters"] = round(float(altitude), 2)

        # Speed
        speed = gps_ifd.get(13)  # GPSSpeed
        speed_ref = gps_ifd.get(12)  # GPSSpeedRef (K=km/h, M=mph, N=knots)
        if speed is not None:
            speed_value = float(speed)
            # Convert to km/h
            if speed_ref == 'M':
                speed_value *= 1.60934  # mph to km/h
            elif speed_ref == 'N':
                speed_value *= 1.852  # knots to km/h
            gps_data["speed_kmh"] = round(speed_value, 2)

        # Direction/bearing
        img_direction = gps_ifd.get(17)  # GPSImgDirection
        if img_direction is not None:
            gps_data["direction_degrees"] = round(float(img_direction), 2)

        # Timestamp
        gps_date = gps_ifd.get(29)  # GPSDateStamp
        gps_time = gps_ifd.get(7)  # GPSTimeStamp
        if gps_date and gps_time:
            try:
                # Parse GPS timestamp (UTC)
                date_parts = gps_date.split(':')
                time_parts = [float(x) for x in gps_time]
                gps_datetime = datetime(
                    int(date_parts[0]), int(date_parts[1]), int(date_parts[2]),
                    int(time_parts[0]), int(time_parts[1]), int(time_parts[2]),
                    tzinfo=timezone.utc
                )
                gps_data["timestamp_utc"] = gps_datetime.isoformat()
            except:
                pass

        # Positioning error/accuracy
        h_positioning_error = gps_ifd.get(31)  # GPSHPositioningError
        if h_positioning_error is not None:
            gps_data["accuracy_meters"] = round(float(h_positioning_error), 2)

        return gps_data

    except Exception:
        return None


def convert_to_degrees(value: tuple) -> float:
    """
    Convert GPS coordinates from degrees/minutes/seconds to decimal degrees.

    Args:
        value: Tuple of (degrees, minutes, seconds)

    Returns:
        Decimal degrees
    """
    degrees = float(value[0])
    minutes = float(value[1])
    seconds = float(value[2])

    return degrees + (minutes / 60.0) + (seconds / 3600.0)


async def reverse_geocode(
    lat: float,
    lon: float,
    zoom: int,
    ctx: Context
) -> Optional[dict[str, Any]]:
    """
    Reverse geocode coordinates to address using Nominatim (OpenStreetMap).

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        zoom: Zoom level (18=street, 16=area, 14=city)
        ctx: FastMCP context

    Returns:
        Dictionary with address components or None on failure
    """
    base_url = "https://nominatim.openstreetmap.org/reverse"

    params = {
        'lat': lat,
        'lon': lon,
        'format': 'json',
        'zoom': zoom,
        'addressdetails': 1
    }

    headers = {
        'User-Agent': 'MCPExifExtractor/1.0 (FastMCP Server)'
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    address = result.get('address', {})

                    # Extract useful address components
                    return {
                        "road": address.get('road'),
                        "house_number": address.get('house_number'),
                        "neighbourhood": address.get('neighbourhood'),
                        "suburb": address.get('suburb'),
                        "city": address.get('city') or address.get('town') or address.get('village'),
                        "county": address.get('county'),
                        "state": address.get('state'),
                        "postcode": address.get('postcode'),
                        "country": address.get('country'),
                        "country_code": address.get('country_code'),
                        "display_name": result.get('display_name'),
                    }
                else:
                    await ctx.warning(f"Geocoding failed with status {response.status}")
                    return None

    except Exception as e:
        await ctx.warning(f"Geocoding error: {str(e)}")
        return None


@mcp.tool(
    name="analyze_image_metadata",
    description="Extract EXIF metadata from images including GPS coordinates and reverse geocode to addresses",
)
async def analyze_image_metadata(
    images: list[str] = Field(
        min_length=1,
        max_length=50,
        description="List of image file paths (1-50 images)"
    ),
    include_address: bool = Field(
        default=True,
        description="Reverse geocode GPS coordinates to street addresses"
    ),
    zoom_level: int = Field(
        default=18,
        ge=0,
        le=18,
        description="Nominatim zoom level for geocoding (18=street, 16=area, 14=city)"
    ),
    batch_size: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum images to process concurrently"
    ),
    ctx: Context = None,
) -> ToolResult:
    """
    Extract EXIF metadata from images with focus on GPS data and timestamps.

    Processes multiple images in batches with progress reporting. Automatically
    reverse geocodes GPS coordinates to human-readable addresses using OpenStreetMap.
    """
    total = len(images)
    start_time = datetime.now(timezone.utc)

    await ctx.info(f"Starting EXIF extraction for {total} image(s)")

    results = {}
    stats = {
        "total": total,
        "successful": 0,
        "failed": 0,
        "with_gps": 0,
        "geocoded": 0,
    }

    # Process images
    for i, image_path in enumerate(images):
        try:
            await ctx.report_progress(i, total, f"Processing image {i+1}/{total}")

            path = Path(image_path)
            if not path.exists():
                results[image_path] = {"error": "File not found"}
                stats["failed"] += 1
                continue

            # Extract EXIF data
            exif_dict, image = extract_exif_data(path)

            # Build result structure
            result = {
                "format": image.format,
                "size": list(image.size),
                "mode": image.mode,
            }

            # Add basic EXIF metadata
            exif_output = {}
            priority_fields = [
                "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
                "DateTimeDigitized", "Orientation", "XResolution", "YResolution",
                "HostComputer", "LensMake", "LensModel"
            ]

            for field in priority_fields:
                if field in exif_dict:
                    value = exif_dict[field]
                    # Skip GPSInfo tag (handled separately)
                    if field != "GPSInfo":
                        # Convert PIL IFDRational to float for JSON serialization
                        if hasattr(value, '__float__'):
                            value = float(value)
                        exif_output[field.lower()] = value

            if exif_output:
                result["exif"] = exif_output

            # Parse GPS data
            gps_data = parse_gps_data(exif_dict, image)
            if gps_data:
                result["gps"] = gps_data
                stats["with_gps"] += 1

                # Reverse geocode if requested
                if include_address:
                    await ctx.debug(
                        f"Geocoding coordinates: {gps_data['latitude']}, {gps_data['longitude']}"
                    )

                    address = await reverse_geocode(
                        gps_data["latitude"],
                        gps_data["longitude"],
                        zoom_level,
                        ctx
                    )

                    if address:
                        # Remove None values for cleaner output
                        result["address"] = {k: v for k, v in address.items() if v is not None}
                        stats["geocoded"] += 1

                    # Respect Nominatim rate limit
                    await asyncio.sleep(NOMINATIM_RATE_LIMIT)

            results[image_path] = result
            stats["successful"] += 1

        except Exception as e:
            await ctx.error(f"Failed to process {image_path}: {str(e)}")
            results[image_path] = {"error": str(e)}
            stats["failed"] += 1

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    await ctx.report_progress(total, total, "Metadata extraction complete")

    # Create human-readable summary
    summary_lines = [
        f"✓ Processed {stats['successful']}/{total} image(s) in {duration:.2f}s",
        f"  • {stats['with_gps']} image(s) with GPS data",
    ]

    if include_address:
        summary_lines.append(f"  • {stats['geocoded']} location(s) geocoded")

    if stats["failed"] > 0:
        summary_lines.append(f"  • {stats['failed']} failed")

    summary_lines.append("\nResults:")

    for img_path, data in results.items():
        display_path = img_path if len(img_path) < 60 else f"...{img_path[-57:]}"
        summary_lines.append(f"\n{display_path}:")

        if "error" in data:
            summary_lines.append(f"  ❌ Error: {data['error']}")
            continue

        summary_lines.append(f"  Format: {data.get('format', 'Unknown')}, Size: {data['size'][0]}x{data['size'][1]}")

        if "gps" in data:
            gps = data["gps"]
            summary_lines.append(
                f"  📍 GPS: {gps['latitude']}, {gps['longitude']}"
            )

            if "altitude_meters" in gps:
                summary_lines.append(f"     Altitude: {gps['altitude_meters']}m")

            if "address" in data:
                addr = data["address"]
                location_parts = []
                if addr.get("road"):
                    location_parts.append(addr["road"])
                if addr.get("city"):
                    location_parts.append(addr["city"])
                if addr.get("country"):
                    location_parts.append(addr["country"])

                if location_parts:
                    summary_lines.append(f"     📍 {', '.join(location_parts)}")
        else:
            summary_lines.append("  (No GPS data)")

    summary = "\n".join(summary_lines)

    return ToolResult(
        content=[{"type": "text", "text": summary}],
        structured_content={
            "metadata": results,
            "stats": {
                **stats,
                "duration_seconds": round(duration, 2),
            },
            "processing_info": {
                "processed_at": start_time.isoformat(),
                "include_address": include_address,
                "zoom_level": zoom_level,
                "batch_size": batch_size,
            }
        }
    )
