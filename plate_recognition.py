"""
FastMCP Plate Recognition Server

Analyzes traffic violation photos to extract license plates and identify
which vehicle is most likely committing an offense.
"""

import asyncio
import base64
import io
import json
import os
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from fastmcp import FastMCP, Context
from fastmcp.tools.tool import ToolResult
from PIL import Image
from pydantic import Field

# Try to import HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False


mcp = FastMCP("Plate Recognition")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Only configure if API key is available (allows importing for tests)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Default model for vision tasks
DEFAULT_MODEL = "gemini-flash-latest"


def optimize_image(
    image: Image.Image,
    max_dimension: int = 1920,
    quality: int = 85
) -> tuple[bytes, str]:
    """
    Optimize image for Gemini API to reduce token usage.

    Args:
        image: PIL Image object
        max_dimension: Maximum width or height
        quality: JPEG compression quality (1-100)

    Returns:
        Tuple of (optimized_image_bytes, mime_type)
    """
    width, height = image.size

    # Only resize if image exceeds max dimensions
    if width > max_dimension or height > max_dimension:
        # Calculate new size maintaining aspect ratio
        ratio = min(max_dimension / width, max_dimension / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert to RGB if necessary (for JPEG)
    if image.mode in ('RGBA', 'LA', 'L', 'P'):
        # Create white background for transparency
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA' or image.mode == 'LA':
            background.paste(image, mask=image.split()[-1])
        else:
            background.paste(image)
        image = background

    # Save optimized image to bytes
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=quality, optimize=True)
    return output.getvalue(), 'image/jpeg'


async def load_image(image_path: str, ctx: Context) -> tuple[bytes, str]:
    """
    Load image from file path.

    Args:
        image_path: Path to image file
        ctx: FastMCP context for logging

    Returns:
        Tuple of (image_bytes, mime_type)
    """
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Image file not found: {image_path}")

    with open(path, 'rb') as f:
        image_bytes = f.read()

    # Load with PIL to optimize
    image = Image.open(io.BytesIO(image_bytes))
    await ctx.debug(f"Loaded image from file: {path.name}, size: {image.size}")

    # Optimize for Gemini
    return optimize_image(image)


def create_plate_recognition_prompt() -> str:
    """
    Create prompt for plate recognition and violation detection.
    """
    return """You are analyzing a traffic photo from a pedestrian's perspective to identify vehicles and potential traffic violations.

Your tasks:
1. Identify ALL visible license plate numbers in the image (even partially visible ones)
2. Determine which vehicle (if any) is MOST LIKELY committing a traffic violation such as:
   - Parking on a sidewalk
   - Blocking a pedestrian crosswalk
   - Illegal parking in restricted zones
   - Blocking access for pedestrians
   - Other clear traffic violations from a pedestrian's perspective
3. Provide ONE sentence explaining your reasoning

Important:
- Focus on violations that affect pedestrians
- If no clear violation is visible, set violation_vehicle to null
- Be specific about which plate belongs to the violating vehicle
- Extract plates as they appear (with spaces/dashes if visible)

Respond ONLY with valid JSON in this exact format (no additional text):
{
  "plates": ["PLATE1", "PLATE2"],
  "violation_vehicle": "PLATE1",
  "reasoning": "Vehicle PLATE1 is parked on the sidewalk, blocking pedestrian access."
}

If no plates are visible, return:
{
  "plates": [],
  "violation_vehicle": null,
  "reasoning": "No license plates are visible in this image."
}"""


@mcp.tool(
    name="recognize_plates",
    description="Analyze traffic photo to extract license plates and identify which vehicle is most likely committing a violation"
)
async def recognize_plates(
    image_path: str = Field(
        description="Path to the traffic violation photo"
    ),
    model: Optional[str] = Field(
        default=None,
        description="Gemini model to use. Defaults to 'gemini-flash-latest'"
    ),
    ctx: Context = None,
) -> ToolResult:
    """
    Analyze a traffic photo to extract license plates and identify violations.

    This tool uses Gemini's vision API to:
    1. Extract all visible license plate numbers
    2. Identify which vehicle is most likely committing a traffic violation
    3. Provide reasoning for the determination

    Args:
        image_path: Path to the image file
        model: Optional Gemini model name (defaults to gemini-flash-latest)
        ctx: FastMCP context for logging

    Returns:
        ToolResult with detected plates, violation vehicle, and reasoning
    """
    model_name = model or DEFAULT_MODEL

    # Check for API key at runtime
    if not GEMINI_API_KEY:
        error_msg = "GEMINI_API_KEY environment variable is required"
        await ctx.error(error_msg)
        return ToolResult(
            content=[{"type": "text", "text": error_msg}],
            structured_content={"error": "Missing API key"}
        )

    await ctx.info(f"Analyzing image for plate recognition: {image_path}")

    try:
        # Load and optimize image
        image_bytes, mime_type = await load_image(image_path, ctx)
        await ctx.debug(f"Image loaded and optimized, mime_type: {mime_type}")

        # Create prompt
        prompt = create_plate_recognition_prompt()

        # Create model instance
        model_instance = genai.GenerativeModel(model_name)

        # Generate analysis using Gemini
        await ctx.info("Sending image to Gemini for analysis...")
        response = await asyncio.to_thread(
            model_instance.generate_content,
            [
                prompt,
                {
                    "mime_type": mime_type,
                    "data": base64.b64encode(image_bytes).decode()
                }
            ],
            generation_config={
                "temperature": 0.2,  # Lower temperature for more consistent JSON output
                "top_p": 0.8,
                "top_k": 40,
            }
        )

        # Parse response
        response_text = response.text.strip()
        await ctx.debug(f"Raw response: {response_text}")

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith("```"):
            response_text = response_text[3:]  # Remove ```
        if response_text.endswith("```"):
            response_text = response_text[:-3]  # Remove trailing ```
        response_text = response_text.strip()

        # Parse JSON
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            await ctx.error(f"Failed to parse JSON response: {e}")
            await ctx.error(f"Response text: {response_text}")
            return ToolResult(
                content=[{"type": "text", "text": f"Failed to parse Gemini response as JSON: {e}"}],
                structured_content={"error": "JSON parsing failed", "raw_response": response_text}
            )

        # Validate response structure
        if not isinstance(result.get("plates"), list):
            result["plates"] = []

        # Create human-readable summary
        plates_found = len(result["plates"])
        violation_vehicle = result.get("violation_vehicle")
        reasoning = result.get("reasoning", "No reasoning provided")

        summary_lines = [
            f"🚗 Plate Recognition Analysis",
            f"",
            f"Plates detected: {plates_found}",
        ]

        if plates_found > 0:
            for i, plate in enumerate(result["plates"], 1):
                summary_lines.append(f"  {i}. {plate}")

        summary_lines.append("")

        if violation_vehicle:
            summary_lines.append(f"⚠️  Violation detected: {violation_vehicle}")
            summary_lines.append(f"Reasoning: {reasoning}")
        else:
            summary_lines.append(f"✓ No clear violation detected")
            summary_lines.append(f"Note: {reasoning}")

        summary = "\n".join(summary_lines)

        return ToolResult(
            content=[{"type": "text", "text": summary}],
            structured_content={
                "plates": result["plates"],
                "violation_vehicle": violation_vehicle,
                "reasoning": reasoning,
                "metadata": {
                    "image_path": image_path,
                    "model": model_name,
                    "plates_count": plates_found
                }
            }
        )

    except FileNotFoundError:
        error_msg = f"Image file not found: {image_path}"
        await ctx.error(error_msg)
        return ToolResult(
            content=[{"type": "text", "text": error_msg}],
            structured_content={"error": "File not found", "path": image_path}
        )

    except Exception as e:
        error_msg = f"Failed to analyze image: {str(e)}"
        await ctx.error(error_msg)
        return ToolResult(
            content=[{"type": "text", "text": error_msg}],
            structured_content={"error": str(e)}
        )
