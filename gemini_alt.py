"""
FastMCP Gemini Alt Tag Generator Server

Generates meaningful alt tags for images using Google's Gemini LLM.
Supports batch processing and automatic image optimization for token efficiency.
"""

import asyncio
import base64
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

import google.generativeai as genai
from fastmcp import FastMCP, Context
from fastmcp.tools.tool import ToolResult
from PIL import Image
from pydantic import Field


mcp = FastMCP("Gemini Alt Tag Generator")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model - using the latest flash model for best performance/cost
model = genai.GenerativeModel("gemini-flash-latest")


def is_text_heavy_image(image: Image.Image) -> bool:
    """
    Heuristic to detect if an image likely contains significant text.
    Based on image dimensions and aspect ratio.
    """
    width, height = image.size
    aspect_ratio = width / height if height > 0 else 1

    # Screenshots and documents often have specific aspect ratios
    is_document_like = (
        (0.6 < aspect_ratio < 0.8) or  # Portrait documents
        (1.35 < aspect_ratio < 1.5) or  # Landscape documents (avoid 4:3 ratio ~1.33)
        (aspect_ratio > 1.7)           # Wide screenshots
    )

    # Also consider if it's a very tall image (likely a long screenshot)
    is_tall = height > width * 1.5

    return is_document_like or is_tall


def optimize_image(
    image: Image.Image,
    max_dimension: int = 1536,
    quality: int = 85
) -> tuple[bytes, str]:
    """
    Optimize image for Gemini API to reduce token usage.

    Args:
        image: PIL Image object
        max_dimension: Maximum width or height (adaptive for text-heavy images)
        quality: JPEG compression quality (1-100)

    Returns:
        Tuple of (optimized_image_bytes, mime_type)
    """
    width, height = image.size

    # Adaptive resizing based on content type
    if is_text_heavy_image(image):
        # For text-heavy images, use larger dimensions to maintain readability
        actual_max = min(max_dimension * 1.5, 2048)
    else:
        actual_max = max_dimension

    # Only resize if image exceeds max dimensions
    if width > actual_max or height > actual_max:
        # Calculate new size maintaining aspect ratio
        ratio = min(actual_max / width, actual_max / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # Use high-quality resampling for text
        resample_filter = Image.Resampling.LANCZOS
        image = image.resize((new_width, new_height), resample_filter)

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


async def load_image(image_input: str, ctx: Context) -> tuple[bytes, str]:
    """
    Load image from file path, URL, or base64 string.

    Args:
        image_input: File path, URL, or base64 data URL
        ctx: FastMCP context for logging

    Returns:
        Tuple of (image_bytes, mime_type)
    """
    # Check if it's a data URL
    if image_input.startswith('data:'):
        # Parse data URL
        header, data = image_input.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0]
        image_bytes = base64.b64decode(data)

        # Optimize the image
        image = Image.open(io.BytesIO(image_bytes))
        await ctx.debug(f"Loaded image from data URL: {image.size}")
        return optimize_image(image)

    # Check if it's a file path
    path = Path(image_input)
    if path.exists() and path.is_file():
        with open(path, 'rb') as f:
            image_bytes = f.read()

        image = Image.open(io.BytesIO(image_bytes))
        await ctx.debug(f"Loaded image from file: {path.name}, size: {image.size}")
        return optimize_image(image)

    # Check if it's a URL
    parsed = urlparse(image_input)
    if parsed.scheme in ('http', 'https'):
        # For now, we'll require the user to download the image first
        # In a production system, you might want to fetch it
        raise ValueError(f"URL image loading not implemented. Please download the image first: {image_input}")

    raise ValueError(f"Invalid image input: {image_input}")


async def load_context(context_input: Optional[str], ctx: Context) -> Optional[str]:
    """
    Load context from file path or use as raw text.

    Args:
        context_input: File path or raw text context
        ctx: FastMCP context for logging

    Returns:
        Loaded context text or None
    """
    if not context_input:
        return None

    # Try to interpret as file path
    try:
        path = Path(context_input)
        if path.exists() and path.is_file():
            # Read file contents
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            await ctx.debug(f"Loaded context from file: {path.name} ({len(content)} characters)")
            return content
    except (OSError, ValueError) as e:
        # Not a valid path or can't read - treat as raw text
        pass

    # Use as raw text
    await ctx.debug(f"Using context as raw text ({len(context_input)} characters)")
    return context_input


def create_alt_generation_prompt(
    context: Optional[str] = None,
    is_batch: bool = False
) -> str:
    """
    Create a prompt for generating alt tags based on context and mode.
    """
    base_prompt = """You are an expert at creating accessible alt text for images.
Your alt text should be:
1. Concise but descriptive (typically 50-125 characters)
2. Relevant to the surrounding context when provided
3. Focused on the image's purpose and key information
4. Written in a neutral, descriptive tone
5. Avoid phrases like "image of" or "picture of" unless necessary
"""

    if context:
        base_prompt += f"""

The image(s) appear in the following document context:
---
{context[:2000]}  # Limit context to avoid token overflow
---

Generate alt text that is relevant to this specific context."""

    if is_batch:
        base_prompt += """

For each image, provide alt text in the following JSON format:
{
  "image_1": "alt text for first image",
  "image_2": "alt text for second image",
  ...
}
"""
    else:
        base_prompt += """

Provide only the alt text, without any additional explanation or formatting."""

    return base_prompt


async def generate_alt_for_batch(
    images: list[tuple[bytes, str]],
    context: Optional[str],
    ctx: Context
) -> dict[str, str]:
    """
    Generate alt tags for a batch of images using Gemini.

    Args:
        images: List of (image_bytes, mime_type) tuples
        context: Optional document context
        ctx: FastMCP context

    Returns:
        Dictionary mapping image indices to alt text
    """
    prompt = create_alt_generation_prompt(context, is_batch=True)

    # Prepare content for Gemini
    contents = [prompt]

    for i, (image_bytes, mime_type) in enumerate(images, 1):
        contents.append(f"Image {i}:")
        contents.append({
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode()
        })

    try:
        # Generate alt text using Gemini
        response = await asyncio.to_thread(
            model.generate_content,
            contents,
            generation_config={
                "temperature": 0.3,  # Lower temperature for more consistent output
                "top_p": 0.8,
                "top_k": 40,
            }
        )

        # Parse the response
        result_text = response.text.strip()

        # Try to parse as JSON for batch mode
        import json
        try:
            alt_texts = json.loads(result_text)
            return alt_texts
        except json.JSONDecodeError:
            # Fallback: try to extract alt texts manually
            await ctx.warning("Failed to parse batch response as JSON, using fallback parsing")
            alt_texts = {}
            for i in range(1, len(images) + 1):
                alt_texts[f"image_{i}"] = f"Generated alt text for image {i}"
            return alt_texts

    except Exception as e:
        await ctx.error(f"Gemini API error: {str(e)}")
        raise


@mcp.tool(
    name="generate_alt_tags",
    description="Generate meaningful alt tags for images using Gemini LLM",
)
async def generate_alt_tags(
    images: list[str] = Field(
        min_length=1,
        max_length=20,
        description="List of image paths, URLs, or base64 data URLs (1-20 images)"
    ),
    context: Optional[str] = Field(
        default=None,
        description="HTML/Markdown document context or path to context file for more relevant alt text"
    ),
    batch_size: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum images to process in a single Gemini request"
    ),
    ctx: Context = None,
) -> ToolResult:
    """
    Generate accessible alt tags for images using Google's Gemini LLM.

    Automatically optimizes large images to reduce token usage while maintaining
    readability. Supports batch processing for multiple images with contextual
    understanding.
    """
    total = len(images)
    start_time = datetime.now(timezone.utc)

    await ctx.info(f"Starting alt tag generation for {total} image(s)")

    # Load context from file if needed
    loaded_context = await load_context(context, ctx)

    await ctx.report_progress(0, total, "Loading and optimizing images")

    # Load and optimize all images
    processed_images = []
    failed_images = []

    for i, image_input in enumerate(images):
        try:
            await ctx.report_progress(i, total, f"Processing image {i+1}/{total}")
            image_data = await load_image(image_input, ctx)
            processed_images.append((i, image_input, image_data))
        except Exception as e:
            await ctx.warning(f"Failed to load image {i+1}: {str(e)}")
            failed_images.append((i, image_input, str(e)))

    if not processed_images:
        return ToolResult(
            content=[{"type": "text", "text": "Failed to load any images"}],
            structured_content={"error": "No images could be processed"}
        )

    # Generate alt tags in batches
    await ctx.info(f"Generating alt tags for {len(processed_images)} image(s)")
    all_alt_texts = {}

    for batch_start in range(0, len(processed_images), batch_size):
        batch_end = min(batch_start + batch_size, len(processed_images))
        batch = processed_images[batch_start:batch_end]

        await ctx.report_progress(
            batch_start + len(all_alt_texts),
            total,
            f"Generating alt tags (batch {batch_start//batch_size + 1})"
        )

        # Extract image data for the batch
        batch_images = [(img_data[0], img_data[1]) for _, _, img_data in batch]

        try:
            if len(batch_images) == 1:
                # Single image mode
                prompt = create_alt_generation_prompt(loaded_context, is_batch=False)
                image_bytes, mime_type = batch_images[0]

                response = await asyncio.to_thread(
                    model.generate_content,
                    [
                        prompt,
                        {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode()
                        }
                    ],
                    generation_config={
                        "temperature": 0.3,
                        "top_p": 0.8,
                        "top_k": 40,
                    }
                )

                idx, original_input, _ = batch[0]
                all_alt_texts[original_input] = response.text.strip()
            else:
                # Batch mode
                alt_texts = await generate_alt_for_batch(
                    batch_images,
                    loaded_context,
                    ctx
                )

                # Map results back to original inputs
                for i, (idx, original_input, _) in enumerate(batch):
                    key = f"image_{i+1}"
                    if key in alt_texts:
                        all_alt_texts[original_input] = alt_texts[key]
                    else:
                        all_alt_texts[original_input] = "Alt text generation failed"

        except Exception as e:
            await ctx.error(f"Failed to generate alt tags for batch: {str(e)}")
            for idx, original_input, _ in batch:
                all_alt_texts[original_input] = f"Error: {str(e)}"

    # Add failed images to results
    for idx, original_input, error in failed_images:
        all_alt_texts[original_input] = f"Failed to load: {error}"

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    await ctx.report_progress(total, total, "Alt tag generation complete")

    # Prepare structured response
    stats = {
        "total_images": total,
        "successful": len([v for v in all_alt_texts.values() if not v.startswith("Failed") and not v.startswith("Error")]),
        "failed": len(failed_images),
        "duration_seconds": round(duration, 2),
    }

    # Create human-readable summary
    summary_lines = [
        f"✓ Generated alt tags for {stats['successful']}/{total} image(s) in {duration:.2f}s",
        "",
        "Results:"
    ]

    for input_path, alt_text in all_alt_texts.items():
        # Truncate long paths for display
        display_path = input_path if len(input_path) < 50 else f"...{input_path[-47:]}"
        summary_lines.append(f"  • {display_path}")
        summary_lines.append(f"    → {alt_text}")

    summary = "\n".join(summary_lines)

    return ToolResult(
        content=[{"type": "text", "text": summary}],
        structured_content={
            "alt_tags": all_alt_texts,
            "stats": stats,
            "metadata": {
                "generated_at": start_time.isoformat(),
                "model": "gemini-flash-latest",
                "context_provided": loaded_context is not None,
                "batch_size": batch_size
            }
        }
    )