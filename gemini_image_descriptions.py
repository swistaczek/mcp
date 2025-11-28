"""
Gemini Image Description Generator - generates alt text and accessible descriptions for images/GIFs.
"""

import asyncio
import base64
import io
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

import google.generativeai as genai
from google import genai as genai_new
from google.genai import types as genai_types
from fastmcp import FastMCP, Context
from fastmcp.tools.tool import ToolResult
from PIL import Image
from pydantic import Field


mcp = FastMCP("Image Descriptions")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

# Default model for best performance/cost
DEFAULT_MODEL = "gemini-flash-latest"

# GIF processing timeout (seconds)
GIF_PROCESSING_TIMEOUT = 60


def is_gif_file(file_path: Path) -> bool:
    """
    Check if a file is a GIF by extension and magic bytes.

    Args:
        file_path: Path to the file

    Returns:
        True if the file is a GIF
    """
    if file_path.suffix.lower() != '.gif':
        return False

    # Verify magic bytes (GIF87a or GIF89a)
    try:
        with open(file_path, 'rb') as f:
            header = f.read(6)
        return header in (b'GIF87a', b'GIF89a')
    except (OSError, IOError):
        return False


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


async def convert_gif_to_mp4(gif_path: Path, ctx: Context) -> Path:
    """
    Convert a GIF to MP4 using FFmpeg for Gemini video processing.

    Args:
        gif_path: Path to the GIF file
        ctx: FastMCP context for logging

    Returns:
        Path to the converted MP4 file (temporary file, caller must clean up)

    Raises:
        RuntimeError: If FFmpeg is not available or conversion fails
    """
    if not check_ffmpeg_available():
        raise RuntimeError(
            "FFmpeg is required for GIF processing but was not found. "
            "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

    # Create temporary file for output
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = Path(tmp_file.name)
    tmp_file.close()

    # FFmpeg command for GIF to MP4 conversion
    # -movflags faststart: Optimize for streaming
    # -pix_fmt yuv420p: H.264 compatible pixel format
    # -vf scale=...: Ensure even dimensions (required for H.264)
    cmd = [
        "ffmpeg",
        "-i", str(gif_path),
        "-movflags", "faststart",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-y",  # Overwrite output file
        str(output_path)
    ]

    await ctx.debug(f"Converting GIF to MP4: {gif_path.name}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg conversion failed: {stderr.decode()}")

    original_size = gif_path.stat().st_size / 1024
    converted_size = output_path.stat().st_size / 1024
    await ctx.debug(
        f"GIF converted: {original_size:.1f}KB -> {converted_size:.1f}KB "
        f"({100 - (converted_size/original_size*100):.0f}% reduction)"
    )

    return output_path


async def upload_video_to_gemini(
    video_path: Path,
    ctx: Context,
    timeout: int = GIF_PROCESSING_TIMEOUT
) -> Any:
    """
    Upload a video to Gemini using the File API.

    Args:
        video_path: Path to the video file
        ctx: FastMCP context for logging
        timeout: Maximum time to wait for processing (seconds)

    Returns:
        Gemini File object reference for use in generate_content

    Raises:
        RuntimeError: If upload or processing fails
    """
    # Use the new genai client for File API
    client = genai_new.Client(api_key=GEMINI_API_KEY)

    await ctx.debug(f"Uploading video to Gemini: {video_path.name}")

    # Upload the video
    video_file = await asyncio.to_thread(
        client.files.upload,
        file=str(video_path),
        config={"display_name": video_path.name}
    )

    await ctx.debug(f"Upload complete: {video_file.name}, waiting for processing...")

    # Wait for processing with timeout
    start_time = asyncio.get_event_loop().time()
    while video_file.state.name == "PROCESSING":
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            raise RuntimeError(
                f"Video processing timed out after {timeout}s. "
                "The GIF may be too large or complex."
            )

        await asyncio.sleep(2)
        video_file = await asyncio.to_thread(client.files.get, name=video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Video processing failed with state: {video_file.state.name}")

    await ctx.debug(f"Video ready: {video_file.name}")
    return video_file


def create_gif_description_prompt(
    context: Optional[str] = None,
    description_type: str = "alt"
) -> str:
    """
    Create a prompt for generating alt text for GIF/animation content.

    Args:
        context: Optional document context
        description_type: "alt" for concise (50-125 chars), "description" for detailed

    Returns:
        Prompt string for Gemini
    """
    if description_type == "description":
        base_prompt = """You are an expert at creating detailed animation descriptions for visually impaired users.
Your descriptions will be read aloud by screen readers and assistive technology.

Your descriptions should:
1. Be detailed and comprehensive (150-300 characters)
2. Describe the sequence of actions and movements step by step
3. Mention timing and pacing (fast, slow, smooth, abrupt)
4. Include visual details like colors, UI elements, and text shown
5. Describe any buttons, menus, or interface elements that appear
6. Explain the purpose or outcome of the demonstrated action
7. Be written in clear, natural language as if describing to a friend who cannot see
"""
    else:
        base_prompt = """You are an expert at creating accessible alt text for animations and GIFs.
Your alt text should be:
1. Concise but descriptive (typically 50-125 characters)
2. Focus on the action or motion being shown
3. Describe key visual elements and their changes
4. Capture the animation's purpose or mood
5. Avoid phrases like "GIF of" or "animation showing" unless necessary
"""

    if context:
        base_prompt += f"""

The animation appears in the following document context:
---
{context}
---

Generate alt text that is relevant to this specific context."""

    base_prompt += """

Provide only the alt text, without any additional explanation or formatting."""

    return base_prompt


async def generate_description_for_gif(
    video_file: Any,
    context: Optional[str],
    model_name: str,
    ctx: Context,
    description_type: str = "alt"
) -> str:
    """
    Generate a description for a GIF (uploaded as video) using Gemini.

    Args:
        video_file: Gemini File object from upload_video_to_gemini
        context: Optional document context
        model_name: Gemini model to use
        ctx: FastMCP context
        description_type: "alt" for concise, "description" for detailed

    Returns:
        Generated alt text for the GIF
    """
    client = genai_new.Client(api_key=GEMINI_API_KEY)
    prompt = create_gif_description_prompt(context, description_type)

    await ctx.debug(f"Generating description for GIF using {model_name}")

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model_name,
        contents=[
            genai_types.Part.from_uri(file_uri=video_file.uri, mime_type="video/mp4"),
            prompt
        ],
        config={
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
        }
    )

    return response.text.strip()


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
    is_batch: bool = False,
    description_type: str = "alt"
) -> str:
    """
    Create a prompt for generating alt tags based on context and mode.

    Args:
        context: Optional document context
        is_batch: Whether to generate batch JSON format
        description_type: "alt" for concise (50-125 chars), "description" for detailed
    """
    if description_type == "description":
        base_prompt = """You are an expert at creating detailed image descriptions for visually impaired users.
Your descriptions will be read aloud by screen readers and assistive technology.

Your descriptions should:
1. Be detailed and comprehensive (150-300 characters)
2. Describe spatial layout (left, right, foreground, background)
3. Include colors, textures, and visual details
4. Mention any text visible in the image
5. Describe people's actions, expressions, and positioning
6. Convey the mood, context, and purpose of the image
7. Be written in clear, natural language as if describing to a friend who cannot see
"""
    else:
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
{context}
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
    model_name: str,
    ctx: Context,
    description_type: str = "alt"
) -> dict[str, str]:
    """
    Generate alt tags for a batch of images using Gemini.

    Args:
        images: List of (image_bytes, mime_type) tuples
        context: Optional document context
        model_name: Gemini model to use
        ctx: FastMCP context
        description_type: "alt" for concise, "description" for detailed

    Returns:
        Dictionary mapping image indices to alt text
    """
    prompt = create_alt_generation_prompt(context, is_batch=True, description_type=description_type)

    # Prepare content for Gemini
    contents = [prompt]

    for i, (image_bytes, mime_type) in enumerate(images, 1):
        contents.append(f"Image {i}:")
        contents.append({
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode()
        })

    try:
        # Create model instance
        model = genai.GenerativeModel(model_name)

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
    name="generate_image_descriptions",
    description="Generate meaningful descriptions for images and GIFs using Gemini LLM",
)
async def generate_image_descriptions(
    images: list[str] = Field(
        min_length=1,
        max_length=20,
        description="List of image/GIF paths, URLs, or base64 data URLs (1-20 items). GIFs require FFmpeg."
    ),
    type: str = Field(
        default="alt",
        description="Description type: 'alt' for concise alt text (50-125 chars), 'description' for detailed accessible descriptions optimized for screen readers (150-300 chars)"
    ),
    context: Optional[str] = Field(
        default=None,
        description="HTML/Markdown document context or path to context file for more relevant descriptions"
    ),
    batch_size: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum images to process in a single Gemini request"
    ),
    model: Optional[str] = Field(
        default=None,
        description="Gemini model to use (e.g., 'gemini-pro', 'gemini-flash-latest'). Defaults to 'gemini-flash-latest'"
    ),
    ctx: Context = None,
) -> ToolResult:
    """
    Generate accessible descriptions for images and GIFs using Google's Gemini LLM.

    Supports two description types:
    - "alt": Concise alt text (50-125 chars) - suitable for HTML alt attributes
    - "description": Detailed descriptions (150-300 chars) - optimized for screen readers

    Automatically optimizes large images to reduce token usage while maintaining
    readability. Supports batch processing for multiple images with contextual
    understanding.

    GIF support:
    - GIFs are converted to MP4 using FFmpeg and processed as video
    - Requires FFmpeg to be installed on the system
    - GIFs are processed individually (not batched with images)
    """
    total = len(images)
    start_time = datetime.now(timezone.utc)
    model_name = model or DEFAULT_MODEL
    description_type = type  # Alias to avoid shadowing builtin

    # Validate description type
    if description_type not in ("alt", "description"):
        return ToolResult(
            content=[{"type": "text", "text": f"Invalid type '{description_type}'. Must be 'alt' or 'description'."}],
            structured_content={"error": f"Invalid type: {description_type}"}
        )

    type_label = "alt text" if description_type == "alt" else "detailed descriptions"
    await ctx.info(f"Starting {type_label} generation for {total} image(s) using model: {model_name}")

    # Load context from file if needed
    loaded_context = await load_context(context, ctx)

    await ctx.report_progress(0, total, "Loading and optimizing images")

    # Separate GIFs from regular images
    processed_images = []
    gif_inputs = []
    failed_images = []

    for i, image_input in enumerate(images):
        try:
            await ctx.report_progress(i, total, f"Processing image {i+1}/{total}")

            # Check if it's a GIF file
            path = Path(image_input)
            if path.exists() and path.is_file() and is_gif_file(path):
                await ctx.info(f"Detected GIF: {path.name}")
                gif_inputs.append((i, image_input, path))
            else:
                # Regular image processing
                image_data = await load_image(image_input, ctx)
                processed_images.append((i, image_input, image_data))
        except Exception as e:
            await ctx.warning(f"Failed to load image {i+1}: {str(e)}")
            failed_images.append((i, image_input, str(e)))

    if not processed_images and not gif_inputs:
        return ToolResult(
            content=[{"type": "text", "text": "Failed to load any images or GIFs"}],
            structured_content={"error": "No images could be processed"}
        )

    # Generate descriptions in batches
    if processed_images:
        await ctx.info(f"Generating {type_label} for {len(processed_images)} image(s)")
    all_alt_texts = {}

    for batch_start in range(0, len(processed_images), batch_size):
        batch_end = min(batch_start + batch_size, len(processed_images))
        batch = processed_images[batch_start:batch_end]

        await ctx.report_progress(
            batch_start + len(all_alt_texts),
            total,
            f"Generating descriptions (batch {batch_start//batch_size + 1})"
        )

        # Extract image data for the batch
        batch_images = [(img_data[0], img_data[1]) for _, _, img_data in batch]

        try:
            if len(batch_images) == 1:
                # Single image mode
                prompt = create_alt_generation_prompt(loaded_context, is_batch=False, description_type=description_type)
                image_bytes, mime_type = batch_images[0]

                # Create model instance for single image
                model_instance = genai.GenerativeModel(model_name)

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
                    model_name,
                    ctx,
                    description_type
                )

                # Map results back to original inputs
                for i, (idx, original_input, _) in enumerate(batch):
                    key = f"image_{i+1}"
                    if key in alt_texts:
                        all_alt_texts[original_input] = alt_texts[key]
                    else:
                        all_alt_texts[original_input] = "Alt text generation failed"

        except Exception as e:
            await ctx.error(f"Failed to generate descriptions for batch: {str(e)}")
            for idx, original_input, _ in batch:
                all_alt_texts[original_input] = f"Error: {str(e)}"

    # Process GIFs (each GIF is processed individually due to video upload requirement)
    if gif_inputs:
        await ctx.info(f"Processing {len(gif_inputs)} GIF(s) via video pipeline")
        temp_files = []  # Track temp files for cleanup

        for idx, original_input, gif_path in gif_inputs:
            mp4_path = None
            try:
                await ctx.report_progress(
                    len(processed_images) + gif_inputs.index((idx, original_input, gif_path)),
                    total,
                    f"Processing GIF: {gif_path.name}"
                )

                # Convert GIF to MP4
                mp4_path = await convert_gif_to_mp4(gif_path, ctx)
                temp_files.append(mp4_path)

                # Upload to Gemini File API
                video_file = await upload_video_to_gemini(mp4_path, ctx)

                # Generate description
                alt_text = await generate_description_for_gif(
                    video_file,
                    loaded_context,
                    model_name,
                    ctx,
                    description_type
                )
                all_alt_texts[original_input] = alt_text

            except Exception as e:
                await ctx.error(f"Failed to process GIF {gif_path.name}: {str(e)}")
                all_alt_texts[original_input] = f"Error: {str(e)}"
                failed_images.append((idx, original_input, str(e)))

        # Cleanup temp files
        for temp_file in temp_files:
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass  # Best effort cleanup

    # Add failed images to results
    for idx, original_input, error in failed_images:
        if original_input not in all_alt_texts:  # Don't overwrite existing error messages
            all_alt_texts[original_input] = f"Failed to load: {error}"

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    await ctx.report_progress(total, total, "Description generation complete")

    # Prepare structured response
    successful_count = len([v for v in all_alt_texts.values() if not v.startswith("Failed") and not v.startswith("Error")])
    stats = {
        "total_images": total,
        "images": len(processed_images),
        "gifs": len(gif_inputs),
        "successful": successful_count,
        "failed": total - successful_count,
        "duration_seconds": round(duration, 2),
    }

    # Create human-readable summary
    summary_lines = [
        f"✓ Generated {type_label} for {stats['successful']}/{total} image(s) in {duration:.2f}s",
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
            "descriptions": all_alt_texts,
            "stats": stats,
            "metadata": {
                "generated_at": start_time.isoformat(),
                "model": model_name,
                "context_provided": loaded_context is not None,
                "batch_size": batch_size,
                "description_type": description_type
            }
        }
    )