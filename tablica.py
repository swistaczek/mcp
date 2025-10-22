"""
FastMCP Server for tablica-rejestracyjna.pl

Polish license plate reporting website integration.
Allows fetching comments and submitting complaints about vehicles.
"""

import asyncio
import base64
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlencode

import aiohttp
from bs4 import BeautifulSoup
from fastmcp import FastMCP, Context
from fastmcp.tools.tool import ToolResult
from PIL import Image
from pydantic import Field

# Try to import pillow-heif for HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

mcp = FastMCP("Tablica Rejestracyjna PL")

# Constants
BASE_URL = "https://tablica-rejestracyjna.pl"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"

# Image optimization settings
MAX_IMAGE_DIMENSION = 1920  # Max width or height for uploaded images
JPEG_QUALITY = 85  # JPEG compression quality


def validate_polish_plate(plate: str) -> bool:
    """
    Validate Polish license plate format.
    Polish plates follow patterns like: XX 12345, XXX 1234, X1 2345, etc.
    """
    # Remove spaces and convert to uppercase
    plate = plate.upper().replace(" ", "").replace("-", "")

    # Basic length check (5-8 characters)
    if len(plate) < 5 or len(plate) > 8:
        return False

    # Must contain both letters and numbers
    has_letter = any(c.isalpha() for c in plate)
    has_digit = any(c.isdigit() for c in plate)

    return has_letter and has_digit


def optimize_image(
    image: Image.Image,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    quality: int = JPEG_QUALITY
) -> tuple[bytes, str]:
    """
    Optimize and downscale image if too large.

    Args:
        image: PIL Image object
        max_dimension: Maximum width or height
        quality: JPEG compression quality

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

        # Use high-quality resampling
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
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Save as JPEG
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality, optimize=True)
    image_bytes = buffer.getvalue()

    return image_bytes, 'image/jpeg'


async def load_image(image_input: str, ctx: Context) -> tuple[bytes, str]:
    """
    Load image from file path or base64 data.
    Handles HEIC conversion if available.

    Args:
        image_input: File path or base64 data URL
        ctx: FastMCP context

    Returns:
        Tuple of (image_bytes, mime_type)
    """
    # Check if it's a data URL
    if image_input.startswith('data:'):
        # Parse data URL
        header, data = image_input.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0]
        image_bytes = base64.b64decode(data)

        # Load into PIL for optimization
        image = Image.open(io.BytesIO(image_bytes))
    else:
        # Load from file path
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_input}")

        # Check if HEIC and we don't have support
        if image_path.suffix.lower() in ['.heic', '.heif'] and not HEIC_SUPPORT:
            await ctx.warning("HEIC support not available. Install pillow-heif for HEIC conversion.")
            raise ValueError("HEIC images require pillow-heif to be installed")

        # Load image
        image = Image.open(image_path)

        # Log if we're converting HEIC
        if image_path.suffix.lower() in ['.heic', '.heif']:
            await ctx.info(f"Converting HEIC image to JPEG: {image_path.name}")

    # Optimize and downscale if needed
    return optimize_image(image)


def parse_comments_html(html: str) -> list[dict]:
    """
    Parse comments from the HTML response.

    Args:
        html: HTML content from the website

    Returns:
        List of comment dictionaries
    """
    soup = BeautifulSoup(html, 'lxml')
    comments = []

    # Find comment containers (adjust selectors based on actual HTML structure)
    comment_elements = soup.find_all('div', class_='komentarz') or \
                      soup.find_all('div', class_='comment') or \
                      soup.find_all('article')

    for elem in comment_elements:
        comment = {}

        # Extract comment text (can be in span, p, or div with class 'text')
        text_elem = elem.find('span', class_='text') or elem.find('p') or elem.find('div', class_='text')
        if text_elem:
            # Get text but remove image links
            text_content = text_elem.get_text(strip=True)
            comment['text'] = text_content

        # Extract timestamp
        time_elem = elem.find('time') or elem.find('span', class_='date')
        if time_elem:
            comment['timestamp'] = time_elem.get_text(strip=True)

        # Extract user info (can be 'name', 'name-not-verified', 'user', or 'author')
        user_elem = (elem.find('span', class_='name') or
                    elem.find('span', class_='name-not-verified') or
                    elem.find('span', class_='user') or
                    elem.find('div', class_='author'))
        if user_elem:
            comment['user'] = user_elem.get_text(strip=True)

        # Extract ratings - look for plusMinus divs or vote count
        plus_elem = elem.find('div', class_='plus') or elem.find('span', class_='plus')
        minus_elem = elem.find('div', class_='minus') or elem.find('span', class_='minus')
        vote_count_elem = elem.find('div', class_='voteCount')

        if plus_elem and minus_elem and vote_count_elem:
            try:
                vote_count = int(vote_count_elem.get_text(strip=True) or 0)
                comment['rating'] = {
                    'score': vote_count,
                    'positive': max(0, vote_count),
                    'negative': max(0, -vote_count)
                }
            except ValueError:
                pass

        # Extract image if present
        img_elem = elem.find('img')
        if img_elem and img_elem.get('src'):
            comment['image'] = img_elem['src']

        # Only add if we extracted at least some data
        if comment:
            comments.append(comment)

    # If no structured comments found, try to extract any text content
    if not comments:
        # Look for any text that might be comments
        content_area = soup.find('div', id='content') or soup.find('main') or soup.body
        if content_area:
            # Find all text paragraphs
            paragraphs = content_area.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 20:  # Filter out very short text
                    comments.append({'text': text})

    return comments


@mcp.tool(
    name="fetch_comments",
    description="Fetch all comments/reports for a Polish license plate from tablica-rejestracyjna.pl"
)
async def fetch_comments(
    plate_number: str = Field(
        ...,
        description="Polish license plate number (e.g., 'WW 12345', 'KR1234')"
    ),
    ctx: Context = None
) -> ToolResult:
    """
    Fetch all comments and reports for a given Polish license plate.

    Args:
        plate_number: The license plate to look up
        ctx: FastMCP context for logging

    Returns:
        ToolResult with comments data
    """
    # Validate plate format
    if not validate_polish_plate(plate_number):
        return ToolResult(
            content=[{"type": "text", "text": f"Invalid Polish license plate format: {plate_number}"}],
            structured_content={"error": "Invalid plate format"}
        )

    # Normalize plate number (remove spaces and dashes)
    normalized_plate = plate_number.upper().replace(" ", "").replace("-", "")

    await ctx.info(f"Fetching comments for plate: {normalized_plate}")

    # Construct URL
    url = f"{BASE_URL}/{normalized_plate}"

    # Headers to appear as a regular browser
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'pl-PL,pl;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 404:
                    return ToolResult(
                        content=[{"type": "text", "text": f"No data found for plate: {plate_number}"}],
                        structured_content={
                            "plate": normalized_plate,
                            "found": False,
                            "comments": []
                        }
                    )

                response.raise_for_status()
                html_content = await response.text()

                # Parse comments from HTML
                comments = parse_comments_html(html_content)

                # Create summary
                summary = f"Found {len(comments)} comment(s) for plate {normalized_plate}"
                if comments:
                    summary += "\\n\\nRecent comments:"
                    for i, comment in enumerate(comments[:3], 1):
                        text = comment.get('text', 'No text')[:100]
                        summary += f"\\n{i}. {text}..."

                return ToolResult(
                    content=[{"type": "text", "text": summary}],
                    structured_content={
                        "plate": normalized_plate,
                        "found": True,
                        "comment_count": len(comments),
                        "comments": comments,
                        "url": url
                    }
                )

    except aiohttp.ClientError as e:
        await ctx.error(f"HTTP error fetching comments: {str(e)}")
        return ToolResult(
            content=[{"type": "text", "text": f"Failed to fetch comments: {str(e)}"}],
            structured_content={"error": str(e)}
        )
    except Exception as e:
        await ctx.error(f"Unexpected error: {str(e)}")
        return ToolResult(
            content=[{"type": "text", "text": f"Error: {str(e)}"}],
            structured_content={"error": str(e)}
        )


@mcp.tool(
    name="submit_complaint",
    description="""Submit a complaint about a vehicle to tablica-rejestracyjna.pl.

IMPORTANT WORKFLOW - You MUST follow these steps:
1. First, examine the provided image to identify the traffic violation
2. Generate a detailed complaint description in Polish from a pedestrian's perspective
3. Include specific details: location, type of violation, impact on pedestrians/traffic
4. Then call this tool with your generated Polish description

Example complaint descriptions:
- "Samochód zaparkowany na chodniku przy ul. Marszałkowskiej 15, blokuje przejście dla pieszych. Piesi zmuszeni do wchodzenia na jezdnię."
- "Pojazd stoi na miejscu dla niepełnosprawnych bez uprawnienia. Brak widocznej karty parkingowej."
- "Auto zastawia wjazd do posesji przy ul. Piłsudskiego 23. Uniemożliwia wyjazd mieszkańcom."

The tool will automatically:
- Convert HEIC images to JPEG
- Downscale large images to optimize upload
- Submit the complaint anonymously"""
)
async def submit_complaint(
    plate_number: str = Field(
        ...,
        description="Polish license plate number (e.g., 'WW 12345', 'KR1234')"
    ),
    violation_description: str = Field(
        ...,
        min_length=20,
        description="Detailed description of the violation IN POLISH. Should be generated after analyzing the image."
    ),
    image_path: str = Field(
        ...,
        description="Path to the image file showing the violation (supports HEIC, JPG, PNG)"
    ),
    location: Optional[str] = Field(
        None,
        description="Optional: Specific location/address where violation occurred"
    ),
    ctx: Context = None
) -> ToolResult:
    """
    Submit a complaint with image to tablica-rejestracyjna.pl.

    Args:
        plate_number: The license plate to report
        violation_description: Description of the violation in Polish
        image_path: Path to the evidence image
        location: Optional location information
        ctx: FastMCP context for logging

    Returns:
        ToolResult with submission status
    """
    # Validate plate format
    if not validate_polish_plate(plate_number):
        return ToolResult(
            content=[{"type": "text", "text": f"Invalid Polish license plate format: {plate_number}"}],
            structured_content={"error": "Invalid plate format", "success": False}
        )

    # Normalize plate number
    normalized_plate = plate_number.upper().replace(" ", "").replace("-", "")

    await ctx.info(f"Submitting complaint for plate: {normalized_plate}")
    await ctx.info(f"Loading and optimizing image: {image_path}")

    try:
        # Load and optimize image
        image_bytes, mime_type = await load_image(image_path, ctx)
        image_size_kb = len(image_bytes) / 1024
        await ctx.info(f"Image optimized: {image_size_kb:.1f} KB, {mime_type}")

    except Exception as e:
        await ctx.error(f"Failed to process image: {str(e)}")
        return ToolResult(
            content=[{"type": "text", "text": f"Failed to process image: {str(e)}"}],
            structured_content={"error": f"Image processing failed: {str(e)}", "success": False}
        )

    # Prepare the complaint text with location if provided
    full_description = violation_description
    if location:
        full_description = f"{violation_description}\\n\\nLokalizacja: {location}"

    # Construct URL
    url = f"{BASE_URL}/{normalized_plate}"

    # DISABLED: Automated POST submission disabled to avoid IP bans
    # Use Playwright MCP instead for manual submission

    await ctx.info("Returning Playwright MCP workflow instructions (automated POST disabled to avoid IP bans)")

    playwright_instructions = f"""📋 COMPLAINT SUBMISSION WORKFLOW

To avoid IP bans, use Playwright MCP to submit this complaint manually:

**Plate:** {normalized_plate}
**Description:** {violation_description}
**Image:** {image_size_kb:.1f} KB (optimized)
**URL:** {url}

STEP-BY-STEP INSTRUCTIONS:

1️⃣ Navigate to the website:
   mcp__playwright__browser_navigate(url="https://tablica-rejestracyjna.pl/")

2️⃣ Take a snapshot to see the page:
   mcp__playwright__browser_snapshot()

3️⃣ Search for the license plate:
   - Use the search/lookup field to find: {normalized_plate}

4️⃣ Fill the complaint form with these details:
   - Name: "Świadek" (or your name)
   - Comment/Description: {violation_description}
   - Location (if applicable): {location if location else 'N/A'}

5️⃣ Upload the image:
   - Use the file upload field to select: {image_path}

6️⃣ Submit the form

NOTES:
- Use mcp__playwright__browser_snapshot() after each navigation to identify form fields
- The form typically has fields for: name, comment, photos
- Submit button usually says "Dodaj komentarz" (Add comment)
- All form filling and clicking should be done via Playwright MCP tools

This approach:
✅ Avoids IP bans from automated submissions
✅ Allows manual verification before posting
✅ Works with site's bot protection
"""

    return ToolResult(
        content=[{"type": "text", "text": playwright_instructions}],
        structured_content={
            "success": False,
            "plate": normalized_plate,
            "description": full_description,
            "image_path": image_path,
            "image_size_kb": image_size_kb,
            "url": url,
            "submission_method": "playwright_mcp_manual",
            "reason": "Automated POST requests are disabled to avoid IP bans. Use Playwright MCP for manual submission.",
            "prepared_form_data": {
                "plate": normalized_plate,
                "description": violation_description,
                "full_description": full_description,
                "location": location,
                "image_path": image_path,
                "image_size_kb": image_size_kb
            }
        }
    )

    # COMMENTED OUT: Original automated submission code - disabled to avoid IP bans
    # ==================================================================================
    # # Prepare multipart form data
    # form_data = aiohttp.FormData()
    # form_data.add_field('komentarz', full_description)
    # form_data.add_field('tablica', normalized_plate)
    #
    # # Add image field
    # form_data.add_field(
    #     'zdjecie',
    #     image_bytes,
    #     filename='complaint.jpg',
    #     content_type=mime_type
    # )
    #
    # # Headers for the request
    # headers = {
    #     'User-Agent': USER_AGENT,
    #     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    #     'Accept-Language': 'pl-PL,pl;q=0.9,en;q=0.8',
    #     'Origin': BASE_URL,
    #     'Referer': url,
    #     'DNT': '1',
    # }
    #
    # try:
    #     async with aiohttp.ClientSession() as session:
    #         # Step 1: Visit the page first to establish session cookies
    #         await ctx.info(f"Visiting plate page to establish session: {url}")
    #         async with session.get(url, headers=headers) as initial_response:
    #             initial_html = await initial_response.text()
    #             initial_status = initial_response.status
    #
    #             # Log cookies received
    #             cookies_dict = {cookie.key: cookie.value for cookie in session.cookie_jar}
    #             await ctx.info(f"Session established. Status: {initial_status}, Cookies: {len(cookies_dict)}")
    #
    #             # Parse HTML to check for CSRF tokens
    #             soup = BeautifulSoup(initial_html, 'lxml')
    #             csrf_token = None
    #
    #             # Look for common CSRF token patterns
    #             csrf_input = (
    #                 soup.find('input', attrs={'name': 'csrf_token'}) or
    #                 soup.find('input', attrs={'name': '_token'}) or
    #                 soup.find('input', attrs={'name': 'token'}) or
    #                 soup.find('input', attrs={'name': 'csrf'})
    #             )
    #
    #             if csrf_input and csrf_input.get('value'):
    #                 csrf_token = csrf_input['value']
    #                 await ctx.info(f"Found CSRF token: {csrf_token[:20]}...")
    #                 # Add CSRF token to form data
    #                 form_data.add_field(csrf_input['name'], csrf_token)
    #             else:
    #                 # Check meta tag
    #                 csrf_meta = soup.find('meta', attrs={'name': 'csrf-token'})
    #                 if csrf_meta and csrf_meta.get('content'):
    #                     csrf_token = csrf_meta['content']
    #                     await ctx.info(f"Found CSRF token in meta: {csrf_token[:20]}...")
    #                     form_data.add_field('csrf_token', csrf_token)
    #
    #         # Step 2: Now submit the form with session cookies
    #         await ctx.info(f"Submitting complaint with {len(cookies_dict)} cookies")
    #         async with session.post(
    #             url,
    #             data=form_data,
    #             headers=headers,
    #             allow_redirects=True
    #         ) as response:
    #             response_text = await response.text()
    #             response_headers = dict(response.headers)
    #
    #             # Check for success indicators in response
    #             success = (
    #                 response.status == 200 or
    #                 'sukces' in response_text.lower() or
    #                 'dodano' in response_text.lower() or
    #                 'zapisano' in response_text.lower()
    #             )
    #
    #             if success:
    #                 summary = f"""✅ Complaint submitted successfully!
    #
    # Plate: {normalized_plate}
    # Description: {violation_description[:100]}...
    # Image: {image_size_kb:.1f} KB
    # URL: {url}"""
    #
    #                 return ToolResult(
    #                     content=[{"type": "text", "text": summary}],
    #                     structured_content={
    #                         "success": True,
    #                         "plate": normalized_plate,
    #                         "description": full_description,
    #                         "image_size_kb": image_size_kb,
    #                         "url": url,
    #                         "status_code": response.status,
    #                         "cookies_used": len(cookies_dict),
    #                         "csrf_token_found": csrf_token is not None
    #                     }
    #                 )
    #             else:
    #                 # Try to extract error message from response
    #                 soup = BeautifulSoup(response_text, 'lxml')
    #                 error_elem = soup.find('div', class_='error') or soup.find('div', class_='alert')
    #
    #                 # Generate helpful error message based on status code
    #                 if error_elem:
    #                     error_msg = error_elem.get_text(strip=True)
    #                 elif response.status == 403:
    #                     error_msg = "Access forbidden (403) - Site may have bot protection or rate limiting"
    #                 elif response.status == 404:
    #                     error_msg = "Page not found (404)"
    #                 elif response.status == 429:
    #                     error_msg = "Too many requests (429) - Rate limit exceeded"
    #                 elif response.status == 500:
    #                     error_msg = "Server error (500)"
    #                 elif response.status >= 400:
    #                     error_msg = f"HTTP error {response.status}"
    #                 else:
    #                     error_msg = "Submission failed - no success confirmation found"
    #
    #                 await ctx.warning(f"Submission may have failed. Status: {response.status}")
    #
    #                 # Enhanced error reporting
    #                 response_preview = response_text[:1000] if len(response_text) > 1000 else response_text
    #
    #                 # Generate fallback instructions for Playwright MCP
    #                 playwright_fallback = ""
    #                 if response.status == 403:
    #                     playwright_fallback = f"""
    #
    # FALLBACK: Use Playwright MCP to submit manually
    #
    # If Playwright MCP server is available, use this workflow:
    #
    # 1. Navigate to the website:
    #    mcp__playwright__browser_navigate(url="https://tablica-rejestracyjna.pl/")
    #
    # 2. Search for the plate:
    #    - Type in search box: mcp__playwright__browser_type(element="search textbox", ref="[textbox ref]", text="{normalized_plate}")
    #    - Click search: mcp__playwright__browser_click(element="search button", ref="[button ref]")
    #
    # 3. Fill the complaint form:
    #    - Name field: mcp__playwright__browser_type(element="name field", ref="[name input ref]", text="Świadek")
    #    - Description field: mcp__playwright__browser_type(element="comment field", ref="[comment textarea ref]", text="{violation_description[:100]}...")
    #
    # 4. Upload image:
    #    - Trigger file dialog: mcp__playwright__browser_evaluate(function="() => {{ document.getElementById('photos').click(); return true; }}")
    #    - Upload file: mcp__playwright__browser_file_upload(paths=["{image_path}"])
    #
    # 5. Submit form:
    #    - Click submit: mcp__playwright__browser_click(element="submit button", ref="[submit button ref]")
    #
    # Notes:
    # - Use browser_snapshot first to get element refs
    # - The form typically has fields: "Imię i nazwisko" (name), "Komentarz" (comment), "Zdjęcia z dysku" (photos)
    # - File input ID is usually "photos" or "photos[]"
    # - Submit button text is usually "Dodaj komentarz"
    # """
    #
    #                 suggestion = f"The site may be blocking automated requests.{playwright_fallback}" if response.status == 403 else "Check the error details above."
    #
    #                 return ToolResult(
    #                     content=[{"type": "text", "text": f"""❌ Submission failed
    #
    # HTTP Status: {response.status}
    # Error: {error_msg}
    #
    # Debug Info:
    # - Cookies used: {len(cookies_dict)}
    # - CSRF token: {'Found' if csrf_token else 'Not found'}
    # - Response preview: {response_preview[:200]}...
    #
    # {suggestion}"""}],
    #                     structured_content={
    #                         "success": False,
    #                         "plate": normalized_plate,
    #                         "status_code": response.status,
    #                         "error": error_msg,
    #                         "cookies_received": list(cookies_dict.keys()),
    #                         "csrf_token_found": csrf_token is not None,
    #                         "response_html_preview": response_preview,
    #                         "response_headers": {k: v for k, v in response_headers.items() if k.lower() in ['content-type', 'set-cookie', 'location']},
    #                         "playwright_fallback": {
    #                             "available": response.status == 403,
    #                             "workflow": [
    #                                 {"step": 1, "action": "navigate", "url": "https://tablica-rejestracyjna.pl/"},
    #                                 {"step": 2, "action": "search_plate", "plate": normalized_plate},
    #                                 {"step": 3, "action": "fill_form", "name": "Świadek", "description": violation_description},
    #                                 {"step": 4, "action": "upload_image", "path": image_path},
    #                                 {"step": 5, "action": "submit"}
    #                             ]
    #                         } if response.status == 403 else None
    #                     }
    #                 )
    #
    # except aiohttp.ClientError as e:
    #     await ctx.error(f"HTTP error submitting complaint: {str(e)}")
    #     return ToolResult(
    #         content=[{"type": "text", "text": f"Failed to submit complaint: {str(e)}"}],
    #         structured_content={"error": str(e), "success": False}
    #     )
    # except Exception as e:
    #     await ctx.error(f"Unexpected error: {str(e)}")
    #     return ToolResult(
    #         content=[{"type": "text", "text": f"Error: {str(e)}"}],
    #         structured_content={"error": str(e), "success": False}
    #     )


# Server metadata
if __name__ == "__main__":
    import sys
    print(f"Tablica Rejestracyjna PL MCP Server", file=sys.stderr)
    print(f"HEIC support: {'✅ Available' if HEIC_SUPPORT else '❌ Not available (install pillow-heif)'}", file=sys.stderr)
    mcp.run()