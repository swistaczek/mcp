"""FastMCP server + CLI for printing courier labels (InPost, DPD, DHL, …) on
direct-thermal label printers attached to this machine.

The module does three things:

1. **Discovers thermal printers.** ``lpinfo -v`` lists every device CUPS can
   reach; ``lpstat -v`` lists the queues that already exist. Device URIs are
   matched against a table of known label-printer manufacturers so a Zebra or
   Dymo can be told apart from an office laser.
2. **Provisions a CUPS queue on demand.** macOS 26 refuses raw queues
   (``lpadmin: Raw queues are no longer supported``), so a driver-backed queue
   is created instead — ``zebraep2.ppd`` for EPL2-era Zebras, ``zebra.ppd``
   for ZPL models, ``dymo.ppd`` for Dymo.
3. **Renders the label itself instead of trusting the CUPS raster path.**
   A thermal head is strictly 1-bit: a dot is burnt or it is not. Letting
   CUPS anti-alias a PDF at 203 dpi turns thin strokes into grey, which the
   ``rastertolabel`` filter then dithers into sparse dots — the print comes
   out visibly faint. Instead the page is rasterised at 4× resolution,
   box-filtered down to the exact device grid, and hard-thresholded, which
   thickens strokes rather than thinning them.

The rendered page is emitted as a 1-bit PDF whose MediaBox equals the label
exactly (4×6 in → 288×432 pt → 812×1218 dots at 203 dpi), so CUPS has no
reason to rescale and the bitmap reaches the head 1:1.

Verified against a Zebra TLP2844 (USB, EPL2) with InPost labels.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from PIL import Image, ImageFilter
from pydantic import Field

import escpos_printer

mcp = FastMCP("Label Printer")

# --------------------------------------------------------------------------
# Printer knowledge base
# --------------------------------------------------------------------------

#: Substrings that identify a label/receipt printer inside a CUPS device URI or
#: an ``lpstat`` make-and-model string, mapped to the driver used for them.
THERMAL_MAKES: dict[str, str] = {
    "zebra": "zebra",
    "dymo": "dymo",
    "tsc": "tsc",
    "godex": "godex",
    "citizen": "citizen",
    "bixolon": "bixolon",
    "sato": "sato",
    "argox": "argox",
    "intermec": "intermec",
    "honeywell": "honeywell",
    "brother ql": "brother_ql",
    "toshiba tec": "toshiba",
    "seiko": "seiko",
}

#: Zebra models that speak EPL2 rather than ZPL. The ``-Z`` suffixed variants
#: of the same models speak ZPL, hence the explicit exclusion below.
EPL2_ZEBRA_MODELS = (
    "tlp2824", "tlp2844", "tlp3842", "tlp2742",
    "lp2824", "lp2844", "lp2442", "lp2622",
    "gk420", "gx420", "gc420", "gk888", "gx430",
)

#: CUPS driver URIs, in preference order per make.
DRIVERS: dict[str, tuple[str, ...]] = {
    "zebra_epl2": ("drv:///sample.drv/zebraep2.ppd",),
    "zebra_zpl": ("drv:///sample.drv/zebra.ppd",),
    "dymo": ("drv:///sample.drv/dymo.ppd",),
    "generic_zpl": ("drv:///sample.drv/zebra.ppd",),
}

#: Printer resolution in dots per inch. Every thermal label printer this module
#: targets is 203 dpi; 300 dpi models exist but need a different PPD default.
DEFAULT_DPI = 203

#: Supersampling factor used before the box-filter downscale. 4× is enough to
#: average away anti-aliasing artefacts without a large memory cost.
SUPERSAMPLE = 4

#: Luminance below which a pixel becomes a burnt dot. Tuned on InPost labels:
#: high enough that anti-aliased stroke edges survive as solid black.
DEFAULT_THRESHOLD = 200

#: Blank border kept around the artwork, in PostScript points. InPost labels
#: carry a printed frame right at the page edge which a full-bleed fit clips.
DEFAULT_MARGIN_PT = 8.0

#: Head burn energy, 0–30 on Zebra PPDs. Direct-thermal stock needs the top of
#: the range for solid blacks.
DEFAULT_DARKNESS = 30

#: Inches per second. Slower travel means longer dwell time per dot, i.e. a
#: darker print.
DEFAULT_SPEED = "1"

LABEL_PRESETS: dict[str, tuple[float, float]] = {
    "4x6": (288.0, 432.0),
    "4x4": (288.0, 288.0),
    "4x3": (288.0, 216.0),
    "2x1": (144.0, 72.0),
    "a6": (283.5, 396.9),
}

DEFAULT_LABEL = "4x6"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------
# Shell helpers
# --------------------------------------------------------------------------


def _run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    """Run a command, capturing both streams as text."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


class PrinterError(RuntimeError):
    """Raised when the printing subsystem cannot satisfy a request."""


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


@dataclass
class ThermalPrinter:
    """A thermal label printer, whether or not it has a CUPS queue yet."""

    device_uri: str
    make: str
    model: str
    driver_key: str
    queue: Optional[str] = None
    state: Optional[str] = None
    is_default: bool = False
    connected: bool = False

    @property
    def label(self) -> str:
        return f"{self.make} {self.model}".strip()

    @property
    def language(self) -> str:
        """The page-description language the driver will emit."""
        return "EPL2" if self.driver_key == "zebra_epl2" else "ZPL/native"

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue": self.queue,
            "device_uri": self.device_uri,
            "make": self.make,
            "model": self.model,
            "driver": self.driver_key,
            "language": self.language,
            "installed": self.queue is not None,
            "connected": self.connected,
            "state": self.state,
            "is_default": self.is_default,
        }


def _classify(text: str) -> Optional[tuple[str, str]]:
    """Map a device URI or make-and-model string to ``(make, driver_key)``.

    Returns ``None`` when the text does not look like a label printer.
    """
    lowered = text.lower()
    for needle, make in THERMAL_MAKES.items():
        if needle not in lowered:
            continue
        if make == "zebra":
            # The ``-Z`` variants of the EPL2 models ship ZPL firmware.
            is_epl2 = any(
                m in lowered.replace("-", "").replace("_", "")
                for m in EPL2_ZEBRA_MODELS
            ) and not re.search(r"\d-?z\b", lowered)
            return make, "zebra_epl2" if is_epl2 else "zebra_zpl"
        if make == "dymo":
            return make, "dymo"
        return make, "generic_zpl"
    return None


def _model_from_uri(device_uri: str) -> str:
    """Pull a human-readable model out of a CUPS device URI.

    ``usb://Zebra/TLP2844?serial=41A071000746`` → ``TLP2844``.
    """
    without_scheme = re.sub(r"^[a-z]+://", "", device_uri)
    path = without_scheme.split("?", 1)[0]
    parts = [p for p in path.split("/") if p]
    tail = parts[-1] if parts else without_scheme
    return re.sub(r"%20", " ", tail).strip()


def _parse_lpinfo(output: str) -> list[tuple[str, str]]:
    """Parse ``lpinfo -v`` into ``(scheme_class, device_uri)`` pairs."""
    devices: list[tuple[str, str]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        kind, uri = parts
        if "://" not in uri:
            continue
        devices.append((kind, uri))
    return devices


def _parse_lpstat_v(output: str) -> dict[str, str]:
    """Parse ``lpstat -v`` into ``{queue_name: device_uri}``."""
    queues: dict[str, str] = {}
    for line in output.splitlines():
        match = re.match(r"device for (.+?): (.+)$", line.strip())
        if match:
            queues[match.group(1)] = match.group(2)
    return queues


def _parse_lpstat_p(output: str) -> dict[str, str]:
    """Parse ``lpstat -p`` into ``{queue_name: state_sentence}``.

    A state line reads ``printer NAME is idle.  enabled since <date>`` — the
    trailing clause varies, so everything after the name is kept verbatim.
    """
    states: dict[str, str] = {}
    for line in output.splitlines():
        match = re.match(r"printer (\S+) (.+)$", line.strip())
        if match:
            states[match.group(1)] = match.group(2).strip()
    return states


def _default_queue() -> Optional[str]:
    result = _run(["lpstat", "-d"])
    match = re.search(r"system default destination:\s*(\S+)", result.stdout)
    return match.group(1) if match else None


def discover_thermal_printers() -> list[ThermalPrinter]:
    """Find every thermal label printer reachable from this machine.

    Both connected-but-uninstalled devices (from ``lpinfo -v``) and existing
    CUPS queues (from ``lpstat``) are considered, then merged on device URI so
    an installed printer is reported once.
    """
    queues = _parse_lpstat_v(_run(["lpstat", "-v"]).stdout)
    states = _parse_lpstat_p(_run(["lpstat", "-p"]).stdout)
    default = _default_queue()

    present = {uri for _kind, uri in _parse_lpinfo(_run(["lpinfo", "-v"]).stdout)}
    found: dict[str, ThermalPrinter] = {}

    for uri in present:
        classified = _classify(uri)
        if classified is None:
            continue
        make, driver_key = classified
        found[uri] = ThermalPrinter(
            device_uri=uri,
            make=make.capitalize(),
            model=_model_from_uri(uri),
            driver_key=driver_key,
            connected=True,
        )

    for queue_name, uri in queues.items():
        classified = _classify(uri) or _classify(queue_name)
        if classified is None:
            continue
        make, driver_key = classified
        printer = found.get(uri)
        if printer is None:
            printer = ThermalPrinter(
                device_uri=uri,
                make=make.capitalize(),
                model=_model_from_uri(uri),
                driver_key=driver_key,
                # A queue outlives the device it points at. USB presence is
                # observable through lpinfo; network devices are only listed
                # when they advertise themselves, so absence proves nothing.
                connected=not uri.startswith("usb:"),
            )
            found[uri] = printer
        printer.queue = queue_name
        printer.state = states.get(queue_name)
        printer.is_default = queue_name == default

    return sorted(found.values(), key=lambda p: (not p.connected, p.queue is None, p.label))


def _queue_name_for(printer: ThermalPrinter) -> str:
    """Build a CUPS-legal queue name (no spaces, slashes, ``#`` or quotes)."""
    raw = f"{printer.make}_{printer.model}"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return cleaned or "Thermal_Label_Printer"


def ensure_queue(printer: ThermalPrinter) -> str:
    """Return the printer's CUPS queue, creating one if it has none.

    Raw queues are rejected on modern macOS, so a driver-backed queue is
    created from the bundled sample drivers.
    """
    if printer.queue:
        return printer.queue

    name = _queue_name_for(printer)
    errors: list[str] = []
    for driver in DRIVERS.get(printer.driver_key, DRIVERS["generic_zpl"]):
        result = _run(["lpadmin", "-p", name, "-E", "-v", printer.device_uri, "-m", driver])
        if result.returncode == 0:
            printer.queue = name
            return name
        errors.append(f"{driver}: {result.stderr.strip() or result.stdout.strip()}")

    raise PrinterError(
        f"Could not create a CUPS queue for {printer.label}. Tried: " + "; ".join(errors)
    )


def resolve_printer(requested: Optional[str] = None) -> ThermalPrinter:
    """Pick the printer to use.

    Without ``requested``, the sole detected thermal printer is used; if more
    than one is present the caller must disambiguate.
    """
    printers = discover_thermal_printers()
    if not printers:
        raise PrinterError(
            "No thermal label printer detected. Check that it is powered on and "
            "connected, then re-run discovery."
        )

    if requested:
        for printer in printers:
            if requested in (printer.queue, printer.device_uri) or requested.lower() in printer.label.lower():
                return printer
        names = ", ".join(p.queue or p.label for p in printers)
        raise PrinterError(f"No thermal printer matches {requested!r}. Detected: {names}")

    if len(printers) == 1:
        return printers[0]

    installed = [p for p in printers if p.queue]
    if len(installed) == 1:
        return installed[0]

    names = ", ".join(p.queue or p.label for p in printers)
    raise PrinterError(f"Multiple thermal printers detected ({names}); specify one.")


# --------------------------------------------------------------------------
# Label geometry
# --------------------------------------------------------------------------


def parse_label_size(spec: str) -> tuple[float, float]:
    """Resolve a label size to ``(width_pt, height_pt)``.

    Accepts presets (``4x6``, ``a6``) and explicit dimensions with a unit
    suffix (``100x150mm``, ``4x6in``, ``288x432pt``). A bare ``WxH`` is read as
    inches, matching how label stock is sold.
    """
    key = spec.strip().lower().replace(" ", "")
    if key in LABEL_PRESETS:
        return LABEL_PRESETS[key]

    match = re.fullmatch(r"([\d.]+)x([\d.]+)(mm|cm|in|pt)?", key)
    if not match:
        presets = ", ".join(sorted(LABEL_PRESETS))
        raise ValueError(
            f"Unrecognised label size {spec!r}. Use a preset ({presets}) or "
            "dimensions like 100x150mm, 4x6in, 288x432pt."
        )

    width, height = float(match.group(1)), float(match.group(2))
    unit = match.group(3) or "in"
    per_unit = {"mm": 72.0 / 25.4, "cm": 72.0 / 2.54, "in": 72.0, "pt": 1.0}[unit]
    return width * per_unit, height * per_unit


def page_size_option(width_pt: float, height_pt: float, supported: Iterable[str]) -> str:
    """Choose the PPD ``PageSize`` keyword for a label.

    Zebra PPDs name their media ``w<width_pt>h<height_pt>``. When no listed
    size matches, CUPS accepts a ``Custom.<w>x<h>`` fallback.
    """
    named = f"w{round(width_pt)}h{round(height_pt)}"
    return named if named in set(supported) else f"Custom.{round(width_pt)}x{round(height_pt)}"


def supported_options(queue: str) -> dict[str, list[str]]:
    """Read the option keywords a queue's PPD actually accepts.

    Passing an unsupported ``-o`` to ``lp`` is silently ignored by CUPS, so
    options are filtered against this rather than assumed.
    """
    result = _run(["lpoptions", "-p", queue, "-l"])
    options: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        match = re.match(r"([^/]+)/[^:]*:\s*(.+)$", line.strip())
        if not match:
            continue
        keyword = match.group(1)
        values = [v.lstrip("*") for v in match.group(2).split()]
        options[keyword] = values
    return options


def fit_geometry(
    src_w: float,
    src_h: float,
    target_w: float,
    target_h: float,
    margin: float,
) -> tuple[float, float, float, bool]:
    """Compute how to place artwork on a label.

    Returns ``(scale, offset_x, offset_y, rotate)``. Artwork whose orientation
    disagrees with the label is rotated a quarter turn before fitting, so a
    landscape receipt still fills a portrait label.
    """
    rotate = (src_w > src_h) != (target_w > target_h)
    if rotate:
        src_w, src_h = src_h, src_w

    avail_w = max(target_w - 2 * margin, 1.0)
    avail_h = max(target_h - 2 * margin, 1.0)
    scale = min(avail_w / src_w, avail_h / src_h)
    offset_x = (target_w - src_w * scale) / 2
    offset_y = (target_h - src_h * scale) / 2
    return scale, offset_x, offset_y, rotate


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _rasterise_pdf(pdf_path: Path, dpi: int, workdir: Path) -> list[Image.Image]:
    """Render every page of a PDF to greyscale bitmaps via poppler."""
    if shutil.which("pdftoppm") is None:
        raise PrinterError(
            "pdftoppm not found — install poppler (`brew install poppler`) or "
            "pass bilevel=False to fall back on the CUPS raster path."
        )

    prefix = workdir / "page"
    result = _run(["pdftoppm", "-gray", "-r", str(dpi), "-png", str(pdf_path), str(prefix)])
    if result.returncode != 0:
        raise PrinterError(f"pdftoppm failed on {pdf_path.name}: {result.stderr.strip()}")

    pages = sorted(workdir.glob("page-*.png"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not pages:
        raise PrinterError(f"pdftoppm produced no pages for {pdf_path.name}")
    return [Image.open(p).convert("L") for p in pages]


def _to_bilevel(
    source: Image.Image,
    target_px: tuple[int, int],
    margin_px: int,
    threshold: int,
    bold: int,
) -> Image.Image:
    """Fit a greyscale page onto the device grid and hard-threshold it.

    The downscale uses a box filter so each output dot is the true area average
    of the pixels it covers; thresholding that average keeps anti-aliased edges
    solid instead of letting a dither scatter them.
    """
    target_w, target_h = target_px
    avail_w = max(target_w - 2 * margin_px, 1)
    avail_h = max(target_h - 2 * margin_px, 1)

    if (source.width > source.height) != (target_w > target_h):
        source = source.transpose(Image.Transpose.ROTATE_90)

    scale = min(avail_w / source.width, avail_h / source.height)
    fitted = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.BOX,
    )

    if bold:
        # A minimum filter spreads dark pixels outward, thickening strokes.
        fitted = fitted.filter(ImageFilter.MinFilter(2 * bold + 1))

    canvas = Image.new("L", (target_w, target_h), 255)
    canvas.paste(fitted, ((target_w - fitted.width) // 2, (target_h - fitted.height) // 2))
    return canvas.point(lambda p: 0 if p < threshold else 255, mode="L").convert("1")


def render_label(
    source: Path,
    width_pt: float,
    height_pt: float,
    *,
    dpi: int = DEFAULT_DPI,
    threshold: int = DEFAULT_THRESHOLD,
    margin_pt: float = DEFAULT_MARGIN_PT,
    bold: int = 0,
    output: Optional[Path] = None,
) -> tuple[Path, dict[str, Any]]:
    """Turn a label file into a 1-bit PDF matched to the printer's dot grid.

    The output page is exactly ``width_pt × height_pt`` and its image is
    exactly ``width_in × dpi`` dots wide, so CUPS passes it through untouched.
    """
    if not source.exists():
        raise PrinterError(f"Label file not found: {source}")

    target_px = (round(width_pt / 72 * dpi), round(height_pt / 72 * dpi))
    margin_px = round(margin_pt / 72 * dpi)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        if source.suffix.lower() == ".pdf":
            pages = _rasterise_pdf(source, dpi * SUPERSAMPLE, workdir)
        elif source.suffix.lower() in IMAGE_SUFFIXES:
            pages = [Image.open(source).convert("L")]
        else:
            raise PrinterError(
                f"Unsupported label format {source.suffix!r}; expected a PDF or an image."
            )

        rendered = [_to_bilevel(p, target_px, margin_px, threshold, bold) for p in pages]

    destination = output or Path(tempfile.mkstemp(suffix=".pdf", prefix="label-")[1])
    rendered[0].save(
        destination,
        resolution=dpi,
        save_all=len(rendered) > 1,
        append_images=rendered[1:],
    )

    # On a 1-bit image the histogram has counts only in bins 0 (black) and 255.
    total_px = target_px[0] * target_px[1]
    ink = rendered[0].histogram()[0] / total_px
    return destination, {
        "pages": len(rendered),
        "dots": f"{target_px[0]}x{target_px[1]}",
        "page_size_pt": f"{round(width_pt)}x{round(height_pt)}",
        "dpi": dpi,
        "threshold": threshold,
        "margin_pt": margin_pt,
        "ink_coverage": round(ink, 4),
    }


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------


@dataclass
class PrintOutcome:
    job_id: str
    queue: str
    printer: str
    copies: int
    options: dict[str, str]
    render: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "queue": self.queue,
            "printer": self.printer,
            "copies": self.copies,
            "options": self.options,
            "render": self.render,
        }


def print_file(
    source: Path,
    *,
    printer: Optional[str] = None,
    label_size: str = DEFAULT_LABEL,
    copies: int = 1,
    darkness: int = DEFAULT_DARKNESS,
    speed: str = DEFAULT_SPEED,
    threshold: int = DEFAULT_THRESHOLD,
    margin_pt: float = DEFAULT_MARGIN_PT,
    bold: int = 0,
    bilevel: bool = True,
    dry_run: bool = False,
) -> PrintOutcome:
    """Render a label file and submit it to a thermal printer."""
    target = resolve_printer(printer)
    queue = ensure_queue(target)
    width_pt, height_pt = parse_label_size(label_size)

    available = supported_options(queue)
    options: dict[str, str] = {
        "PageSize": page_size_option(width_pt, height_pt, available.get("PageSize", []))
    }
    # Darkness and print rate exist on Zebra PPDs but not on every driver.
    if "Darkness" in available:
        options["Darkness"] = str(darkness)
    if "zePrintRate" in available:
        options["zePrintRate"] = speed

    render_info: dict[str, Any] = {}
    to_print = source
    scratch: Optional[Path] = None
    if bilevel:
        to_print, render_info = render_label(
            source,
            width_pt,
            height_pt,
            threshold=threshold,
            margin_pt=margin_pt,
            bold=bold,
        )
        scratch = to_print
    else:
        options["fit-to-page"] = "true"

    cmd = ["lp", "-d", queue, "-n", str(copies)]
    for key, value in options.items():
        cmd += ["-o", f"{key}={value}"]
    cmd.append(str(to_print))

    try:
        if dry_run:
            return PrintOutcome(
                job_id="(dry-run)",
                queue=queue,
                printer=target.label,
                copies=copies,
                options=options,
                render={**render_info, "command": " ".join(cmd)},
            )

        result = _run(cmd)
        if result.returncode != 0:
            raise PrinterError(f"lp failed: {result.stderr.strip() or result.stdout.strip()}")

        match = re.search(r"request id is (\S+)", result.stdout)
        return PrintOutcome(
            job_id=match.group(1) if match else "(unknown)",
            queue=queue,
            printer=target.label,
            copies=copies,
            options=options,
            render=render_info,
        )
    finally:
        # ``lp`` spools the file before returning, so the scratch render is
        # safe to drop as soon as the command completes.
        if scratch is not None:
            scratch.unlink(missing_ok=True)


def queue_status(queue: str) -> dict[str, Any]:
    """Report a queue's state and its pending jobs."""
    state = _parse_lpstat_p(_run(["lpstat", "-p", queue]).stdout).get(queue)
    jobs = []
    for line in _run(["lpstat", "-o", queue]).stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            jobs.append({"job_id": parts[0], "owner": parts[1], "size": parts[2]})
    return {"queue": queue, "state": state, "idle": bool(state and "idle" in state), "jobs": jobs}


# --------------------------------------------------------------------------
# MCP tools
# --------------------------------------------------------------------------


@mcp.tool
async def list_thermal_printers() -> ToolResult:
    """List thermal label printers connected to this machine.

    Detects both printers that already have a CUPS queue and ones that are
    plugged in but not yet installed — printing to the latter installs a queue
    automatically. Use this to confirm a printer is reachable before printing,
    or to get the queue name when several printers are attached.
    """
    printers = await asyncio.to_thread(discover_thermal_printers)
    payload = [p.to_dict() for p in printers]

    # ESC/POS receipt printers are invisible to this list otherwise: CUPS can
    # see the device but has no driver for the language, so they are driven
    # directly over USB instead.
    try:
        escpos = await asyncio.to_thread(escpos_printer.find_escpos_printers)
    except escpos_printer.EscPosError:
        escpos = []
    payload += [p.to_dict() for p in escpos]

    if not payload:
        text = "No thermal printer detected. Check power and USB connection."
    else:
        lines = []
        for p in printers:
            status = p.state or ("installed" if p.queue else "not installed")
            presence = "" if p.connected else " (NOT CONNECTED — stale queue)"
            lines.append(f"- {p.label} [{p.language}] — {p.queue or 'no queue yet'}: {status}{presence}")
        for p in escpos:
            lines.append(f"- {p.name} [ESC/POS] — USB {p.usb_id}, {p.width_mm} mm printable")
        text = "Thermal printers:\n" + "\n".join(lines)

    return ToolResult(content=text, structured_content={"printers": payload, "count": len(payload)})


@mcp.tool
async def print_label(
    file_path: str = Field(description="Path to the label file — a PDF (e.g. an InPost courier label) or an image."),
    printer: Optional[str] = Field(default=None, description="Queue name, device URI, or make/model fragment. Omit when a single thermal printer is attached."),
    label_size: str = Field(default=DEFAULT_LABEL, description="Label stock size: a preset (4x6, 4x4, 4x3, 2x1, a6) or explicit dimensions such as 100x150mm or 4x6in."),
    copies: int = Field(default=1, ge=1, le=50, description="Number of copies to print."),
    darkness: int = Field(default=DEFAULT_DARKNESS, ge=0, le=30, description="Head burn energy. Higher is darker; direct-thermal stock usually needs 25-30."),
    speed: str = Field(default=DEFAULT_SPEED, description="Print speed in inches/second ('1' is slowest and darkest)."),
    threshold: int = Field(default=DEFAULT_THRESHOLD, ge=1, le=254, description="Black/white cutoff, 1-254. Raise it for heavier ink, lower it if solid areas smear."),
    bold: int = Field(default=0, ge=0, le=3, description="Stroke thickening passes. Use 1 when barcodes still scan poorly."),
    dry_run: bool = Field(default=False, description="Render and report the exact lp command without printing."),
) -> ToolResult:
    """Print a courier label on a direct-thermal label printer.

    The label is rasterised at 4x the printer's resolution, downsampled to the
    exact dot grid, and hard-thresholded to pure black and white — thermal
    heads are 1-bit, and letting the print system dither an anti-aliased page
    produces visibly faint output with barcodes that scan poorly.

    Artwork is scaled to fit the label with a small margin, and rotated a
    quarter turn when its orientation disagrees with the stock, so a label
    designed at a slightly different size still prints complete rather than
    clipped at the edges.
    """
    try:
        outcome = await asyncio.to_thread(
            print_file,
            Path(file_path).expanduser(),
            printer=printer,
            label_size=label_size,
            copies=copies,
            darkness=darkness,
            speed=speed,
            threshold=threshold,
            bold=bold,
            dry_run=dry_run,
        )
    except (PrinterError, ValueError) as exc:
        return ToolResult(
            content=f"Print failed: {exc}",
            structured_content={"error": str(exc), "file_path": file_path},
        )

    verb = "Would print" if dry_run else "Printed"
    pages = outcome.render.get("pages", 1)
    text = (
        f"{verb} {Path(file_path).name} ({pages} page(s), {outcome.copies} cop(y/ies)) "
        f"on {outcome.printer} via queue {outcome.queue}. Job {outcome.job_id}."
    )
    return ToolResult(content=text, structured_content=outcome.to_dict())


def print_qr(
    data: str,
    *,
    printer: Optional[str] = None,
    module_dots: int = 8,
    error_correction: str = "M",
    caption: Optional[str] = None,
    label_size: str = DEFAULT_LABEL,
) -> dict[str, Any]:
    """Print a QR code, choosing the cheapest route the hardware supports.

    An ESC/POS receipt printer is driven directly over USB; a CUPS-backed label
    printer gets the same symbol rendered onto its label stock.
    """
    escpos = escpos_printer.find_escpos_printers()
    if escpos:
        target = escpos[0]
        if printer:
            matches = [p for p in escpos if printer.lower() in (p.name + p.usb_id).lower()]
            target = matches[0] if matches else target
        symbol = escpos_printer.qr_image(
            data,
            width_dots=target.width_dots,
            module_dots=module_dots,
            error_correction=error_correction,
        )
        payload = escpos_printer.raster_payload(symbol, target.width_dots)
        if caption:
            payload += escpos_printer.text_payload(caption, centred=True)
        with escpos_printer.EscPosDevice(target) as device:
            device.print_page(payload)
        return {
            "printer": target.name,
            "transport": "escpos-usb",
            "data": data,
            "symbol_dots": f"{symbol.width}x{symbol.height}",
            "bytes": len(payload),
        }

    # No ESC/POS device — fall back to the CUPS label path.
    width_pt, height_pt = parse_label_size(label_size)
    width_dots = round(width_pt / 72 * DEFAULT_DPI)
    symbol = escpos_printer.qr_image(
        data,
        width_dots=width_dots,
        module_dots=module_dots,
        error_correction=error_correction,
    )
    scratch = Path(tempfile.mkstemp(suffix=".png", prefix="qr-")[1])
    try:
        symbol.save(scratch)
        outcome = print_file(scratch, printer=printer, label_size=label_size)
    finally:
        scratch.unlink(missing_ok=True)
    return {**outcome.to_dict(), "transport": "cups", "data": data}


@mcp.tool
async def print_qr_code(
    data: str = Field(description="The value to encode — a URL, tracking number, Wi-Fi string, or any text."),
    printer: Optional[str] = Field(default=None, description="Printer name fragment. Omit when a single printer is attached."),
    module_dots: int = Field(default=8, ge=1, le=16, description="Size of one QR module in printer dots. Larger scans more reliably; the symbol is shrunk automatically if it would overflow the paper."),
    error_correction: str = Field(default="M", description="QR error-correction level: L (7%), M (15%), Q (25%) or H (30%). Use H for labels that may get scuffed."),
    caption: Optional[str] = Field(default=None, description="Optional line of text printed under the code."),
) -> ToolResult:
    """Print a QR code encoding any value on an attached thermal printer.

    Works on both supported printer types: an ESC/POS receipt printer is driven
    directly over USB, while a CUPS-backed label printer gets the symbol
    rendered onto its label stock.

    The code is rasterised rather than delegated to the printer's built-in QR
    command — much budget firmware silently ignores that command and emits
    nothing, whereas an image prints everywhere.
    """
    try:
        result = await asyncio.to_thread(
            print_qr,
            data,
            printer=printer,
            module_dots=module_dots,
            error_correction=error_correction,
            caption=caption,
        )
    except (PrinterError, escpos_printer.EscPosError, ValueError) as exc:
        return ToolResult(
            content=f"QR print failed: {exc}",
            structured_content={"error": str(exc), "data": data},
        )

    return ToolResult(
        content=f"Printed QR code for {data!r} on {result.get('printer')} ({result.get('transport')}).",
        structured_content=result,
    )


@mcp.tool
async def get_printer_status(
    printer: Optional[str] = Field(default=None, description="Queue name or make/model fragment. Omit when a single thermal printer is attached."),
) -> ToolResult:
    """Check whether a thermal printer is idle and what jobs are still queued.

    Useful after printing to confirm the job drained to the device, or to see
    why a label has not come out — a stopped queue usually means the printer is
    out of media or offline.
    """
    try:
        target = await asyncio.to_thread(resolve_printer, printer)
        if not target.queue:
            return ToolResult(
                content=f"{target.label} is connected but has no CUPS queue yet; printing will create one.",
                structured_content={"printer": target.to_dict(), "jobs": []},
            )
        status = await asyncio.to_thread(queue_status, target.queue)
    except PrinterError as exc:
        return ToolResult(content=f"Status unavailable: {exc}", structured_content={"error": str(exc)})

    pending = len(status["jobs"])
    text = f"{target.label} ({status['queue']}): {status['state'] or 'unknown'}; {pending} job(s) queued."
    return ToolResult(content=text, structured_content={**status, "printer": target.to_dict()})


@mcp.tool
async def cancel_print_jobs(
    printer: Optional[str] = Field(default=None, description="Queue name or make/model fragment. Omit when a single thermal printer is attached."),
    job_id: Optional[str] = Field(default=None, description="Specific job to cancel, e.g. 'Zebra_TLP2844-105'. Omit to cancel every job on the queue."),
) -> ToolResult:
    """Cancel queued print jobs — use when a label was sent by mistake.

    Jobs already handed to the printer may still finish; this only clears what
    the queue has not yet transmitted.
    """
    try:
        target = await asyncio.to_thread(resolve_printer, printer)
    except PrinterError as exc:
        return ToolResult(content=f"Cancel failed: {exc}", structured_content={"error": str(exc)})

    if not target.queue:
        return ToolResult(
            content=f"{target.label} has no queue, so there is nothing to cancel.",
            structured_content={"cancelled": [], "queue": None},
        )

    before = await asyncio.to_thread(queue_status, target.queue)
    cmd = ["cancel", job_id] if job_id else ["cancel", "-a", target.queue]
    result = await asyncio.to_thread(_run, cmd)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        return ToolResult(content=f"Cancel failed: {message}", structured_content={"error": message})

    cancelled = [job_id] if job_id else [j["job_id"] for j in before["jobs"]]
    return ToolResult(
        content=f"Cancelled {len(cancelled)} job(s) on {target.queue}.",
        structured_content={"cancelled": cancelled, "queue": target.queue},
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="label_printer",
        description="Print courier labels on a direct-thermal label printer.",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")

    # Repeating --json on each subcommand lets it appear on either side of the
    # verb. SUPPRESS keeps the subparser from resetting a value already given
    # to the top-level parser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit machine-readable output",
    )

    sub = parser.add_subparsers(dest="command", required=True, parser_class=argparse.ArgumentParser)

    def add(name: str, **kwargs) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[common], **kwargs)

    add("detect", help="list connected thermal label printers")

    status = add("status", help="show queue state and pending jobs")
    status.add_argument("-p", "--printer", help="queue name or make/model fragment")

    install = add("install", help="create a CUPS queue for a detected printer")
    install.add_argument("-p", "--printer", help="queue name or make/model fragment")

    render = add("render", help="write the 1-bit label PDF without printing")
    render.add_argument("file")
    render.add_argument("-o", "--output", required=True, help="destination PDF")
    render.add_argument("-s", "--label-size", default=DEFAULT_LABEL)
    render.add_argument("-t", "--threshold", type=int, default=DEFAULT_THRESHOLD)
    render.add_argument("-m", "--margin", type=float, default=DEFAULT_MARGIN_PT)
    render.add_argument("-b", "--bold", type=int, default=0)

    print_cmd = add("print", help="render and print a label")
    print_cmd.add_argument("file")
    print_cmd.add_argument("-p", "--printer", help="queue name or make/model fragment")
    print_cmd.add_argument("-s", "--label-size", default=DEFAULT_LABEL)
    print_cmd.add_argument("-n", "--copies", type=int, default=1)
    print_cmd.add_argument("-d", "--darkness", type=int, default=DEFAULT_DARKNESS)
    print_cmd.add_argument("--speed", default=DEFAULT_SPEED)
    print_cmd.add_argument("-t", "--threshold", type=int, default=DEFAULT_THRESHOLD)
    print_cmd.add_argument("-m", "--margin", type=float, default=DEFAULT_MARGIN_PT)
    print_cmd.add_argument("-b", "--bold", type=int, default=0)
    print_cmd.add_argument("--no-bilevel", action="store_true", help="let CUPS rasterise instead")
    print_cmd.add_argument("--dry-run", action="store_true")

    qr = add("qr", help="print a QR code for a value")
    qr.add_argument("data", help="value to encode (URL, tracking number, text)")
    qr.add_argument("-p", "--printer", help="printer name fragment")
    qr.add_argument("-M", "--module-dots", type=int, default=8)
    qr.add_argument("-e", "--error-correction", default="M", choices=["L", "M", "Q", "H"])
    qr.add_argument("-c", "--caption", help="text line printed under the code")

    cancel = add("cancel", help="cancel queued jobs")
    cancel.add_argument("-p", "--printer", help="queue name or make/model fragment")
    cancel.add_argument("-j", "--job", help="specific job id")

    return parser


def _emit(payload: Any, text: str, as_json: bool) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False) if as_json else text)


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.command == "detect":
            printers = discover_thermal_printers()
            try:
                escpos = escpos_printer.find_escpos_printers()
            except escpos_printer.EscPosError:
                escpos = []
            if not printers and not escpos:
                _emit({"printers": []}, "No thermal printer detected.", args.json)
                return 1
            lines = [
                f"{p.queue or '(not installed)':<24} {p.label:<24} {p.device_uri}"
                f"{'' if p.connected else '   [NOT CONNECTED]'}"
                for p in printers
            ]
            lines += [
                f"{'(direct usb)':<24} {p.name:<24} {p.usb_id}   [ESC/POS, {p.width_mm} mm]"
                for p in escpos
            ]
            payload = [p.to_dict() for p in printers] + [p.to_dict() for p in escpos]
            _emit({"printers": payload}, "\n".join(lines), args.json)
            return 0

        if args.command == "status":
            target = resolve_printer(args.printer)
            if not target.queue:
                _emit({"printer": target.to_dict()}, f"{target.label}: connected, no queue yet.", args.json)
                return 0
            status = queue_status(target.queue)
            text = f"{target.label} ({status['queue']}): {status['state']}; {len(status['jobs'])} job(s)"
            _emit(status, text, args.json)
            return 0

        if args.command == "install":
            target = resolve_printer(args.printer)
            queue = ensure_queue(target)
            _emit({"queue": queue, "printer": target.to_dict()}, f"Queue ready: {queue}", args.json)
            return 0

        if args.command == "render":
            width_pt, height_pt = parse_label_size(args.label_size)
            out, info = render_label(
                Path(args.file).expanduser(),
                width_pt,
                height_pt,
                threshold=args.threshold,
                margin_pt=args.margin,
                bold=args.bold,
                output=Path(args.output).expanduser(),
            )
            _emit({"output": str(out), **info}, f"Wrote {out} ({info['dots']} dots, {info['pages']} page(s))", args.json)
            return 0

        if args.command == "print":
            outcome = print_file(
                Path(args.file).expanduser(),
                printer=args.printer,
                label_size=args.label_size,
                copies=args.copies,
                darkness=args.darkness,
                speed=args.speed,
                threshold=args.threshold,
                margin_pt=args.margin,
                bold=args.bold,
                bilevel=not args.no_bilevel,
                dry_run=args.dry_run,
            )
            verb = "Would print" if args.dry_run else "Submitted"
            text = f"{verb} {args.file} -> {outcome.printer} ({outcome.queue}), job {outcome.job_id}"
            _emit(outcome.to_dict(), text, args.json)
            return 0

        if args.command == "qr":
            result = print_qr(
                args.data,
                printer=args.printer,
                module_dots=args.module_dots,
                error_correction=args.error_correction,
                caption=args.caption,
            )
            _emit(result, f"Printed QR for {args.data} on {result.get('printer')}", args.json)
            return 0

        if args.command == "cancel":
            target = resolve_printer(args.printer)
            if not target.queue:
                _emit({"cancelled": []}, "Nothing to cancel.", args.json)
                return 0
            cmd = ["cancel", args.job] if args.job else ["cancel", "-a", target.queue]
            result = _run(cmd)
            if result.returncode != 0:
                print(result.stderr.strip() or result.stdout.strip(), file=sys.stderr)
                return 1
            _emit({"cancelled": True, "queue": target.queue}, f"Cancelled on {target.queue}.", args.json)
            return 0

    except (PrinterError, escpos_printer.EscPosError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
