"""ESC/POS transport for USB receipt printers that CUPS cannot drive.

Cheap thermal receipt printers (Winbond ``0416:5011`` and friends, usually
branded Xprinter/Gprinter/Zjiang or not branded at all) expose a USB
printer-class interface but speak **ESC/POS**, for which macOS ships no
driver — ``rastertolabel`` only knows EPL, ZPL, CPCL and Dymo. CUPS therefore
lists the device and then has nothing to send it.

This module talks to them directly over libusb instead:

- :func:`find_escpos_printers` enumerates candidates by USB vendor/product id
- :class:`EscPosDevice` claims the interface and writes a page in one transfer
- :func:`raster_payload` encodes a bitmap, :func:`qr_image` renders a QR code

Three constraints, each established the hard way against a Winbond
``0416:5011`` unit, are enforced here rather than left to callers:

1. **Never exceed the head width.** A raster wider than the printhead does not
   error — it wedges the firmware until the printer is power-cycled.
2. **Send a page in exactly one write.** Splitting the raster into bands looks
   safer and is not. Bulk USB already provides flow control — a full buffer
   NAKs and the host controller retries in hardware, within the single
   transfer — whereas separate transfers leave gaps this firmware treats as
   the end of the raster, after which it errors and wedges.
3. **Rasterise QR codes.** The native ``GS ( k`` QR command is far cheaper on
   the wire, but this firmware silently ignores it and prints nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image

#: USB ids known to be ESC/POS thermal printers, with the printable width of
#: the head in dots. 384 dots = 48 mm on 58 mm paper; 576 = 72 mm on 80 mm.
KNOWN_ESCPOS_DEVICES: dict[tuple[int, int], dict[str, Any]] = {
    (0x0416, 0x5011): {"name": "Generic ESC/POS (Winbond)", "width_dots": 384},
    (0x0483, 0x5743): {"name": "STM ESC/POS", "width_dots": 384},
    (0x0FE6, 0x811E): {"name": "ICS Advent POS-58", "width_dots": 384},
    (0x1A86, 0x7584): {"name": "QinHeng ESC/POS", "width_dots": 384},
    (0x6868, 0x0200): {"name": "Zjiang ESC/POS", "width_dots": 576},
    (0x28E9, 0x0289): {"name": "GD32 ESC/POS", "width_dots": 384},
}

#: Fallback when a device is recognised but its head width is unknown. 58 mm
#: paper is by far the most common, and printing narrow is harmless whereas
#: printing too wide wedges the device.
DEFAULT_WIDTH_DOTS = 384

DPI = 203

#: Timeout for a whole-page bulk write. The transfer blocks while the head
#: prints, so it must cover the printing time, not just the wire time.
WRITE_TIMEOUT_MS = 60000

INIT = b"\x1b@"
FEED = b"\n\n\n\n"

#: Pillow's mode ``1`` stores 0 and 255. Passing a bare ``1`` as a fill colour
#: is accepted but stores a literal 1, which reads back as neither black nor
#: white — name the values instead of writing them inline.
BLACK, WHITE = 0, 255

QR_ERROR_LEVELS = {"L": 48, "M": 49, "Q": 50, "H": 51}


class EscPosError(RuntimeError):
    """Raised when an ESC/POS printer cannot be reached or driven."""


def _require_pyusb():
    try:
        import usb.core
        import usb.util
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EscPosError(
            "pyusb is required to talk to ESC/POS printers "
            "(`uv add pyusb`, plus libusb via `brew install libusb`)."
        ) from exc
    return usb.core, usb.util


@dataclass
class EscPosPrinter:
    """A USB ESC/POS printer and what is known about its head."""

    vendor_id: int
    product_id: int
    name: str
    width_dots: int
    serial: Optional[str] = None

    @property
    def usb_id(self) -> str:
        return f"{self.vendor_id:04x}:{self.product_id:04x}"

    @property
    def width_mm(self) -> float:
        return round(self.width_dots / DPI * 25.4, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "usb_id": self.usb_id,
            "name": self.name,
            "serial": self.serial,
            "width_dots": self.width_dots,
            "width_mm": self.width_mm,
            "language": "ESC/POS",
            "transport": "libusb",
        }


def find_escpos_printers() -> list[EscPosPrinter]:
    """Enumerate attached USB devices that are known ESC/POS printers."""
    usb_core, _usb_util = _require_pyusb()

    printers: list[EscPosPrinter] = []
    for device in usb_core.find(find_all=True) or []:
        key = (device.idVendor, device.idProduct)
        known = KNOWN_ESCPOS_DEVICES.get(key)
        if known is None:
            continue
        try:
            serial = device.serial_number
        except (ValueError, NotImplementedError):  # pragma: no cover - driver dependent
            serial = None
        printers.append(
            EscPosPrinter(
                vendor_id=device.idVendor,
                product_id=device.idProduct,
                name=known["name"],
                width_dots=known.get("width_dots", DEFAULT_WIDTH_DOTS),
                serial=serial,
            )
        )
    return printers


class EscPosDevice:
    """An open connection to an ESC/POS printer's bulk OUT endpoint."""

    def __init__(self, printer: EscPosPrinter):
        self.printer = printer
        self._usb_core, self._usb_util = _require_pyusb()
        self._device = None
        self._endpoint = None
        self._claimed: Optional[int] = None

    def __enter__(self) -> "EscPosDevice":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def open(self) -> None:
        device = self._usb_core.find(
            idVendor=self.printer.vendor_id, idProduct=self.printer.product_id
        )
        if device is None:
            raise EscPosError(f"ESC/POS printer {self.printer.usb_id} is not connected.")

        try:
            device.set_configuration()
        except self._usb_core.USBError as exc:
            raise EscPosError(
                f"Could not configure {self.printer.usb_id} ({exc}). The printer is "
                "most likely wedged from an earlier failed transfer — power-cycle it."
            ) from exc

        interface = device.get_active_configuration()[(0, 0)]

        # CUPS lists this device too and its usb backend probes periodically.
        # A short write slips between probes, but a multi-second raster
        # transfer will eventually collide with one and fail with EIO. Claiming
        # the interface takes exclusive ownership for the duration.
        try:
            if device.is_kernel_driver_active(interface.bInterfaceNumber):
                device.detach_kernel_driver(interface.bInterfaceNumber)
        except (NotImplementedError, self._usb_core.USBError):
            pass
        try:
            self._usb_util.claim_interface(device, interface.bInterfaceNumber)
            self._claimed = interface.bInterfaceNumber
        except self._usb_core.USBError:
            self._claimed = None

        endpoint = self._usb_util.find_descriptor(
            interface,
            custom_match=lambda e: self._usb_util.endpoint_direction(e.bEndpointAddress)
            == self._usb_util.ENDPOINT_OUT,
        )
        if endpoint is None:
            raise EscPosError(f"No bulk OUT endpoint on {self.printer.usb_id}.")

        # A previous overrun can leave the endpoint halted.
        try:
            endpoint.clear_halt()
        except self._usb_core.USBError:
            pass

        self._device = device
        self._endpoint = endpoint

    def close(self) -> None:
        if self._device is None:
            return
        if self._claimed is not None:
            try:
                self._usb_util.release_interface(self._device, self._claimed)
            except self._usb_core.USBError:
                pass
            self._claimed = None
        self._usb_util.dispose_resources(self._device)
        self._device = None
        self._endpoint = None

    def write(self, data: bytes, *, timeout: int = WRITE_TIMEOUT_MS) -> None:
        """Write bytes, clearing a halted endpoint once before giving up.

        Pass the whole page in one call. Splitting it across several writes
        looks safer and is not: bulk USB already handles flow control (a full
        buffer NAKs and the host controller retries in hardware, inside the
        single transfer), whereas separate transfers leave gaps that this
        firmware treats as the end of the raster and then errors on the next
        command.
        """
        if self._endpoint is None:
            raise EscPosError("Device is not open.")
        try:
            self._endpoint.write(data, timeout)
        except self._usb_core.USBError as first:
            try:
                self._endpoint.clear_halt()
            except self._usb_core.USBError:
                pass
            time.sleep(0.3)
            try:
                self._endpoint.write(data, timeout)
            except self._usb_core.USBError as second:
                raise EscPosError(
                    f"USB write failed ({second}). The printer buffer has most likely "
                    "overrun and the device needs a power cycle."
                ) from first

    def print_page(self, payload: bytes) -> None:
        """Initialise, send a page in one transfer, and feed the paper clear."""
        self.write(INIT + payload + FEED)


def text_payload(text: str, *, centred: bool = False) -> bytes:
    """Encode a short plain-text line. CP852 covers Polish on most firmware."""
    align = b"\x1b\x61\x01" if centred else b"\x1b\x61\x00"
    try:
        body = text.encode("cp852")
    except UnicodeEncodeError:
        body = text.encode("ascii", errors="replace")
    return align + body + b"\n"


def raster_payload(image: Image.Image, max_width: int) -> bytes:
    """Encode a 1-bit image as a single ``GS v 0`` raster command.

    ESC/POS packs 8 horizontal pixels per byte, most significant bit leftmost,
    and a **set** bit means a burnt dot — the inverse of Pillow's mode ``1``,
    where 0 is black.

    One command for the whole image, not a sequence of bands: splitting the
    raster across several commands makes this firmware abort partway with an
    I/O error, and the device then stays wedged until it is power-cycled.
    """
    if image.mode != "1":
        raise ValueError("raster_payload expects a 1-bit image.")
    if image.width > max_width:
        raise ValueError(
            f"Image is {image.width} dots wide but the head is {max_width}. "
            "Oversized rasters wedge the printer rather than erroring — resize first."
        )
    if image.height > 0xFFFF:
        raise ValueError(f"Image is {image.height} rows tall; GS v 0 allows at most 65535.")

    width_bytes = (image.width + 7) // 8
    padded = Image.new("1", (width_bytes * 8, image.height), WHITE)
    padded.paste(image, (0, 0))
    inverted = bytes(b ^ 0xFF for b in padded.tobytes())

    header = b"\x1d\x76\x30\x00" + bytes(
        [
            width_bytes & 0xFF,
            width_bytes >> 8,
            image.height & 0xFF,
            image.height >> 8,
        ]
    )
    return header + inverted


def qr_image(
    data: str,
    *,
    width_dots: int,
    module_dots: int = 8,
    error_correction: str = "M",
    quiet_zone: int = 4,
) -> Image.Image:
    """Render a QR code as a 1-bit bitmap centred on the paper width.

    The native ESC/POS QR command (``GS ( k``) is far cheaper on the wire, but
    plenty of budget firmware silently ignores it — this one does. Rasterising
    the symbol goes down the same path as any other image, which is known to
    work, at the cost of a few KB.

    ``module_dots`` is the size of one QR module; the symbol is shrunk if the
    requested size would overflow the head.
    """
    try:
        import segno
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EscPosError("segno is required to render QR codes (`uv add segno`).") from exc

    if not data:
        raise ValueError("QR data must not be empty.")
    if error_correction.upper() not in QR_ERROR_LEVELS:
        raise ValueError(
            f"error_correction must be one of {', '.join(QR_ERROR_LEVELS)}; got {error_correction!r}"
        )

    symbol = segno.make(data, error=error_correction.lower())
    matrix = [list(row) for row in symbol.matrix]
    modules = len(matrix) + 2 * quiet_zone

    # Shrink rather than overflow: an oversized raster wedges the printer.
    module_dots = max(1, min(module_dots, width_dots // modules))
    side = modules * module_dots

    symbol_img = Image.new("1", (side, side), WHITE)
    pixels = symbol_img.load()
    for y, row in enumerate(matrix):
        for x, dark in enumerate(row):
            if not dark:
                continue
            x0 = (x + quiet_zone) * module_dots
            y0 = (y + quiet_zone) * module_dots
            for dy in range(module_dots):
                for dx in range(module_dots):
                    pixels[x0 + dx, y0 + dy] = BLACK

    canvas = Image.new("1", (width_dots, side), WHITE)
    canvas.paste(symbol_img, ((width_dots - side) // 2, 0))
    return canvas


def fit_to_head(image: Image.Image, max_width: int, threshold: int = 200) -> Image.Image:
    """Scale a greyscale page to the head width and threshold it to 1-bit.

    Downscaling uses a box filter so each dot is the true area average of the
    pixels it covers; thresholding that average keeps strokes solid instead of
    letting a dither thin them out.
    """
    if image.mode != "L":
        image = image.convert("L")
    if image.width != max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.Resampling.BOX)
    return image.point(lambda p: 0 if p < threshold else 255, mode="L").convert("1")
