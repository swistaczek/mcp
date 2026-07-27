"""Tests for the ESC/POS USB transport.

No hardware is needed: payload encoding and image fitting are pure functions,
and the USB layer is exercised through fakes. The one test marked
``integration`` talks to a real printer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import escpos_printer as ep


class TestDeviceTable:
    def test_known_device_carries_a_head_width(self):
        entry = ep.KNOWN_ESCPOS_DEVICES[(0x0416, 0x5011)]
        assert entry["width_dots"] == 384

    def test_printer_reports_paper_width_in_mm(self):
        printer = ep.EscPosPrinter(0x0416, 0x5011, "Generic", 384)
        # 384 dots at 203 dpi is the 48 mm printable area of 58 mm paper.
        assert printer.width_mm == 48.0
        assert printer.usb_id == "0416:5011"

    def test_wider_head_reports_wider_paper(self):
        assert ep.EscPosPrinter(0x6868, 0x0200, "Zjiang", 576).width_mm == 72.1


class TestFindPrinters:
    def _device(self, vid, pid, serial="1234567890"):
        device = MagicMock()
        device.idVendor = vid
        device.idProduct = pid
        device.serial_number = serial
        return device

    def test_recognises_known_device(self):
        usb_core, usb_util = MagicMock(), MagicMock()
        usb_core.find.return_value = [self._device(0x0416, 0x5011)]
        with patch.object(ep, "_require_pyusb", return_value=(usb_core, usb_util)):
            printers = ep.find_escpos_printers()
        assert len(printers) == 1
        assert printers[0].width_dots == 384
        assert printers[0].serial == "1234567890"

    def test_ignores_unknown_device(self):
        usb_core, usb_util = MagicMock(), MagicMock()
        usb_core.find.return_value = [self._device(0x05AC, 0x1234)]
        with patch.object(ep, "_require_pyusb", return_value=(usb_core, usb_util)):
            assert ep.find_escpos_printers() == []

    def test_handles_unreadable_serial(self):
        device = self._device(0x0416, 0x5011)
        type(device).serial_number = property(lambda _self: (_ for _ in ()).throw(ValueError()))
        usb_core, usb_util = MagicMock(), MagicMock()
        usb_core.find.return_value = [device]
        with patch.object(ep, "_require_pyusb", return_value=(usb_core, usb_util)):
            printers = ep.find_escpos_printers()
        assert printers[0].serial is None


class TestRasterPayload:
    """One command per image — banding wedges this firmware."""

    def test_header_encodes_width_in_bytes_and_height_in_rows(self):
        image = Image.new("1", (384, 100), 255)
        payload = ep.raster_payload(image, 384)
        assert payload[:4] == b"\x1d\x76\x30\x00"
        width_bytes = payload[4] | (payload[5] << 8)
        height = payload[6] | (payload[7] << 8)
        assert width_bytes == 48
        assert height == 100

    def test_is_a_single_command_not_a_sequence(self):
        image = Image.new("1", (384, 600), 255)
        payload = ep.raster_payload(image, 384)
        assert payload.count(b"\x1d\x76\x30\x00") == 1
        assert len(payload) == 8 + 48 * 600

    def test_bit_polarity_is_inverted_for_escpos(self):
        # All-black in Pillow mode '1' is 0; ESC/POS wants 1 for a burnt dot.
        black = Image.new("1", (8, 1), 0)
        assert ep.raster_payload(black, 384)[8:] == b"\xff"
        white = Image.new("1", (8, 1), 255)
        assert ep.raster_payload(white, 384)[8:] == b"\x00"

    def test_narrow_image_is_padded_to_whole_bytes(self):
        image = Image.new("1", (12, 2), 255)
        payload = ep.raster_payload(image, 384)
        assert (payload[4] | (payload[5] << 8)) == 2
        assert len(payload) == 8 + 2 * 2

    def test_oversized_image_is_refused(self):
        image = Image.new("1", (576, 10), 255)
        with pytest.raises(ValueError, match="wedge the printer"):
            ep.raster_payload(image, 384)

    def test_overtall_image_is_refused(self):
        image = Image.new("1", (8, 70000), 255)
        with pytest.raises(ValueError, match="at most 65535"):
            ep.raster_payload(image, 384)

    def test_greyscale_input_is_refused(self):
        with pytest.raises(ValueError, match="1-bit image"):
            ep.raster_payload(Image.new("L", (8, 8), 255), 384)


class TestFitToHead:
    def test_scales_to_head_width(self):
        page = Image.new("L", (1000, 2000), 255)
        fitted = ep.fit_to_head(page, 384)
        assert fitted.width == 384
        assert fitted.height == 768
        assert fitted.mode == "1"

    def test_thresholds_rather_than_dithers(self):
        page = Image.new("L", (384, 10), 128)
        fitted = ep.fit_to_head(page, 384, threshold=200)
        assert fitted.histogram()[255] == 0      # all black, no scattered dots

    def test_light_grey_stays_white(self):
        page = Image.new("L", (384, 10), 250)
        assert ep.fit_to_head(page, 384, threshold=200).histogram()[0] == 0

    def test_already_correct_width_is_not_resized(self):
        page = Image.new("L", (384, 50), 255)
        assert ep.fit_to_head(page, 384).size == (384, 50)


class TestQrImage:
    def test_fits_the_paper_width(self):
        image = ep.qr_image("https://onet.pl", width_dots=384)
        assert image.width == 384
        assert image.mode == "1"
        assert image.height <= 384

    def test_has_both_dark_and_light_modules(self):
        image = ep.qr_image("https://onet.pl", width_dots=384)
        assert image.histogram()[0] > 0
        assert image.histogram()[255] > 0

    def test_module_size_is_clamped_to_fit(self):
        # 16 dots per module on a long payload would overflow 384 dots; the
        # symbol must shrink rather than be refused or clipped.
        image = ep.qr_image("x" * 200, width_dots=384, module_dots=16)
        assert image.width == 384
        assert image.height <= 384

    def test_larger_modules_make_a_larger_symbol(self):
        small = ep.qr_image("https://onet.pl", width_dots=384, module_dots=4)
        large = ep.qr_image("https://onet.pl", width_dots=384, module_dots=8)
        assert large.height > small.height

    def test_quiet_zone_leaves_a_white_border(self):
        image = ep.qr_image("https://onet.pl", width_dots=384, module_dots=6)
        assert image.getpixel((image.width // 2, 1)) == 255

    def test_empty_data_is_refused(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ep.qr_image("", width_dots=384)

    def test_bad_error_correction_is_refused(self):
        with pytest.raises(ValueError, match="error_correction"):
            ep.qr_image("x", width_dots=384, error_correction="Z")


class TestTextPayload:
    def test_plain_ascii(self):
        assert ep.text_payload("HELLO") == b"\x1b\x61\x00HELLO\n"

    def test_centred_sets_alignment(self):
        assert ep.text_payload("HI", centred=True).startswith(b"\x1b\x61\x01")

    def test_polish_characters_survive_via_cp852(self):
        payload = ep.text_payload("Zażółć")
        assert payload.endswith(b"\n")
        assert b"?" not in payload

    def test_unencodable_text_degrades_rather_than_raising(self):
        payload = ep.text_payload("日本語")
        assert payload.endswith(b"\n")


class TestDevice:
    def _open_device(self):
        usb_core, usb_util = MagicMock(), MagicMock()
        usb_core.USBError = RuntimeError
        endpoint = MagicMock()
        usb_util.find_descriptor.return_value = endpoint
        device = MagicMock()
        usb_core.find.return_value = device
        printer = ep.EscPosPrinter(0x0416, 0x5011, "Generic", 384)
        with patch.object(ep, "_require_pyusb", return_value=(usb_core, usb_util)):
            dev = ep.EscPosDevice(printer)
            dev.open()
        return dev, endpoint, usb_core, usb_util

    def test_open_claims_the_interface(self):
        _dev, _ep_, _core, usb_util = self._open_device()
        # CUPS probes the same device; exclusive ownership avoids collisions
        # partway through a multi-second raster transfer.
        assert usb_util.claim_interface.called

    def test_print_page_wraps_payload_in_init_and_feed(self):
        dev, endpoint, _core, _util = self._open_device()
        dev.print_page(b"BODY")
        written = endpoint.write.call_args[0][0]
        assert written.startswith(ep.INIT)
        assert written.endswith(ep.FEED)
        assert b"BODY" in written

    def test_page_goes_out_as_one_transfer(self):
        dev, endpoint, _core, _util = self._open_device()
        endpoint.write.reset_mock()
        dev.print_page(b"x" * 20000)
        assert endpoint.write.call_count == 1

    def test_write_retries_once_after_clearing_a_halt(self):
        dev, endpoint, _core, _util = self._open_device()
        endpoint.write.side_effect = [RuntimeError("stall"), None]
        dev.write(b"data")
        assert endpoint.clear_halt.called
        assert endpoint.write.call_count == 2

    def test_persistent_failure_raises_with_recovery_advice(self):
        dev, endpoint, _core, _util = self._open_device()
        endpoint.write.side_effect = RuntimeError("EIO")
        with pytest.raises(ep.EscPosError, match="power cycle"):
            dev.write(b"data")

    def test_close_releases_the_interface(self):
        dev, _endpoint, _core, usb_util = self._open_device()
        dev.close()
        assert usb_util.release_interface.called
        assert usb_util.dispose_resources.called

    def test_write_before_open_is_refused(self):
        printer = ep.EscPosPrinter(0x0416, 0x5011, "Generic", 384)
        with patch.object(ep, "_require_pyusb", return_value=(MagicMock(), MagicMock())):
            dev = ep.EscPosDevice(printer)
        with pytest.raises(ep.EscPosError, match="not open"):
            dev.write(b"x")

    def test_missing_device_is_reported_clearly(self):
        usb_core, usb_util = MagicMock(), MagicMock()
        usb_core.USBError = RuntimeError
        usb_core.find.return_value = None
        printer = ep.EscPosPrinter(0x0416, 0x5011, "Generic", 384)
        with patch.object(ep, "_require_pyusb", return_value=(usb_core, usb_util)):
            with pytest.raises(ep.EscPosError, match="not connected"):
                ep.EscPosDevice(printer).open()

    def test_wedged_device_gets_a_power_cycle_hint(self):
        usb_core, usb_util = MagicMock(), MagicMock()
        usb_core.USBError = RuntimeError
        device = MagicMock()
        device.set_configuration.side_effect = RuntimeError("Other error")
        usb_core.find.return_value = device
        printer = ep.EscPosPrinter(0x0416, 0x5011, "Generic", 384)
        with patch.object(ep, "_require_pyusb", return_value=(usb_core, usb_util)):
            with pytest.raises(ep.EscPosError, match="power-cycle"):
                ep.EscPosDevice(printer).open()


@pytest.mark.integration
class TestLiveEscPos:
    """Requires an ESC/POS receipt printer on USB."""

    def test_printer_is_discoverable(self):
        try:
            printers = ep.find_escpos_printers()
        except ep.EscPosError as exc:
            pytest.skip(str(exc))
        if not printers:
            pytest.skip("no ESC/POS printer attached")
        assert printers[0].width_dots in (384, 576)
