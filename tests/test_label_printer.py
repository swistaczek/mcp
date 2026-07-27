"""Tests for the thermal label printer server.

Everything here runs without hardware: CUPS output is fed in as captured text
and the rendering path is exercised against the bundled InPost fixture. Only
the tests marked ``integration`` touch a real printer.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

import label_printer as lp

FIXTURES = Path(__file__).parent / "fixtures"
INPOST_LABEL = FIXTURES / "inpost_label_4x6.pdf"

HAS_PDFTOPPM = shutil.which("pdftoppm") is not None
needs_poppler = pytest.mark.skipif(not HAS_PDFTOPPM, reason="pdftoppm not installed")

#: ``_run`` is the single seam for every shell-out, so a test that fakes CUPS
#: would also swallow the rasteriser. Keeping the real callable lets the fakes
#: intercept only the commands they actually stand in for.
REAL_RUN = lp._run
CUPS_COMMANDS = {"lp", "lpinfo", "lpstat", "lpoptions", "lpadmin", "cancel"}


def _black(image: Image.Image) -> int:
    """Count burnt dots. A 1-bit histogram fills only bins 0 and 255."""
    return image.histogram()[0]


def _white(image: Image.Image) -> int:
    return image.histogram()[255]


# Captured from a macOS 26 machine with a Zebra TLP2844 on USB.
LPINFO_OUTPUT = """\
network https
network ipps
direct usb://Zebra/TLP2844?serial=41A071000746
network smb
network socket
network dnssd://EPSON%20L3230%20Series%20%40%20Mac._ipp._tcp.local./cups?uuid=6b996b56
"""

LPSTAT_V_OUTPUT = """\
device for OKI_B412_5CDA2F: dnssd://OKI%20B412._ipp._tcp.local./cups
device for Zebra_TLP2844: usb://Zebra/TLP2844?serial=41A071000746
"""

LPSTAT_P_OUTPUT = """\
printer OKI_B412_5CDA2F is idle.  enabled since Mon Jul 27 12:22:21 2026
printer Zebra_TLP2844 now printing Zebra_TLP2844-105.  enabled since Mon Jul 27 12:26:33 2026
"""

LPOPTIONS_OUTPUT = """\
PageSize/Media Size: w288h360 w288h432 w288h936 Custom.WIDTHxHEIGHT
Resolution/Resolution: *203dpi 300dpi 600dpi
MediaType/Media Type: *Saved Thermal Direct
Darkness/Darkness: *-1 1 2 30
zePrintRate/Print Rate: *Default 1 1.5 2 6
"""


class TestClassify:
    """Device URIs and queue names are mapped to a make and a driver."""

    def test_zebra_epl2_model(self):
        assert lp._classify("usb://Zebra/TLP2844?serial=41A0") == ("zebra", "zebra_epl2")

    def test_zebra_zpl_model(self):
        assert lp._classify("usb://Zebra/ZT230") == ("zebra", "zebra_zpl")

    def test_zebra_z_variant_is_zpl(self):
        # TLP2844-Z ships ZPL firmware despite sharing the EPL2 model number.
        assert lp._classify("usb://Zebra/TLP2844-Z") == ("zebra", "zebra_zpl")

    def test_dymo(self):
        assert lp._classify("usb://DYMO/LabelWriter%20450") == ("dymo", "dymo")

    def test_other_thermal_make(self):
        assert lp._classify("usb://TSC/TE200") == ("tsc", "generic_zpl")

    def test_office_printer_is_not_thermal(self):
        assert lp._classify("dnssd://OKI%20B412._ipp._tcp.local./cups") is None

    def test_case_insensitive(self):
        assert lp._classify("USB://ZEBRA/tlp2844") == ("zebra", "zebra_epl2")


class TestModelFromUri:
    def test_usb_uri(self):
        assert lp._model_from_uri("usb://Zebra/TLP2844?serial=41A071000746") == "TLP2844"

    def test_percent_encoded_space(self):
        assert lp._model_from_uri("usb://DYMO/LabelWriter%20450") == "LabelWriter 450"

    def test_no_path_segment(self):
        assert lp._model_from_uri("usb://Zebra") == "Zebra"


class TestCupsParsers:
    def test_lpinfo_keeps_only_uris(self):
        devices = lp._parse_lpinfo(LPINFO_OUTPUT)
        assert ("direct", "usb://Zebra/TLP2844?serial=41A071000746") in devices
        assert all("://" in uri for _, uri in devices)

    def test_lpstat_v(self):
        queues = lp._parse_lpstat_v(LPSTAT_V_OUTPUT)
        assert queues["Zebra_TLP2844"] == "usb://Zebra/TLP2844?serial=41A071000746"
        assert len(queues) == 2

    def test_lpstat_p(self):
        states = lp._parse_lpstat_p(LPSTAT_P_OUTPUT)
        assert states["OKI_B412_5CDA2F"].startswith("is idle")
        assert "now printing" in states["Zebra_TLP2844"]

    def test_lpoptions(self):
        with patch.object(lp, "_run") as run:
            run.return_value.stdout = LPOPTIONS_OUTPUT
            options = lp.supported_options("Zebra_TLP2844")
        assert "w288h432" in options["PageSize"]
        assert "30" in options["Darkness"]
        # The leading '*' marking the default is stripped from the value.
        assert "203dpi" in options["Resolution"]


class TestDiscovery:
    """Devices and queues are merged on device URI, not reported twice."""

    def _fake_run(self, cmd, check=False):
        outputs = {
            ("lpinfo", "-v"): LPINFO_OUTPUT,
            ("lpstat", "-v"): LPSTAT_V_OUTPUT,
            ("lpstat", "-p"): LPSTAT_P_OUTPUT,
            ("lpstat", "-d"): "system default destination: OKI_B412_5CDA2F\n",
        }

        class Result:
            stdout = outputs.get(tuple(cmd), "")
            stderr = ""
            returncode = 0

        return Result()

    def test_finds_only_the_thermal_printer(self):
        with patch.object(lp, "_run", side_effect=self._fake_run):
            printers = lp.discover_thermal_printers()
        assert len(printers) == 1
        assert printers[0].model == "TLP2844"
        assert printers[0].queue == "Zebra_TLP2844"
        assert printers[0].language == "EPL2"
        assert printers[0].is_default is False

    def test_resolve_picks_the_sole_printer(self):
        with patch.object(lp, "_run", side_effect=self._fake_run):
            assert lp.resolve_printer().queue == "Zebra_TLP2844"

    def test_resolve_matches_on_model_fragment(self):
        with patch.object(lp, "_run", side_effect=self._fake_run):
            assert lp.resolve_printer("tlp2844").queue == "Zebra_TLP2844"

    def test_resolve_rejects_unknown_name(self):
        with patch.object(lp, "_run", side_effect=self._fake_run):
            with pytest.raises(lp.PrinterError, match="No thermal printer matches"):
                lp.resolve_printer("brother")

    def test_no_printers_raises(self):
        with patch.object(lp, "_run") as run:
            run.return_value.stdout = ""
            with pytest.raises(lp.PrinterError, match="No thermal label printer detected"):
                lp.resolve_printer()

    def test_ambiguous_selection_raises(self):
        two = LPINFO_OUTPUT + "direct usb://DYMO/LabelWriter%20450\n"

        def fake(cmd, check=False):
            class Result:
                stdout = two if tuple(cmd) == ("lpinfo", "-v") else ""
                stderr = ""
                returncode = 0

            return Result()

        with patch.object(lp, "_run", side_effect=fake):
            with pytest.raises(lp.PrinterError, match="Multiple thermal printers"):
                lp.resolve_printer()


class TestQueueName:
    def test_sanitises_illegal_characters(self):
        printer = lp.ThermalPrinter(
            device_uri="usb://DYMO/LabelWriter%20450",
            make="Dymo",
            model="LabelWriter 450",
            driver_key="dymo",
        )
        assert lp._queue_name_for(printer) == "Dymo_LabelWriter_450"

    def test_existing_queue_is_reused(self):
        printer = lp.ThermalPrinter(
            device_uri="usb://Zebra/TLP2844",
            make="Zebra",
            model="TLP2844",
            driver_key="zebra_epl2",
            queue="Zebra_TLP2844",
        )
        with patch.object(lp, "_run") as run:
            assert lp.ensure_queue(printer) == "Zebra_TLP2844"
        run.assert_not_called()

    def test_driver_failure_is_reported(self):
        printer = lp.ThermalPrinter(
            device_uri="usb://Zebra/TLP2844",
            make="Zebra",
            model="TLP2844",
            driver_key="zebra_epl2",
        )
        with patch.object(lp, "_run") as run:
            run.return_value.returncode = 1
            run.return_value.stderr = "lpadmin: Raw queues are no longer supported on macOS."
            with pytest.raises(lp.PrinterError, match="Could not create a CUPS queue"):
                lp.ensure_queue(printer)


class TestLabelSize:
    def test_preset(self):
        assert lp.parse_label_size("4x6") == (288.0, 432.0)

    def test_bare_dimensions_are_inches(self):
        assert lp.parse_label_size("4x6") == lp.parse_label_size("4x6in")

    def test_millimetres(self):
        width, height = lp.parse_label_size("100x150mm")
        assert round(width, 1) == 283.5
        assert round(height, 1) == 425.2

    def test_points_pass_through(self):
        assert lp.parse_label_size("288x432pt") == (288.0, 432.0)

    def test_whitespace_and_case(self):
        assert lp.parse_label_size(" A6 ") == lp.LABEL_PRESETS["a6"]

    def test_garbage_rejected(self):
        with pytest.raises(ValueError, match="Unrecognised label size"):
            lp.parse_label_size("huge")


class TestPageSizeOption:
    def test_uses_the_ppd_keyword_when_available(self):
        assert lp.page_size_option(288, 432, ["w288h360", "w288h432"]) == "w288h432"

    def test_falls_back_to_custom(self):
        assert lp.page_size_option(283.5, 425.2, ["w288h432"]) == "Custom.284x425"


class TestFitGeometry:
    def test_centres_and_shrinks_oversized_artwork(self):
        # The InPost label is 297x435 pt — wider than the 4 in print head.
        scale, dx, dy, rotate = lp.fit_geometry(297, 435, 288, 432, 8)
        assert rotate is False
        assert scale < 1
        assert round(297 * scale) <= 288 - 2 * 8
        assert dx == pytest.approx(8.0)
        assert dy > 0

    def test_rotates_when_orientation_disagrees(self):
        scale, _dx, _dy, rotate = lp.fit_geometry(435, 297, 288, 432, 8)
        assert rotate is True
        assert scale < 1

    def test_no_rotation_when_orientation_matches(self):
        _s, _dx, _dy, rotate = lp.fit_geometry(100, 200, 288, 432, 8)
        assert rotate is False

    def test_margin_is_respected_on_both_axes(self):
        scale, dx, dy, _r = lp.fit_geometry(288, 432, 288, 432, 10)
        assert dx >= 10 - 0.001
        assert dy >= 10 - 0.001
        assert scale < 1


class TestBilevelConversion:
    """Thresholding, not dithering — grey must resolve to solid black or white."""

    def test_output_is_one_bit(self):
        grey = Image.new("L", (100, 200), 128)
        result = lp._to_bilevel(grey, (812, 1218), 22, 200, 0)
        assert result.mode == "1"
        assert result.size == (812, 1218)

    def test_mid_grey_below_threshold_becomes_black(self):
        grey = Image.new("L", (100, 200), 128)
        result = lp._to_bilevel(grey, (200, 400), 0, 200, 0)
        assert _white(result) == 0

    def test_light_grey_above_threshold_becomes_white(self):
        grey = Image.new("L", (100, 200), 240)
        result = lp._to_bilevel(grey, (200, 400), 0, 200, 0)
        assert _black(result) == 0

    def test_threshold_controls_ink(self):
        grey = Image.new("L", (100, 200), 210)
        light = lp._to_bilevel(grey, (200, 400), 0, 200, 0)
        heavy = lp._to_bilevel(grey, (200, 400), 0, 235, 0)
        assert _black(light) == 0
        assert _white(heavy) == 0

    def test_margin_leaves_white_border(self):
        black = Image.new("L", (100, 200), 0)
        result = lp._to_bilevel(black, (200, 400), 20, 200, 0)
        assert result.getpixel((2, 2)) == 255
        assert result.getpixel((100, 200)) == 0

    def test_landscape_source_is_rotated_onto_portrait_stock(self):
        wide = Image.new("L", (400, 100), 0)
        result = lp._to_bilevel(wide, (200, 400), 0, 200, 0)
        # After rotation the artwork is taller than it is wide, so it fills the
        # label's height rather than being squeezed into a thin band.
        black_rows = {y for y in range(400) if result.getpixel((100, y)) == 0}
        assert len(black_rows) > 200

    def test_bold_thickens_strokes(self):
        page = Image.new("L", (200, 400), 255)
        for y in range(400):
            page.putpixel((100, y), 0)
        plain = lp._to_bilevel(page, (200, 400), 0, 200, 0)
        bolded = lp._to_bilevel(page, (200, 400), 0, 200, 1)
        assert _black(bolded) > _black(plain)


@needs_poppler
class TestRenderLabel:
    """End-to-end rendering of the real InPost label, no printer involved."""

    def test_renders_to_device_grid(self, tmp_path):
        out, info = lp.render_label(INPOST_LABEL, 288, 432, output=tmp_path / "label.pdf")
        assert out.exists()
        assert info["pages"] == 1
        assert info["dots"] == "812x1218"
        assert info["page_size_pt"] == "288x432"
        assert info["dpi"] == 203

    def test_ink_coverage_is_plausible(self, tmp_path):
        _out, info = lp.render_label(INPOST_LABEL, 288, 432, output=tmp_path / "label.pdf")
        # A courier label is mostly white with dense barcode blocks.
        assert 0.05 < info["ink_coverage"] < 0.40

    def test_higher_threshold_lays_down_more_ink(self, tmp_path):
        _o1, light = lp.render_label(
            INPOST_LABEL, 288, 432, threshold=120, output=tmp_path / "a.pdf"
        )
        _o2, heavy = lp.render_label(
            INPOST_LABEL, 288, 432, threshold=245, output=tmp_path / "b.pdf"
        )
        assert heavy["ink_coverage"] > light["ink_coverage"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(lp.PrinterError, match="Label file not found"):
            lp.render_label(tmp_path / "nope.pdf", 288, 432)

    def test_unsupported_format_raises(self, tmp_path):
        odd = tmp_path / "label.docx"
        odd.write_bytes(b"not a label")
        with pytest.raises(lp.PrinterError, match="Unsupported label format"):
            lp.render_label(odd, 288, 432)

    def test_image_input_is_accepted(self, tmp_path):
        png = tmp_path / "label.png"
        Image.new("L", (400, 600), 0).save(png)
        _out, info = lp.render_label(png, 288, 432, output=tmp_path / "out.pdf")
        assert info["pages"] == 1
        assert info["ink_coverage"] > 0.5


@needs_poppler
class TestPrintFile:
    """The lp invocation is assembled correctly without submitting a job."""

    #: Only CUPS commands are faked. ``pdftoppm`` is let through to the real
    #: subprocess so the rendering half of the pipeline stays honest.
    def _fake_run(self, cmd, check=False):
        outputs = {
            ("lpinfo", "-v"): LPINFO_OUTPUT,
            ("lpstat", "-v"): LPSTAT_V_OUTPUT,
            ("lpstat", "-p"): LPSTAT_P_OUTPUT,
            ("lpstat", "-d"): "system default destination: OKI_B412_5CDA2F\n",
            ("lpoptions", "-p", "Zebra_TLP2844", "-l"): LPOPTIONS_OUTPUT,
        }
        if cmd[0] not in CUPS_COMMANDS:
            return REAL_RUN(cmd, check=check)

        class Result:
            stdout = outputs.get(tuple(cmd), "")
            stderr = ""
            returncode = 0

        return Result()

    def test_dry_run_builds_expected_command(self):
        with patch.object(lp, "_run", side_effect=self._fake_run):
            outcome = lp.print_file(INPOST_LABEL, dry_run=True)
        command = outcome.render["command"]
        assert "-d Zebra_TLP2844" in command
        assert "PageSize=w288h432" in command
        assert "Darkness=30" in command
        assert "zePrintRate=1" in command
        assert outcome.job_id == "(dry-run)"

    def test_unsupported_options_are_omitted(self):
        def fake(cmd, check=False):
            result = self._fake_run(cmd)
            if tuple(cmd)[:2] == ("lpoptions", "-p"):
                result.stdout = "PageSize/Media Size: w288h432 Custom.WIDTHxHEIGHT\n"
            return result

        with patch.object(lp, "_run", side_effect=fake):
            outcome = lp.print_file(INPOST_LABEL, dry_run=True)
        assert "Darkness" not in outcome.options
        assert "zePrintRate" not in outcome.options

    def test_copies_are_passed_through(self):
        with patch.object(lp, "_run", side_effect=self._fake_run):
            outcome = lp.print_file(INPOST_LABEL, copies=3, dry_run=True)
        assert "-n 3" in outcome.render["command"]
        assert outcome.copies == 3

    def test_non_bilevel_path_delegates_scaling_to_cups(self):
        with patch.object(lp, "_run", side_effect=self._fake_run):
            outcome = lp.print_file(INPOST_LABEL, bilevel=False, dry_run=True)
        assert outcome.options["fit-to-page"] == "true"
        assert outcome.render.get("dots") is None

    def test_job_id_is_parsed_from_lp_output(self):
        def fake(cmd, check=False):
            result = self._fake_run(cmd)
            if cmd[0] == "lp":
                result.stdout = "request id is Zebra_TLP2844-105 (1 file(s))\n"
            return result

        with patch.object(lp, "_run", side_effect=fake):
            outcome = lp.print_file(INPOST_LABEL)
        assert outcome.job_id == "Zebra_TLP2844-105"

    def test_lp_failure_raises(self):
        def fake(cmd, check=False):
            result = self._fake_run(cmd)
            if cmd[0] == "lp":
                result.returncode = 1
                result.stderr = "lp: Destination is not accepting jobs."
            return result

        with patch.object(lp, "_run", side_effect=fake):
            with pytest.raises(lp.PrinterError, match="not accepting jobs"):
                lp.print_file(INPOST_LABEL)


class TestQueueStatus:
    def test_reports_state_and_jobs(self):
        def fake(cmd, check=False):
            class Result:
                returncode = 0
                stderr = ""
                stdout = (
                    LPSTAT_P_OUTPUT
                    if tuple(cmd)[:2] == ("lpstat", "-p")
                    else "Zebra_TLP2844-105 ernest 41984 Mon Jul 27 12:26:33 2026\n"
                )

            return Result()

        with patch.object(lp, "_run", side_effect=fake):
            status = lp.queue_status("Zebra_TLP2844")
        assert status["idle"] is False
        assert status["jobs"][0]["job_id"] == "Zebra_TLP2844-105"
        assert status["jobs"][0]["owner"] == "ernest"


class TestCli:
    # `detect` reaches for USB as well as CUPS, so both discovery paths must be
    # stubbed — otherwise a printer plugged into the dev machine leaks in.
    def test_detect_reports_failure_exit_code(self, capsys):
        with (
            patch.object(lp, "discover_thermal_printers", return_value=[]),
            patch.object(lp.escpos_printer, "find_escpos_printers", return_value=[]),
        ):
            assert lp.main(["detect"]) == 1
        assert "No thermal printer" in capsys.readouterr().out

    def test_detect_json_output(self, capsys):
        printer = lp.ThermalPrinter(
            device_uri="usb://Zebra/TLP2844",
            make="Zebra",
            model="TLP2844",
            driver_key="zebra_epl2",
            queue="Zebra_TLP2844",
        )
        with (
            patch.object(lp, "discover_thermal_printers", return_value=[printer]),
            patch.object(lp.escpos_printer, "find_escpos_printers", return_value=[]),
        ):
            assert lp.main(["--json", "detect"]) == 0
        assert '"language": "EPL2"' in capsys.readouterr().out

    def test_detect_lists_escpos_printers_too(self, capsys):
        receipt = lp.escpos_printer.EscPosPrinter(0x0416, 0x5011, "Generic ESC/POS", 384)
        with (
            patch.object(lp, "discover_thermal_printers", return_value=[]),
            patch.object(lp.escpos_printer, "find_escpos_printers", return_value=[receipt]),
        ):
            assert lp.main(["detect"]) == 0
        assert "ESC/POS" in capsys.readouterr().out

    @needs_poppler
    def test_render_writes_a_pdf(self, tmp_path, capsys):
        out = tmp_path / "rendered.pdf"
        assert lp.main(["render", str(INPOST_LABEL), "-o", str(out)]) == 0
        assert out.exists()
        assert "812x1218" in capsys.readouterr().out

    def test_bad_label_size_exits_nonzero(self, tmp_path, capsys):
        code = lp.main(["render", str(INPOST_LABEL), "-o", str(tmp_path / "x.pdf"), "-s", "huge"])
        assert code == 1
        assert "Unrecognised label size" in capsys.readouterr().err


@pytest.mark.integration
class TestLivePrinter:
    """Requires a thermal printer attached to this machine."""

    def test_a_thermal_printer_is_reachable(self):
        printers = lp.discover_thermal_printers()
        if not printers:
            pytest.skip("no thermal printer attached")
        assert printers[0].device_uri.startswith(("usb://", "socket://", "dnssd://"))

    @needs_poppler
    def test_dry_run_against_real_cups(self):
        try:
            outcome = lp.print_file(INPOST_LABEL, dry_run=True)
        except lp.PrinterError as exc:
            pytest.skip(str(exc))
        assert outcome.options["PageSize"] == "w288h432"
        assert outcome.render["dots"] == "812x1218"
