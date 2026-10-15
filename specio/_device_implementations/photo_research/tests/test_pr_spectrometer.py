"""The PR-655/670 driver, against a scripted remote-mode device."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import serial

from specio._device_implementations.photo_research import (
    MeasurementOutOfRange,
    PRCommandError,
    PRSpectrometer,
    PRStatus,
)
from specio._device_implementations.photo_research import (
    _pr_spectrometers as pr,
)
from specio.common.exceptions import DeviceError

if TYPE_CHECKING:
    from collections.abc import Callable

WAVELENGTHS = range(380, 781, 4)

# Replies recorded from a PR-655 (serial 65081802, software 2.51) reading
# a ~1200 cd/m² white LED tile, 2026-10-15.
RECORDED = Path(__file__).parent / "data"


def spectrum_reply(code: str = "00000", value: float = 1.0e-3) -> bytes:
    """An ``M5`` reply in the manual's format."""
    lines = [f"{code},0,5.520e+002,4.016e-01,1.115e+15"]
    lines += [f"{nm},{value:.3e}" for nm in WAVELENGTHS]
    return ("\r\n".join(lines) + "\r\n").encode()


class FakePort:
    """A remote-mode PR-6xx: answers each command from a script.

    `script` maps a command to its reply, or to a list of replies served
    in order. Unscripted commands answer ``-1000`` (illegal command).
    """

    def __init__(self, script: dict[str, bytes | list[bytes]]) -> None:
        self.script = {k: v if isinstance(v, list) else [v] for k, v in script.items()}
        self.written = bytearray()
        self.writes = 0
        self.commands: list[str] = []
        self.closed = False
        self._pending = b""
        self._line = bytearray()

    def write(self, data: bytes, /) -> int:
        assert len(data) == 1, "the device needs one character per write"
        self.written += data
        self.writes += 1
        self._line += data
        if bytes(self._line) == b"PHOTO":
            self._answer("PHOTO")
        elif data == b"\r":
            self._answer(self._line[:-1].decode())
        elif bytes(self._line) == b"Q":
            self.commands.append("Q")
            self._line.clear()
        return 1

    def _answer(self, command: str) -> None:
        self.commands.append(command)
        self._line.clear()
        replies = self.script.get(command)
        if replies is None:
            self._pending = b"-1000\r\n"
        elif len(replies) > 1:
            self._pending = replies.pop(0)
        else:
            self._pending = replies[0]

    def read(self, size: int = 1, /) -> bytes:
        out, self._pending = self._pending[:size], self._pending[size:]
        return out

    def reset_input_buffer(self) -> None:
        self._pending = b""

    def close(self) -> None:
        self.closed = True


def ticking_clock(step: float = 0.1) -> Callable[[], float]:
    """A clock that advances on every read, so waits end without sleeping."""
    now = [0.0]

    def clock() -> float:
        now[0] += step
        return now[0]

    return clock


BASE_SCRIPT: dict[str, bytes | list[bytes]] = {
    "PHOTO": b" REMOTE MODE",
    "SU1": b"00000\r\n",
    "SS1": b"00000\r\n",
    "SS0": b"00000\r\n",
    "D110": b"00000,65041234\r\n",
    "D111": b"00000,PR-655\r\n",
    "D114": b"00000,2.22D\r\n",
    "D13": b"00000,Normal,1250 msec\r\n",
    "M5": spectrum_reply(),
}


def connect(**overrides: bytes | list[bytes]) -> tuple[PRSpectrometer, FakePort]:
    port = FakePort({**BASE_SCRIPT, **overrides})
    device = PRSpectrometer(port, sleep=lambda _: None, clock=ticking_clock())
    return device, port


class TestConnecting:
    def test_it_enters_remote_mode_then_sets_si_units_and_auto_sync(self) -> None:
        device, port = connect()

        assert port.commands == ["PHOTO", "SU1", "SS1"]
        assert device.synchronised

    def test_photo_is_sent_without_a_terminator(self) -> None:
        _, port = connect()

        assert port.written.startswith(b"PHOTOSU1\r")

    def test_without_sync_it_selects_no_sync(self) -> None:
        port = FakePort(dict(BASE_SCRIPT))
        device = PRSpectrometer(
            port, sync=False, sleep=lambda _: None, clock=ticking_clock()
        )

        assert port.commands[-1] == "SS0"
        assert not device.synchronised

    def test_a_device_that_never_answers_is_refused_and_released(self) -> None:
        port = FakePort({"PHOTO": b"", "D111": b""})

        with pytest.raises(DeviceError, match="remote mode"):
            PRSpectrometer(port, sleep=lambda _: None, clock=ticking_clock())
        assert port.closed

    def test_a_device_already_in_remote_mode_is_accepted(self) -> None:
        """No banner a second time, but it answers commands."""
        port = FakePort({**BASE_SCRIPT, "PHOTO": b""})

        PRSpectrometer(port, sleep=lambda _: None, clock=ticking_clock())

        assert port.commands[:2] == ["PHOTO", "D111"]


class TestIdentity:
    def test_it_reads_model_serial_and_software_version(self) -> None:
        device, _ = connect()

        assert device.manufacturer == "Photo Research"
        assert device.model == "PR-655"
        assert device.serial_number == "65041234"
        assert device.firmware == "2.22D"
        assert device.readable_id == "PR-655 - 65041234"

    def test_a_version_reply_that_is_not_a_version_is_no_firmware(self) -> None:
        device, _ = connect(D114=b"00000,0,MS-75,Primary\r\n")

        assert device.firmware is None


class TestMeasuring:
    def test_it_returns_the_spectrum_at_native_sampling(self) -> None:
        device, _ = connect()

        m = device.measure()

        assert m.spd.shape.start == 380
        assert m.spd.shape.end == 780
        assert m.spd.shape.interval == 4
        assert len(m.spd.values) == len(WAVELENGTHS)
        assert m.XYZ[1] > 0

    def test_luminance_is_683_times_the_photopic_integral(self) -> None:
        """Spectral radiance in W/(sr·m²·nm) gives Y in cd/m²."""
        device, _ = connect(M5=spectrum_reply(value=1.0e-3))

        y = device.measure().XYZ[1]

        # A flat 1 mW/(sr·m²·nm) spectrum: ∫ȳ dλ ≈ 106.9 nm.
        assert y == pytest.approx(683 * 1.0e-3 * 106.9, rel=0.01)

    def test_exposure_is_the_device_s_own_in_seconds(self) -> None:
        device, _ = connect()

        assert device.measure().exposure == pytest.approx(1.25)

    def test_an_unreadable_exposure_is_the_unknown_sentinel(self) -> None:
        device, _ = connect(D13=b"-1000\r\n")

        assert device.measure().exposure == -1

    @pytest.mark.parametrize("code", [PRStatus.WEAK_LIGHT, PRStatus.LIGHT_OVERLOAD])
    def test_light_out_of_range_is_its_own_result(self, code: int) -> None:
        device, port = connect(M5=f"{code:05d}\r\n".encode())

        with pytest.raises(MeasurementOutOfRange) as caught:
            device.measure()

        assert caught.value.code == code
        assert port.commands.count("M5") == 1, "a result is not retried"

    def test_out_of_range_is_matched_by_name_downstream(self) -> None:
        """display-measure recognises the type by its name, not an import."""
        assert MeasurementOutOfRange.__name__ == "MeasurementOutOfRange"
        assert issubclass(MeasurementOutOfRange, PRCommandError)
        assert issubclass(MeasurementOutOfRange, DeviceError)

    def test_any_other_status_is_a_command_error(self) -> None:
        device, _ = connect(M5=b"-0012\r\n")

        with pytest.raises(PRCommandError) as caught:
            device.measure()

        assert not isinstance(caught.value, MeasurementOutOfRange)
        assert caught.value.code == PRStatus.ADAPTIVE_TIMEOUT

    def test_a_sync_failure_falls_back_to_unsynchronised(self) -> None:
        device, port = connect(M5=[b"-0010\r\n", spectrum_reply()])

        m = device.measure()

        measuring = [c for c in port.commands if c in ("M5", "SS0", "SS1")]
        assert measuring == ["SS1", "M5", "SS0", "M5"]
        assert not device.synchronised
        assert len(m.spd.values) == len(WAVELENGTHS)

    def test_a_sync_failure_when_unsynchronised_is_raised(self) -> None:
        port = FakePort({**BASE_SCRIPT, "M5": b"-0003\r\n"})
        device = PRSpectrometer(
            port, sync=False, sleep=lambda _: None, clock=ticking_clock()
        )

        with pytest.raises(PRCommandError):
            device.measure()

    def test_unevenly_spaced_wavelengths_are_refused(self) -> None:
        reply = b"00000,0,5.5e+02,1.0,1.0\r\n380,1e-3\r\n384,1e-3\r\n390,1e-3\r\n"
        device, _ = connect(M5=reply)

        with pytest.raises(DeviceError, match="evenly"):
            device.measure()

    def test_a_garbled_spectrum_line_is_refused(self) -> None:
        reply = b"00000,0,5.5e+02,1.0,1.0\r\n380,1e-3\r\nREMOTE\r\n"
        device, _ = connect(M5=reply)

        with pytest.raises(DeviceError, match="unexpected line"):
            device.measure()


class TestRecordedDevice:
    """Replies captured from a real PR-655, played back."""

    @staticmethod
    def recorded() -> tuple[PRSpectrometer, FakePort]:
        script: dict[str, bytes | list[bytes]] = {
            cmd: (RECORDED / f"{cmd}.bin").read_bytes()
            for cmd in ("M5", "D13", "D110", "D111", "D114")
        }
        port = FakePort({**BASE_SCRIPT, **script})
        return (
            PRSpectrometer(port, sleep=lambda _: None, clock=ticking_clock()),
            port,
        )

    def test_identity(self) -> None:
        device, _ = self.recorded()

        assert device.readable_id == "PR655 - 65081802"
        assert device.firmware == "2.51"

    def test_the_spectrum_reproduces_the_device_s_own_tristimulus(self) -> None:
        """XYZ integrated from M5 matches the instrument's D2 reading."""
        device, _ = self.recorded()
        d2 = (RECORDED / "D2.bin").read_bytes().decode().strip().split(",")
        expected = [float(v) for v in d2[2:]]

        m = device.measure()

        assert m.spd.shape.start == 380
        assert m.spd.shape.end == 780
        assert m.spd.shape.interval == 4
        assert list(m.XYZ) == pytest.approx(expected, rel=2e-3)
        assert m.exposure == pytest.approx(0.067)


class TestClosing:
    def test_it_leaves_remote_mode_and_closes_once(self) -> None:
        device, port = connect()

        device.close()
        device.close()

        assert port.commands.count("Q") == 1
        assert port.closed

    def test_as_a_context_manager(self) -> None:
        device, port = connect()

        with device:
            pass

        assert port.commands[-1] == "Q"
        assert port.closed


class TestDiscovery:
    class _Info:
        def __init__(self, device: str, vid: int | None) -> None:
            self.device = device
            self.vid = vid

    def test_it_opens_only_photo_research_ports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[str] = []
        ports = [self._Info("COM4", 0x0403), self._Info("COM7", 0x180B)]
        monkeypatch.setattr(pr.serial.tools.list_ports, "comports", lambda: ports)

        def fake_init(self: PRSpectrometer, port: str, **_: object) -> None:
            opened.append(port)

        monkeypatch.setattr(PRSpectrometer, "__init__", fake_init)

        PRSpectrometer.discover()

        assert opened == ["COM7"]

    def test_no_instrument_is_a_serial_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            pr.serial.tools.list_ports,
            "comports",
            lambda: [self._Info("COM4", 0x0403)],
        )

        with pytest.raises(serial.SerialException, match="Power Saving"):
            PRSpectrometer.discover()
