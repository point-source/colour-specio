"""
Photo Research spectroradiometer implementation (PR-655, PR-670).

The PR-6xx family speaks a plain-text "remote mode" over a USB CDC serial
port. The protocol below follows the PR-655/670 SpectraScan User Manual,
"Remote Mode Operation":

- ``PHOTO``, sent as single characters with no terminator, enters remote
  mode; the device answers ``REMOTE MODE``. ``Q`` leaves it.
- Every other command is sent as single characters terminated by a
  carriage return. A reply leads with a signed status code (``00000`` is
  OK) and is terminated by CR LF.
- ``M5`` measures and returns ``status,units,peak,integrated,photon``
  followed by one ``nm,value`` line per sample: spectral radiance, 380-780
  nm in 4 nm steps on a PR-655.
- ``D13`` returns the last exposure time; ``D110`` the serial number;
  ``D111`` the model; ``D114`` the software version.
- ``SU1`` selects SI units; ``SS1`` and ``SS0`` turn source-synchronised
  integration on and off.

Measurement status codes follow the manual's "Remote Control Measurement
Errors": ``-8`` weak light and ``-2`` overload are results (the light is
outside what the device can report), the sync codes mean the device could
not lock to the source, and anything else is a failed exchange.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, Self, final

import serial
import serial.tools.list_ports
from colour import SpectralDistribution, SpectralShape

from specio.common import RawSPDMeasurement, SpecRadiometer
from specio.common.exceptions import DeviceError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

__all__ = [
    "PHOTO_RESEARCH_USB_VID",
    "MeasurementOutOfRange",
    "PRCommandError",
    "PRReply",
    "PRSpectrometer",
    "PRStatus",
    "SerialPort",
]

log = logging.getLogger(__name__)

PHOTO_RESEARCH_USB_VID = 0x180B
"""USB vendor id the PR-655 enumerates with (observed: ``180B:0524``)."""

_PR_SERIAL_KWARGS: Mapping = MappingProxyType(
    {
        "baudrate": 9600,
        "bytesize": 8,
        "parity": "N",
        "stopbits": 1,
        "rtscts": False,
        "timeout": 0.1,
        # Without it a sleeping instrument blocks a write forever.
        "write_timeout": 2.0,
    }
)

_CHARACTER_GAP = 0.05
"""Seconds between written characters; the device drops faster input."""

_QUIET_GAP = 0.5
"""Seconds of silence, after a reply has started, that end the reply."""

_COMMAND_TIMEOUT = 5.0
"""Seconds to wait for the first byte of a non-measuring command's reply."""

_DEFAULT_MEASURE_TIMEOUT = 60.0
"""Seconds to wait for a measurement: adaptive exposure tops out at 6 s per
cycle on a PR-655, plus sync, dark current and the spectrum's transfer."""

_UNKNOWN_EXPOSURE = -1.0
"""Exposure when ``D13`` gives nothing readable; the CR driver's sentinel."""

_NUMBER = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"
_SPECTRAL_LINE = re.compile(rf"^\s*({_NUMBER})\s*,\s*({_NUMBER})\s*$")
_EXPOSURE = re.compile(r"(\d+(?:\.\d+)?)\s*msec", re.IGNORECASE)
_VERSION = re.compile(r"^\d+\.\d+\w*$")


class SerialPort(Protocol):
    """The slice of :class:`serial.Serial` this driver uses."""

    def write(self, data: bytes, /) -> int | None:
        """Write bytes to the port."""
        ...

    def read(self, size: int = 1, /) -> bytes:
        """Read up to `size` bytes, returning early on the port timeout."""
        ...

    def reset_input_buffer(self) -> None:
        """Discard anything already received."""
        ...

    def close(self) -> None:
        """Close the port."""
        ...


class PRStatus:
    """Status codes from the manual's "Remote Control Error Codes"."""

    OK = 0
    SOURCE_NOT_CONSTANT = -1
    LIGHT_OVERLOAD = -2
    CANNOT_SYNC = -3
    ADAPTIVE_MODE_ERROR = -4
    WEAK_LIGHT = -8
    SYNC_ERROR = -9
    CANNOT_AUTO_SYNC = -10
    ADAPTIVE_TIMEOUT = -12
    ILLEGAL_COMMAND = -1000

    OUT_OF_RANGE = frozenset({WEAK_LIGHT, LIGHT_OVERLOAD})
    SYNC_FAILURES = frozenset({CANNOT_SYNC, SYNC_ERROR, CANNOT_AUTO_SYNC})


@dataclass(frozen=True)
class PRReply:
    """A parsed remote-mode reply: the status code and the remaining lines."""

    command: str
    code: int
    fields: tuple[str, ...]
    lines: tuple[str, ...]

    @property
    def last_field(self) -> str:
        """The final comma-separated field of the status line."""
        return self.fields[-1] if self.fields else ""


class PRCommandError(DeviceError):
    """The device answered a command with a non-OK status code."""

    def __init__(self, reply: PRReply, *args: object) -> None:
        """
        Initialize from the reply that carried the error.

        Parameters
        ----------
        reply : PRReply
            The parsed reply, including the device's status code.
        *args : object
            Passed through to :class:`Exception`.
        """
        self.reply = reply
        super().__init__(*args)

    @property
    def code(self) -> int:
        """The device's own status code."""
        return self.reply.code


class MeasurementOutOfRange(PRCommandError):
    """The device measured, and the light is outside what it can report.

    A result rather than a failed exchange: weak light routes the patch to
    a more sensitive instrument, and overload means an aperture or filter
    problem. Neither is a retry. Named to match the Colorimetry Research
    driver's type of the same name, which callers match on by name.
    """


@final
class PRSpectrometer(SpecRadiometer):
    """Interface with a Photo Research PR-655 or PR-670 in remote mode.

    Implements :class:`specio.spectrometers.SpecRadiometer`. Measures with
    the device's adaptive exposure, synchronised to the source when it can
    lock to one, and leaves remote mode on :meth:`close`.

    Raises
    ------
    serial.SerialException
        If `PRSpectrometer.discover` finds no instrument, or the port fails.
    DeviceError
        If the device does not enter remote mode or sends an unreadable reply.
    PRCommandError
        If the device answers a command with an error status.
    MeasurementOutOfRange
        If the light is outside the device's reportable range.
    """

    @classmethod
    def discover(cls, **kwargs: object) -> PRSpectrometer:
        """Find a Photo Research instrument by USB vendor id and connect.

        Only ports with Photo Research's vendor id are opened, so discovery
        writes nothing to unrelated serial devices.

        Parameters
        ----------
        **kwargs : object
            Passed to :class:`PRSpectrometer`.

        Returns
        -------
        PRSpectrometer
            A connected spectrometer in remote mode.

        Raises
        ------
        serial.SerialException
            If no Photo Research instrument answers.
        """
        candidates = [
            p.device
            for p in serial.tools.list_ports.comports()
            if p.vid == PHOTO_RESEARCH_USB_VID
        ]
        failures: list[str] = []
        for device in candidates:
            try:
                return cls(device, **kwargs)  # pyright: ignore[reportArgumentType]
            except (serial.SerialException, DeviceError) as e:
                failures.append(f"{device}: {e}")
        detail = "; ".join(failures) if failures else "no port with USB VID 180B"
        raise serial.SerialException(
            f"Could not connect to a Photo Research spectroradiometer ({detail}). "
            "Check the USB connection, that the instrument is on, and that its "
            "Power Saving is off."
        )

    def __init__(
        self,
        port: str | SerialPort,
        *,
        sync: bool = True,
        measure_timeout: float = _DEFAULT_MEASURE_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Open the port, enter remote mode, and set SI units and sync.

        Parameters
        ----------
        port : str | SerialPort
            A serial device path (``COM7``, ``/dev/ttyACM0``) or an open port.
        sync : bool, optional
            Synchronise integration to the source's refresh frequency (the
            device's Auto Sync). A measurement that cannot lock falls back
            to unsynchronised. Default True.
        measure_timeout : float, optional
            Seconds to wait for a measurement before giving up.
        sleep : Callable[[float], None], optional
            Sleep function; injectable for tests.
        clock : Callable[[], float], optional
            Monotonic clock; injectable for tests.
        """
        self._sleep = sleep
        self._clock = clock
        self.measure_timeout = measure_timeout
        self._port: SerialPort | None = (
            serial.Serial(port, **_PR_SERIAL_KWARGS) if isinstance(port, str) else port
        )
        self.synchronised = False
        self._serial_number: str | None = None
        try:
            self._enter_remote_mode()
            self._command("SU1")
            self._set_sync(sync)
        except BaseException:
            self._port.close()
            self._port = None
            raise

    def __enter__(self) -> Self:
        """Return self; the port is already open."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Leave remote mode and close the port."""
        self.close()

    def close(self) -> None:
        """Leave remote mode and close the port. Safe to call twice."""
        port = self._port
        if port is None:
            return
        try:
            self._write("Q", terminate=False)
        except (serial.SerialException, OSError):
            log.debug("could not send Q while closing", exc_info=True)
        finally:
            port.close()
            self._port = None

    @property
    def manufacturer(self) -> str:
        """The device manufacturer name."""
        return "Photo Research"

    @cached_property
    def model(self) -> str:
        """The model reported by the device, e.g. ``PR655`` (no hyphen)."""
        return self._command("D111").last_field

    @property
    def serial_number(self) -> str:
        """The serial number reported by the device, read once."""
        if self._serial_number is None:
            self._serial_number = self._command("D110").last_field
        return self._serial_number

    @cached_property
    def firmware(self) -> str | None:
        """The device's software version, or None if it gives none readable."""
        try:
            version = self._command("D114").last_field
        except DeviceError:
            return None
        return version if _VERSION.match(version) else None

    def _raw_measure(self) -> RawSPDMeasurement:
        """
        Trigger one measurement and return its spectral radiance.

        Returns
        -------
        RawSPDMeasurement
            Spectral radiance in W/(sr·m²·nm) at the device's native sampling,
            with the exposure the device chose, in seconds.

        Raises
        ------
        MeasurementOutOfRange
            If the light is outside what the device can report.
        PRCommandError
            For any other error status.
        """
        try:
            reply = self._command("M5", timeout=self.measure_timeout)
        except PRCommandError as e:
            if e.code not in PRStatus.SYNC_FAILURES or not self.synchronised:
                raise
            log.warning(
                "%s could not sync to the source (status %d); "
                "measuring unsynchronised from here on",
                self.readable_id,
                e.code,
            )
            self._set_sync(False)
            reply = self._command("M5", timeout=self.measure_timeout)

        return RawSPDMeasurement(
            spd=_parse_spectrum(reply),
            spectrometer_id=self.readable_id,
            exposure=self._last_exposure(),
            anc_data={"status_line": ",".join((f"{reply.code}", *reply.fields))},
        )

    def _last_exposure(self) -> float:
        """The last measurement's exposure in seconds, from ``D13``."""
        try:
            reply = self._command("D13")
        except DeviceError:
            return _UNKNOWN_EXPOSURE
        match = _EXPOSURE.search(",".join(reply.fields))
        return float(match[1]) / 1000 if match else _UNKNOWN_EXPOSURE

    def _enter_remote_mode(self) -> None:
        """Send ``PHOTO`` and require the remote-mode banner.

        A device already in remote mode does not repeat the banner, so on
        silence fall back to asking for its model.
        """
        assert self._port is not None
        self._port.reset_input_buffer()
        self._write("PHOTO", terminate=False)
        banner = self._read_reply(_COMMAND_TIMEOUT)
        if b"REMOTE MODE" in banner.upper():
            return
        try:
            self._command("D111")
        except DeviceError as e:
            raise DeviceError(
                f"device did not enter remote mode (answered {banner!r})"
            ) from e

    def _set_sync(self, sync: bool) -> None:
        """Select Auto Sync or No Sync for the measurements that follow."""
        self._command("SS1" if sync else "SS0")
        self.synchronised = sync

    def _command(self, command: str, timeout: float = _COMMAND_TIMEOUT) -> PRReply:
        """Send `command` and return its reply, raising on an error status."""
        assert self._port is not None, "the port is closed"
        self._port.reset_input_buffer()
        self._write(command)
        reply = _parse_reply(command, self._read_reply(timeout))
        if reply.code in PRStatus.OUT_OF_RANGE:
            what = "weak light" if reply.code == PRStatus.WEAK_LIGHT else "overload"
            raise MeasurementOutOfRange(
                reply, f"{command}: {what}, outside the measurable range"
            )
        if reply.code != PRStatus.OK:
            raise PRCommandError(reply, f"{command}: device status {reply.code}")
        return reply

    def _write(self, text: str, *, terminate: bool = True) -> None:
        """Write one character at a time; the device drops faster input."""
        assert self._port is not None, "the port is closed"
        data = text.encode("ascii") + (b"\r" if terminate else b"")
        for byte in data:
            self._port.write(bytes((byte,)))
            self._sleep(_CHARACTER_GAP)

    def _read_reply(self, timeout: float) -> bytes:
        """Read until the reply has started and then gone quiet.

        Returns whatever arrived, possibly nothing, once `timeout` passes
        with no first byte.
        """
        assert self._port is not None, "the port is closed"
        start = self._clock()
        last: float | None = None
        buffer = bytearray()
        while True:
            chunk = self._port.read(4096)
            now = self._clock()
            if chunk:
                buffer += chunk
                last = now
            elif (last is not None and now - last >= _QUIET_GAP) or (
                last is None and now - start >= timeout
            ):
                return bytes(buffer)


def _parse_reply(command: str, raw: bytes) -> PRReply:
    """Split a reply into its status line and the lines after it."""
    lines = [
        line.strip()
        for line in raw.decode("ascii", errors="replace").splitlines()
        if line.strip()
    ]
    if not lines:
        raise DeviceError(f"{command}: no reply from device")
    status, *rest = lines
    head, *fields = status.split(",")
    try:
        code = int(head)
    except ValueError as e:
        raise DeviceError(f"{command}: unreadable status line {status!r}") from e
    return PRReply(
        command=command,
        code=code,
        fields=tuple(f.strip() for f in fields),
        lines=tuple(rest),
    )


def _parse_spectrum(reply: PRReply) -> SpectralDistribution:
    """Read the ``nm,value`` lines of an ``M5`` reply into a distribution."""
    samples: list[tuple[float, float]] = []
    for line in reply.lines:
        match = _SPECTRAL_LINE.match(line)
        if match is None:
            raise DeviceError(f"M5: unexpected line in spectrum {line!r}")
        samples.append((float(match[1]), float(match[2])))
    if len(samples) < 2:
        raise DeviceError(f"M5: spectrum has {len(samples)} samples")

    wavelengths = [w for w, _ in samples]
    interval = wavelengths[1] - wavelengths[0]
    expected = [wavelengths[0] + i * interval for i in range(len(wavelengths))]
    if interval <= 0 or any(
        abs(w - e) > 1e-6 for w, e in zip(wavelengths, expected, strict=True)
    ):
        raise DeviceError("M5: wavelengths are not evenly spaced")

    shape = SpectralShape(wavelengths[0], wavelengths[-1], interval)
    return SpectralDistribution(data=[v for _, v in samples], domain=shape)
