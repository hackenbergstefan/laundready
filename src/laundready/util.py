import wave
from pathlib import Path

import numpy as np


def wave_read(path: str) -> np.ndarray:
    """Read a WAV file and return sampling frequency and normalized data."""
    with wave.open(path, "rb") as wf:
        sampling_frequency = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.uint8)

    data = data.astype(np.float32)
    # Normalize to -1.0 .. 1.0
    data = (data - 127) / 128
    assert np.max(data) <= 1.0 and np.min(data) >= -1.0  # noqa: PT018
    return sampling_frequency, data


def wave_write(path: str, data: np.typing.ArrayLike, sampling_frequency: int) -> None:
    """Write normalized data to a WAV file with given sampling frequency."""
    # Normalize to 0 .. 255
    assert np.max(data) <= 1.0 and np.min(data) >= -1.0  # noqa: PT018
    data = (data * 128 + 127).astype(np.uint8)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(sampling_frequency)
        wf.writeframes(data.tobytes())


def prepare_sample(
    input: str | Path,
    output: str | Path,
    minmax: tuple[int, int],
    threshold_start: float,
    threshold_end: float,
) -> np.array:
    """Prepare a sample WAV file by bandpass filtering and trimming silence."""
    from . import butter_bandpass  # noqa: PLC0415

    freq, data = wave_read(input)
    data = butter_bandpass(data, minmax, freq)
    first_over = np.argmax(np.abs(data) > threshold_start)
    last_over = -np.argmax(np.abs(data[::-1]) > threshold_end)
    wave_write(output, data[first_over:last_over], freq)
