import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
import pyaudio
import scipy.signal

from .util import wave_read, wave_write

VISUAL_DEBUG = False

if VISUAL_DEBUG:
    import matplotlib.pyplot as plt

_log = logging.getLogger(__name__)


def butter_bandpass(
    data: np.typing.ArrayLike,
    minmax: tuple[int, int],
    sampling_frequency: int,
    order: int = 5,
) -> np.ndarray:
    """
    Bandpass filter the data between minmax frequencies.

    Args:
        data: Input signal data.
        minmax: Tuple of (lowcut, highcut) frequencies.
        sampling_frequency: Sampling frequency of the data.
        order: Order of the Butterworth filter.

    Returns:
        Filtered signal as a numpy array normaled to max amplitude of 1.

    """
    nyq = 0.5 * sampling_frequency
    minmax = np.array(minmax) / nyq
    b, a = scipy.signal.butter(order, minmax, btype="band", analog=False)
    y = scipy.signal.lfilter(b, a, data)
    y /= np.max(np.abs(y))
    return y


def bandpass_trigger(
    data: np.typing.ArrayLike,
    minmax: tuple[int, int],
    sampling_frequency: int,
    threshold: float,
    distance: int = 0,
) -> list[int]:
    """
    Apply bandpass filter and detect trigger points where signal exceeds threshold.

    Args:
        data: Input signal data.
        minmax: Tuple of (lowcut, highcut) frequencies for bandpass filter.
        sampling_frequency: Sampling frequency of the data.
        threshold: Amplitude threshold for trigger detection.
        distance: Minimum distance between triggers.

    Returns:
        List of indices where triggers occur.

    """
    data = butter_bandpass(data, minmax, sampling_frequency)
    if VISUAL_DEBUG:
        plt.plot(data)
        plt.show()
    starts = np.argwhere(np.abs(data) > threshold).flatten()
    diffed = [int(starts[0])]
    for s in starts[1:]:
        if s - diffed[-1] > distance:
            diffed.append(int(s))
    return diffed


def normalized_fftconvolve(
    data: np.typing.ArrayLike,
    sample: np.typing.ArrayLike,
) -> np.ndarray:
    """Perform FFT convolution and normalize output to -1.0 .. 1.0."""
    conv = scipy.signal.fftconvolve(
        data,
        sample,
        mode="valid",
    )
    conv /= sample.var() * len(sample)
    return conv


def search_sample(  # noqa: PLR0913
    data: np.typing.ArrayLike,
    sample: np.typing.ArrayLike,
    sampling_frequency: int,
    bandpass_minmax: tuple[int, int],
    bandpass_threshold: float = 0.1,
    conv_threshold: float = 0.003,
) -> list[int]:
    """Search for occurrences of sample in data."""
    starts = bandpass_trigger(
        data,
        minmax=bandpass_minmax,
        sampling_frequency=sampling_frequency,
        threshold=bandpass_threshold,
        distance=len(sample),
    )
    occurrences = []
    for s in starts:
        conv = normalized_fftconvolve(
            data[s : s + 2 * len(sample)],
            sample,
        )
        _log.debug(f"search_sample: Search at {s} -> {conv.max()}")
        if conv.max() > conv_threshold:
            _log.info(
                f"search_sample: Found sample at {s + conv.argmax()} with {conv.max()}"
            )
            occurrences.append(int(s + conv.argmax()))
    return occurrences


class Laundready:
    def __init__(
        self,
        sample: str | Path,
        sampling_frequency: int,
        bandpass_minmax: tuple[int, int] = (2350, 2400),
        bandpass_threshold: float = 0.1,
        conv_threshold: float = 0.003,
    ) -> None:
        freq, self.sample = wave_read(sample)
        assert freq == sampling_frequency, (
            f"Sample framerate {freq} does not match expected {sampling_frequency}"
        )

        self.sampling_frequency = sampling_frequency
        self.bandpass_minmax = bandpass_minmax
        self.bandpass_threshold = bandpass_threshold
        self.conv_threshold = conv_threshold

    @contextmanager
    def record(self) -> Generator[pyaudio.Stream, None, None]:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paUInt8,
            channels=1,
            rate=self.sampling_frequency,
            input=True,
            frames_per_buffer=len(self.sample),
        )
        yield stream
        stream.close()
        audio.terminate()

    def loop_forever(self) -> None:
        last = None
        with self.record() as stream:
            while True:
                if last is None:
                    last = np.frombuffer(stream.read(len(self.sample)), dtype=np.uint8)
                    last = last.astype(np.float32)
                    last = (last - 127) / 128
                cur = np.frombuffer(stream.read(len(self.sample)), dtype=np.uint8)
                cur = cur.astype(np.float32)
                cur = (cur - 127) / 128
                data = np.concatenate((last, cur))
                if search_sample(
                    data,
                    self.sample,
                    self.sampling_frequency,
                    self.bandpass_minmax,
                    self.bandpass_threshold,
                    self.conv_threshold,
                ):
                    self.detected()

                wave_write("debug.wav", data, self.sampling_frequency)
                last = cur

    def detected(self) -> None:
        _log.info("Laundready: Detected sample!")


class MqttLaundready(Laundready):
    def __init__(  # noqa: PLR0913
        self,
        sample: str | Path,
        sampling_frequency: int,
        mqtt: mqtt.Client,
        mqtt_topic: str = "laundready/detected",
        bandpass_minmax: tuple[int, int] = (2350, 2400),
        bandpass_threshold: float = 0.1,
        conv_threshold: float = 0.003,
    ) -> None:
        super().__init__(
            sample,
            sampling_frequency,
            bandpass_minmax,
            bandpass_threshold,
            conv_threshold,
        )
        self.mqtt = mqtt
        self.mqtt_topic = mqtt_topic

    def detected(self) -> None:
        super().detected()
        self.mqtt.publish(
            self.mqtt_topic,
            payload=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
        )
