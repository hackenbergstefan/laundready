# Laundready - Detect washing machine beeps to notify when laundry is done.

Laundready is a Python application that listens for the beeps of a washing machine to notify you when your laundry is done.
It continously monitors audio input, detects the prerecorded pattern, and sends an mqtt notification.


## Installation and Usage

1. Install Laundready via pip:

```bash
pip install git+https://github.com/stefanhackenberg/laundready.git
```

2. Record a sample WAV file of your washing machine's beep sound and save it as `sample.wav`.

  Note that you can optimize the sample by bandpass filtering and trimming silence using the provided function `laundready.util.prepare_sample`.

3. Create configuration file `laundready.ini` based on `laundready_example.ini`:

```ini
[laundready]
sample_path = sample.wav
sampling_frequency = 8000
bandpass_min = 2350
bandpass_max = 2400
bandpass_threshold = 0.1
conv_threshold = 0.003

mqtt_broker = localhost
mqtt_port = 1883
mqtt_topic = laundready/done
```

4. Run Laundready:

```bash
laundready
```