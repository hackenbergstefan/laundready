from configparser import ConfigParser

from . import Laundready


def main() -> None:
    conf = ConfigParser()
    conf.read("laundready.ini")
    sample_path = conf["laundready"]["sample_path"]
    sampling_frequency = conf.getint("laundready", "sampling_frequency", fallback=8000)
    bandpass_minmax = (
        conf.getfloat("laundready", "bandpass_min"),
        conf.getfloat("laundready", "bandpass_max"),
    )
    bandpass_threshold = conf.getfloat("laundready", "bandpass_threshold", fallback=0.1)
    conv_threshold = conf.getfloat("laundready", "conv_threshold", fallback=0.003)

    laundready = Laundready(
        sample=sample_path,
        sampling_frequency=sampling_frequency,
        bandpass_minmax=bandpass_minmax,
        bandpass_threshold=bandpass_threshold,
        conv_threshold=conv_threshold,
    )
    laundready.loop_forever()


if __name__ == "__main__":
    main()
