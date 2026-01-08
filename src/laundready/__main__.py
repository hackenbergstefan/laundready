import logging
from configparser import ConfigParser

import paho.mqtt.client as mqtt

from . import MqttLaundready


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

    mqtt_host = conf["laundready"]["mqtt_host"]
    mqtt_user = conf["laundready"]["mqtt_user"]
    mqtt_password = conf["laundready"]["mqtt_password"]
    mqtt_topic = conf["laundready"].get("mqtt_topic", "laundready/detected")

    mqtt_client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="laundready",
        clean_session=False,
    )
    mqtt_client.username_pw_set(mqtt_user, mqtt_password)
    mqtt_client.connect(mqtt_host)
    mqtt_client.loop_start()

    logging.basicConfig(level=logging.DEBUG)

    laundready = MqttLaundready(
        sample=sample_path,
        sampling_frequency=sampling_frequency,
        mqtt=mqtt_client,
        mqtt_topic=mqtt_topic,
        bandpass_minmax=bandpass_minmax,
        bandpass_threshold=bandpass_threshold,
        conv_threshold=conv_threshold,
    )
    laundready.loop_forever()


if __name__ == "__main__":
    main()
