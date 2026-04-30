"""
Anonymized MQTT publisher for clinical events.

This module is the ONLY component allowed to send data over the network.
By design, the payload contains:
  - session_id   : opaque random UUID, regenerated at each session start
  - timestamp    : ISO 8601 UTC
  - event_type   : high-level label (FALL_DETECTED, POSTURE_WARNING, ...)
  - severity     : info / warning / alert

NO image, NO keypoint coordinate, NO patient identifier ever leaves
this module. This is enforced by accepting only `Event` objects whose
metadata is dropped before publishing.
"""

import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

import paho.mqtt.client as mqtt

from event_detector import Event


logger = logging.getLogger(__name__)


class AnonymizedEventPublisher:
    """
    Thin wrapper around paho-mqtt that enforces the anonymization contract.

    Usage:
        pub = AnonymizedEventPublisher()
        pub.connect()
        pub.publish_event(event)
        pub.disconnect()
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic: str = "clinical/events",
        qos: int = 1,
        client_id_prefix: str = "jetson-edge",
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.qos = qos

        # Anonymous session id. Reset only when the publisher is recreated.
        self.session_id = uuid.uuid4().hex[:8]
        self.session_started = datetime.now(timezone.utc).isoformat()

        # MQTT client
        client_id = f"{client_id_prefix}-{self.session_id}"
        self._client = mqtt.Client(client_id=client_id, clean_session=True)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish

        self._connected = False
        self._published_count = 0

    # -- lifecycle ----------------------------------------------------------

    def connect(self, timeout_sec: float = 5.0) -> bool:
        """
        Connect to the broker and start the network loop in a background thread.
        Returns True if connection succeeded within timeout.
        """
        try:
            self._client.connect(self.broker_host, self.broker_port,
                                 keepalive=30)
            self._client.loop_start()
        except Exception as e:
            logger.error("MQTT connect failed: %s", e)
            return False

        # Wait briefly for on_connect callback
        deadline = time.time() + timeout_sec
        while time.time() < deadline and not self._connected:
            time.sleep(0.05)

        if self._connected:
            logger.info("MQTT connected to %s:%d (session=%s)",
                        self.broker_host, self.broker_port, self.session_id)
        else:
            logger.warning("MQTT connect timeout")
        return self._connected

    def disconnect(self):
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass

    # -- publish ------------------------------------------------------------

    def publish_event(self, event: Event) -> bool:
        """
        Publish an event after anonymization.
        Returns True if the message was queued for sending.

        IMPORTANT: event.metadata is intentionally NOT included in the
        published payload. It may contain coordinates or other indirect
        identifiers and must stay on-device.
        """
        if not self._connected:
            logger.warning("publish_event called while not connected")
            return False

        payload = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec='seconds'),
            "event_type": event.type.value,
            "severity": event.severity.value,
        }
        body = json.dumps(payload)

        info = self._client.publish(self.topic, body, qos=self.qos)
        # MQTT_ERR_SUCCESS = 0
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error("MQTT publish failed (rc=%d)", info.rc)
            return False
        return True

    # -- callbacks ----------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
        else:
            logger.error("MQTT connect failed with rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning("MQTT unexpected disconnect (rc=%d)", rc)

    def _on_publish(self, client, userdata, mid):
        self._published_count += 1


# ---------------------------------------------------------------------------
# Standalone smoke test:
#   python3 -m mqtt_publisher
# Publishes a fake event every 2 seconds. Use mosquitto_sub to verify.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    from event_detector import EventType, Severity

    pub = AnonymizedEventPublisher()
    if not pub.connect():
        raise SystemExit("Could not connect to MQTT broker")

    print(f"Publishing test events to topic '{pub.topic}'...")
    print(f"Subscribe with: mosquitto_sub -h localhost -t '{pub.topic}' -v")
    print("Press Ctrl+C to stop.")

    try:
        i = 0
        types = [EventType.FALL_DETECTED, EventType.POSTURE_WARNING,
                 EventType.PROLONGED_STILLNESS]
        sev = [Severity.ALERT, Severity.WARNING, Severity.WARNING]
        while True:
            evt = Event(
                type=types[i % len(types)],
                severity=sev[i % len(sev)],
                timestamp=time.time(),
                metadata={"test": True},
            )
            ok = pub.publish_event(evt)
            print(f"[{i}] published {evt.type.value} -> ok={ok}")
            i += 1
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pub.disconnect()