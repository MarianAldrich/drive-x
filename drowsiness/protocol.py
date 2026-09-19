"""Versioned VIGIL-QNX UDP protocol and nonblocking Windows publisher."""
from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import threading
import time
import zlib

MAGIC = 0x5649474C  # "VIGL"
VERSION = 1
TYPE_EVIDENCE = 1
TYPE_STATUS = 2

FLAG_MODEL_READY = 1 << 0
FLAG_FACE_VALID = 1 << 1
FLAG_HEAD_CUE = 1 << 2
FLAG_SENDER_RESET = 1 << 3
FLAG_EMERGENCY_ACTIVE = 1 << 4
FLAG_GPS_FIX_VALID = 1 << 5

STATUS_SENSOR_LOST = 0
STATUS_WARMING_UP = 1
STATUS_ALERT = 2
STATUS_LOW_VIGILANCE = 3
STATUS_CHECKING = 4
STATUS_DROWSY = 5
STATUS_NAMES = {
    STATUS_SENSOR_LOST: 'sensor_lost', STATUS_WARMING_UP: 'warming_up',
    STATUS_ALERT: 'alert', STATUS_LOW_VIGILANCE: 'low_vigilance',
    STATUS_CHECKING: 'checking', STATUS_DROWSY: 'drowsy',
}

_EVIDENCE_NO_CRC = struct.Struct('!IBBHIQffII')
_EVIDENCE = struct.Struct('!IBBHIQffIII')
_STATUS_NO_CRC = struct.Struct('!IBBHIQIIII')
_STATUS = struct.Struct('!IBBHIQIIIII')


def _crc(payload: bytes) -> int:
    return zlib.crc32(payload) & 0xFFFFFFFF


def pack_evidence(sequence: int, sender_ns: int, eye_closed: float, yawn: float,
                  inference_us: int, flags: int) -> bytes:
    values = (MAGIC, VERSION, TYPE_EVIDENCE, _EVIDENCE.size, sequence & 0xFFFFFFFF,
              sender_ns & 0xFFFFFFFFFFFFFFFF, float(eye_closed), float(yawn),
              inference_us & 0xFFFFFFFF, flags & 0xFFFFFFFF)
    body = _EVIDENCE_NO_CRC.pack(*values)
    return body + struct.pack('!I', _crc(body))


def unpack_evidence(payload: bytes) -> dict:
    if len(payload) != _EVIDENCE.size:
        raise ValueError(f'evidence packet must be {_EVIDENCE.size} bytes')
    values = _EVIDENCE.unpack(payload)
    if values[:4] != (MAGIC, VERSION, TYPE_EVIDENCE, _EVIDENCE.size):
        raise ValueError('invalid evidence header')
    if _crc(payload[:-4]) != values[-1]:
        raise ValueError('invalid evidence CRC')
    keys = ('magic', 'version', 'type', 'length', 'sequence', 'sender_ns',
            'eye_closed', 'yawn', 'inference_us', 'flags', 'crc32')
    return dict(zip(keys, values))


def unpack_status(payload: bytes) -> dict:
    if len(payload) != _STATUS.size:
        raise ValueError(f'status packet must be {_STATUS.size} bytes')
    values = _STATUS.unpack(payload)
    if values[:4] != (MAGIC, VERSION, TYPE_STATUS, _STATUS.size):
        raise ValueError('invalid status header')
    if _crc(payload[:-4]) != values[-1]:
        raise ValueError('invalid status CRC')
    keys = ('magic', 'version', 'type', 'length', 'sequence', 'qnx_receive_ns',
            'decision_us', 'state', 'flags', 'rto_us', 'crc32')
    result = dict(zip(keys, values))
    result['state_name'] = STATUS_NAMES.get(result['state'], 'unknown')
    return result


def pack_status(sequence: int, receive_ns: int, decision_us: int, state: int,
                flags: int = 0, rto_us: int = 0) -> bytes:
    values = (MAGIC, VERSION, TYPE_STATUS, _STATUS.size, sequence & 0xFFFFFFFF,
              receive_ns & 0xFFFFFFFFFFFFFFFF, decision_us & 0xFFFFFFFF,
              state & 0xFFFFFFFF, flags & 0xFFFFFFFF, rto_us & 0xFFFFFFFF)
    body = _STATUS_NO_CRC.pack(*values)
    return body + struct.pack('!I', _crc(body))


@dataclass
class PublisherStatus:
    enabled: bool
    connected: bool = False
    host: str | None = None
    port: int | None = None
    last_sequence: int | None = None
    last_reply_monotonic: float | None = None
    round_trip_ms: float | None = None
    decision_us: int | None = None
    decision_state: str = 'unavailable'
    recovery_rto_ms: float | None = None
    status_flags: int = 0
    emergency_active: bool = False
    simulated_speed_kph: float = 60.0
    error: str | None = None


class UdpEvidencePublisher:
    """Send evidence without delaying HTTP inference and receive QNX status replies."""

    def __init__(self, host: str | None, port: int = 45555, enabled: bool = True):
        self.enabled = bool(enabled and host)
        self.host, self.port = host, port
        self.socket = None
        self.sequence = 0
        self.sent_at: dict[int, float] = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.latest = PublisherStatus(enabled=self.enabled, host=host, port=port)
        if self.enabled:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.connect((host, port))
            self.socket.settimeout(.25)
            self.thread = threading.Thread(target=self._receive, name='qnx-status', daemon=True)
            self.thread.start()

    def publish(self, output: dict) -> int | None:
        if not self.enabled or not output.get('sampled_this_request'):
            return None
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        predictions = output.get('predictions', {})
        eye = float(output.get('instant_eye_closed', 0.0))
        yawn = float(predictions.get('yawn', {}).get('yawn', 0.0))
        # Suppress isolated weak-label spikes; QNX receives confirmed yawn evidence.
        if not output.get('yawn_detected'):
            yawn = 0.0
        flags = 0
        if all(item.get('loaded') for item in output.get('models', {}).values()):
            flags |= FLAG_MODEL_READY
        if output.get('face_count') == 1:
            flags |= FLAG_FACE_VALID
        if output.get('head', {}).get('cue'):
            flags |= FLAG_HEAD_CUE
        if self.sequence == 1:
            flags |= FLAG_SENDER_RESET
        packet = pack_evidence(self.sequence, time.monotonic_ns(), eye, yawn,
                               round(float(output.get('inference_ms', 0)) * 1000), flags)
        try:
            self.socket.send(packet)
            with self.lock:
                self.sent_at[self.sequence] = time.monotonic()
                self.sent_at = dict(list(self.sent_at.items())[-32:])
                self.latest.error = None
            return self.sequence
        except OSError as exc:
            with self.lock:
                self.latest.error = str(exc)
            return None

    def _receive(self):
        while not self.stop_event.is_set():
            try:
                data = self.socket.recv(256)
                status = unpack_status(data)
                now = time.monotonic()
                with self.lock:
                    sent = self.sent_at.pop(status['sequence'], None)
                    prior_reply = self.latest.last_reply_monotonic
                    elapsed = max(.1, min(2.5, now - prior_reply)) if prior_reply else .5
                    emergency = bool(status['flags'] & FLAG_EMERGENCY_ACTIVE)
                    targets = {
                        'sensor_lost': 0.0, 'warming_up': 40.0, 'alert': 60.0,
                        'low_vigilance': 40.0, 'checking': 30.0, 'drowsy': 20.0,
                    }
                    target = 0.0 if emergency else targets.get(status['state_name'], 0.0)
                    current = self.latest.simulated_speed_kph
                    rate = 12.0 if target < current else 6.0
                    change = rate * elapsed
                    current = max(target, current - change) if target < current else min(target, current + change)
                    self.latest.connected = True
                    self.latest.last_sequence = status['sequence']
                    self.latest.last_reply_monotonic = now
                    self.latest.round_trip_ms = round((now - sent) * 1000, 1) if sent else None
                    self.latest.decision_us = status['decision_us']
                    self.latest.decision_state = status['state_name']
                    self.latest.recovery_rto_ms = round(status['rto_us'] / 1000, 1) if status['rto_us'] else None
                    self.latest.status_flags = status['flags']
                    self.latest.emergency_active = emergency
                    self.latest.simulated_speed_kph = round(current, 1)
                    self.latest.error = None
            except socket.timeout:
                continue
            except (OSError, ValueError) as exc:
                with self.lock:
                    self.latest.error = str(exc)

    def status(self) -> dict:
        with self.lock:
            value = dict(vars(self.latest))
            last = self.latest.last_reply_monotonic
        value['packet_age_ms'] = round((time.monotonic() - last) * 1000, 1) if last else None
        value['connected'] = bool(value['connected'] and value['packet_age_ms'] is not None
                                  and value['packet_age_ms'] < 2500)
        return value

    def close(self):
        self.stop_event.set()
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join(timeout=1)
