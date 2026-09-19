import socket
import threading
import time
import unittest

from drowsiness.protocol import (
    FLAG_EMERGENCY_ACTIVE, FLAG_FACE_VALID, FLAG_MODEL_READY, STATUS_ALERT,
    STATUS_DROWSY, UdpEvidencePublisher,
    pack_evidence, pack_status, unpack_evidence, unpack_status,
)


class ProtocolTests(unittest.TestCase):
    def test_evidence_round_trip_and_crc(self):
        packet = pack_evidence(7, 123456789, .75, .25, 4400,
                               FLAG_MODEL_READY | FLAG_FACE_VALID)
        self.assertEqual(len(packet), 40)
        result = unpack_evidence(packet)
        self.assertEqual(result['sequence'], 7)
        self.assertAlmostEqual(result['eye_closed'], .75)
        damaged = bytearray(packet)
        damaged[20] ^= 1
        with self.assertRaisesRegex(ValueError, 'CRC'):
            unpack_evidence(bytes(damaged))

    def test_status_round_trip(self):
        result = unpack_status(pack_status(9, 1000, 52, STATUS_ALERT, rto_us=1234))
        self.assertEqual(result['state_name'], 'alert')
        self.assertEqual(result['decision_us'], 52)
        self.assertEqual(result['rto_us'], 1234)

    def test_emergency_status_flag_is_preserved(self):
        result = unpack_status(pack_status(10, 2000, 60, STATUS_DROWSY,
                                           flags=FLAG_EMERGENCY_ACTIVE))
        self.assertEqual(result['flags'] & FLAG_EMERGENCY_ACTIVE,
                         FLAG_EMERGENCY_ACTIVE)

    def test_publisher_only_sends_new_samples_and_receives_reply(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(('127.0.0.1', 0))
        port = receiver.getsockname()[1]

        def qnx_mock():
            packet, address = receiver.recvfrom(256)
            evidence = unpack_evidence(packet)
            receiver.sendto(pack_status(evidence['sequence'], 5, 77, STATUS_ALERT), address)

        thread = threading.Thread(target=qnx_mock)
        thread.start()
        publisher = UdpEvidencePublisher('127.0.0.1', port)
        output = dict(sampled_this_request=False)
        self.assertIsNone(publisher.publish(output))
        output = dict(sampled_this_request=True, instant_eye_closed=.2, inference_ms=3.5,
                      face_count=1, head={'cue': False}, predictions={'yawn': {'yawn': .1}},
                      models={'eye_state': {'loaded': True}, 'yawn': {'loaded': True}})
        self.assertEqual(publisher.publish(output), 1)
        thread.join(timeout=2)
        for _ in range(20):
            if publisher.status()['connected']:
                break
            time.sleep(.02)
        status = publisher.status()
        self.assertTrue(status['connected'])
        self.assertEqual(status['decision_us'], 77)
        publisher.close()
        receiver.close()


if __name__ == '__main__':
    unittest.main()
