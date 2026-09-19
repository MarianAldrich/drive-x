"""Send a low-rate fragmented JPEG webcam stream to the optional QNX receiver."""
import argparse
import math
import socket
import struct
import time
import zlib

import cv2

MAGIC = 0x44585646
VERSION = 1
TYPE_JPEG = 1
CHUNK_BYTES = 1200
MAX_FRAME_BYTES = 512 * 1024
HEADER = struct.Struct('!IBBHIQHHIHHHHI')
assert HEADER.size == 40


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, help='QNX Pi IPv4 address')
    parser.add_argument('--port', type=int, default=45556)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--fps', type=float, default=5.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--quality', type=int, default=65)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.fps <= 15:
        parser.error('Port must be valid and FPS must be from 1 to 15.')
    if not 30 <= args.quality <= 90 or args.width < 160 or args.height < 120:
        parser.error('Quality must be 30-90 and dimensions must be at least 160x120.')

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise RuntimeError(f'Could not open Windows camera {args.camera}. Close the web dashboard first.')
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    destination = (args.host, args.port)
    interval, next_frame, frame_id = 1.0 / args.fps, time.monotonic(), 0
    sent, skipped, report_at = 0, 0, time.monotonic()
    print(f'Streaming JPEG {args.width}x{args.height} at {args.fps:g} FPS to '
          f'{args.host}:{args.port}. Press Ctrl+C to stop.', flush=True)
    try:
        while True:
            delay = next_frame - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            next_frame = max(next_frame + interval, time.monotonic())
            ok, frame = camera.read()
            if not ok:
                skipped += 1
                continue
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                skipped += 1
                continue
            payload = encoded.tobytes()
            if len(payload) > MAX_FRAME_BYTES:
                skipped += 1
                continue
            frame_id = (frame_id + 1) & 0xffffffff
            capture_ns = time.monotonic_ns()
            checksum = zlib.crc32(payload) & 0xffffffff
            chunks = math.ceil(len(payload) / CHUNK_BYTES)
            for index in range(chunks):
                part = payload[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES]
                header = HEADER.pack(MAGIC, VERSION, TYPE_JPEG, HEADER.size, frame_id,
                                     capture_ns, index, chunks, len(payload), len(part),
                                     args.width, args.height, 0, checksum)
                sender.sendto(header + part, destination)
            sent += 1
            if args.preview:
                cv2.imshow('DRIVE-X UDP preview', frame)
                if cv2.waitKey(1) & 0xff in (27, ord('q')):
                    break
            now = time.monotonic()
            if now - report_at >= 1:
                print(f'frame={frame_id} jpeg_bytes={len(payload)} chunks={chunks} '
                      f'sent={sent} skipped={skipped}', flush=True)
                report_at = now
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        sender.close()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
