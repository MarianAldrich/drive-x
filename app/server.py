"""Loopback-only camera interface. Images are processed in memory, never recorded."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drowsiness.runtime import InferenceEngine
from drowsiness.protocol import UdpEvidencePublisher
from drowsiness.vision import YUNET_NAME


def make_handler(engine, publisher=None):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type='application/json'):
            raw = json.dumps(body).encode() if content_type == 'application/json' else body
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def local_request(self):
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            origin = self.headers.get('Origin')
            return self.headers.get('Host') in allowed and (not origin or urlparse(origin).netloc in allowed)

        def do_GET(self):
            if not self.local_request():
                return self.reply(403, {'error': 'Local interface only.'})
            if self.path == '/api/status':
                with engine.lock:
                    status = engine.status()
                    status['qnx'] = publisher.status() if publisher else {'enabled': False, 'connected': False}
                    return self.reply(200, status)
            files = {'/': ('index.html', 'text/html; charset=utf-8'),
                     '/app.js': ('app.js', 'application/javascript; charset=utf-8'),
                     '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if self.path not in files:
                return self.reply(404, {'error': 'Not found'})
            name, content_type = files[self.path]
            self.reply(200, (ROOT / 'app/static' / name).read_bytes(), content_type)

        def do_POST(self):
            if not self.local_request():
                return self.reply(403, {'error': 'Local interface only.'})
            if self.path == '/api/reset':
                engine.reset()
                return self.reply(200, {'ok': True})
            if self.path == '/api/reload':
                return self.reply(200, engine.reload())
            if self.path != '/api/detect':
                return self.reply(404, {'error': 'Not found'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 2_000_000:
                    return self.reply(413, {'error': 'JPEG must be between 1 byte and 2 MB.'})
                self.connection.settimeout(10)
                raw = self.rfile.read(length)
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    return self.reply(400, {'error': 'Invalid JPEG image.'})
                if max(frame.shape[:2]) > 1920:
                    return self.reply(400, {'error': 'Frame dimensions exceed 1920 pixels.'})
                output = engine.detect(frame, self.headers.get('X-Session-ID', 'default'))
                if publisher:
                    publisher.publish(output)
                    output['qnx'] = publisher.status()
                else:
                    output['qnx'] = {'enabled': False, 'connected': False}
                return self.reply(200, output)
            except (ValueError, cv2.error) as exc:
                engine.reset()
                return self.reply(400, {'error': str(exc)})
            except Exception as exc:
                engine.reset()
                print(f'Inference error: {exc}', file=sys.stderr, flush=True)
                return self.reply(500, {'error': 'Inference failed; check the server terminal.'})

        def log_message(self, fmt, *args):
            if self.path != '/api/detect':
                super().log_message(fmt, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--models', type=Path, default=ROOT / 'models/trained')
    parser.add_argument('--detector', type=Path, default=ROOT / 'models/assets' / YUNET_NAME)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--qnx-host', help='Raspberry Pi QNX IPv4 address for UDP evidence')
    parser.add_argument('--qnx-port', type=int, default=45555)
    parser.add_argument('--udp-disabled', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.port <= 65535 or not 1 <= args.qnx_port <= 65535:
        parser.error('Invalid threads or port.')
    torch.set_num_threads(args.threads)
    engine = InferenceEngine(args.detector, args.models)
    publisher = UdpEvidencePublisher(args.qnx_host, args.qnx_port, not args.udp_disabled)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(engine, publisher))
    print(f'Open http://127.0.0.1:{args.port} — press Ctrl+C to stop.', flush=True)
    print(json.dumps(engine.status(), indent=2), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        publisher.close()


if __name__ == '__main__':
    main()
