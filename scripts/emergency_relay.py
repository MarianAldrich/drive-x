"""Receive DRIVE-X QNX emergency datagrams and optionally send them with Twilio.

The QNX alert service remains the safety decision source. This small gateway only
bridges its LAN datagram to an HTTPS SMS API because TLS is not assumed on the Pi
image. Without credentials it provides a deterministic, timestamped demo log.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import urllib.parse
import urllib.request


def message_text(event: dict) -> str:
    location = 'GPS fix unavailable'
    if event.get('gps_valid'):
        location = (f"Latitude: {float(event['latitude']):.7f}\n"
                    f"Longitude: {float(event['longitude']):.7f}\n"
                    f"GPS time: {event.get('gps_time', 'unknown')}")
    return ("DROWSINESS ALERT\n\nProlonged driver drowsiness detected.\n\n"
            f"{location}\n\nPlease check the driver.")


def send_twilio(text: str) -> str:
    sid = os.environ.get('TWILIO_ACCOUNT_SID')
    token = os.environ.get('TWILIO_AUTH_TOKEN')
    sender = os.environ.get('TWILIO_FROM_NUMBER')
    recipient = os.environ.get('DRIVEX_EMERGENCY_NUMBER')
    if not all((sid, token, sender, recipient)):
        return 'PRINT_ONLY'
    payload = urllib.parse.urlencode({'From': sender, 'To': recipient, 'Body': text}).encode()
    request = urllib.request.Request(
        f'https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json',
        data=payload, method='POST')
    request.add_header('Authorization', 'Basic ' + base64.b64encode(
        f'{sid}:{token}'.encode()).decode())
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    with urllib.request.urlopen(request, timeout=15) as response:
        result = json.loads(response.read())
    return str(result.get('sid', 'SENT'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=45557)
    parser.add_argument('--log', type=Path, default=Path('results/emergency_relay.jsonl'))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind((args.bind, args.port))
    print(f'DRIVE-X emergency relay listening on UDP {args.bind}:{args.port}', flush=True)
    print('SMS activates only when all TWILIO_* and DRIVEX_EMERGENCY_NUMBER variables are set.',
          flush=True)
    try:
        while True:
            payload, source = receiver.recvfrom(4096)
            try:
                event = json.loads(payload.decode('utf-8'))
                if event.get('type') != 'DROWSINESS_ALERT':
                    raise ValueError('unexpected event type')
                text = message_text(event)
                try:
                    result = send_twilio(text)
                except Exception as exc:  # preserve the event even if the provider fails
                    result = f'FAILED: {exc}'
                record = {'received_utc': datetime.now(timezone.utc).isoformat(),
                          'source': source[0], 'delivery': result, 'event': event}
                with args.log.open('a', encoding='utf-8') as log:
                    log.write(json.dumps(record, separators=(',', ':')) + '\n')
                print('\n' + text + f'\nDelivery: {result}\n', flush=True)
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                print(f'Rejected emergency datagram from {source[0]}: {exc}', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()


if __name__ == '__main__':
    main()
