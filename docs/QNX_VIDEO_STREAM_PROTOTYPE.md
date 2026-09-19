# Optional Windows-to-QNX video prototype

This prototype is separate from the working jury evidence pipeline. Windows sends
640x480 JPEG frames at 5 FPS over UDP port 45556. QNX reassembles every frame,
checks its CRC32, atomically replaces `/tmp/drive_x_latest.jpg`, and decodes the
frame to RGB888 through the native QNX Image Library.

It proves compressed video transport and native RGB decoding on QNX. It does not
yet run face detection or MobileNet on QNX; those require an ARM64 QNX inference
runtime on the target image.

## Run

Copy `build/vigil_video_rx` to the Pi. As root on QNX:

```sh
chmod 755 build/vigil_video_rx
./build/vigil_video_rx 45556 /tmp/drive_x_latest.jpg
```

Close the browser camera first, because Windows normally permits only one process
to own the webcam. On Windows:

```powershell
python scripts/stream_webcam_udp.py --host 192.168.137.135 --port 45556 --fps 5 --preview
```

The QNX terminal reports completed, decoded, decode-failed, dropped, and rejected
frames, along with JPEG decode time and an RGB buffer CRC. Verify:

```sh
ls -l /tmp/drive_x_latest.jpg
```

Copy one received frame back to Windows for visual verification:

```powershell
scp -O -o KexAlgorithms=curve25519-sha256@libssh.org -o Ciphers=aes128-ctr -o MACs=hmac-sha2-256 qnxuser@192.168.137.135:/tmp/drive_x_latest.jpg .
```

UDP does not guarantee delivery. An incomplete frame is dropped when a newer
frame begins. This bounds memory and latency rather than displaying stale video.
