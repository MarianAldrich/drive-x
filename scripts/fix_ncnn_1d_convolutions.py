"""Convert MobileNet SE 1D convolutions to ncnn InnerProduct layers.

Recent pnnx exports global-pool squeeze/excitation layers as 1x1 Convolution.
The portable ncnn backend treats the pooled tensor as one-dimensional and asks
for InnerProduct instead. Weight bytes are layout-compatible for a 1x1 kernel,
so only the param description changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def convert(source: Path, destination: Path) -> int:
    lines = source.read_text(encoding="utf-8").splitlines()
    pending = 0
    converted = 0
    output: list[str] = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "Pooling" and fields[1].startswith("gap_"):
            # The final classifier GAP is followed by Reshape, not SE layers.
            pending = 2
        elif pending and len(fields) >= 4 and fields[0] == "Convolution":
            bottom_count, top_count = int(fields[2]), int(fields[3])
            params_at = 4 + bottom_count + top_count
            params = dict(token.split("=", 1) for token in fields[params_at:] if "=" in token)
            # Only convert a 1x1 convolution with a bias. Spatial convolutions
            # after unrelated pooling operations remain untouched.
            if params.get("1") == "1" and params.get("11") == "1" and params.get("5") == "1":
                replacement = fields[:]
                replacement[0] = "InnerProduct"
                new_params = [f"0={params['0']}", "1=1", f"2={params['6']}"]
                for key in ("8", "9", "10"):
                    if key in params:
                        new_params.append(f"{key}={params[key]}")
                replacement = replacement[:params_at] + new_params
                line = " ".join(replacement)
                pending -= 1
                converted += 1
        elif pending and len(fields) >= 1 and fields[0] in {"Reshape", "Flatten", "Pooling"}:
            pending = 0
        output.append(line)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = convert(args.source, args.destination)
    if count == 0:
        raise SystemExit("No compatible 1D convolution layers found")
    print(f"converted={count} output={args.destination}")


if __name__ == "__main__":
    main()
