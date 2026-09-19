"""Prepare subject-disjoint MRL Eyes splits directly from ZIP or a directory."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import zipfile

from PIL import Image

PATTERN = re.compile(r"(s\d+)_\d+_[01]_[01]_([01])_[012]_[01]_\d+\.png", re.I)
CLASSES = {"closed": 0, "open": 1}


def parse_name(name):
    match = PATTERN.fullmatch(Path(name).name)
    if not match:
        raise ValueError(f"Unexpected MRL filename: {name}")
    return match[1].lower(), int(match[2])


def split_subjects(subjects, seed, val_fraction, test_fraction):
    subjects = sorted(set(subjects))
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required.")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Split fractions must be positive and sum to less than one.")
    random.Random(seed).shuffle(subjects)
    nv = max(1, round(len(subjects) * val_fraction))
    nt = max(1, round(len(subjects) * test_fraction))
    if nv + nt >= len(subjects):
        raise ValueError("Split fractions leave no training subjects.")
    return {s: split for split, group in (
        ("val", subjects[:nv]), ("test", subjects[nv:nv + nt]),
        ("train", subjects[nv + nt:])) for s in group}


def prepare(source, output, seed=42, val_fraction=0.2, test_fraction=0.1):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        names = sorted(n for n in archive.namelist() if n.lower().endswith('.png')) if archive else sorted(
            str(p.relative_to(source)) for p in source.rglob('*.png'))
        parsed = [(name, *parse_name(name)) for name in names]
        assignments = split_subjects([r[1] for r in parsed], seed, val_fraction, test_fraction)
        output.mkdir(parents=True)
        rows, rejected, hashes = {}, [], {}
        for index, (name, subject, label) in enumerate(parsed, 1):
            raw = archive.read(name) if archive else (source / name).read_bytes()
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    rgb = im.convert('RGB')
                    digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            except (OSError, ValueError) as exc:
                rejected.append({'source': name, 'reason': str(exc)})
                continue
            if digest in hashes:
                if hashes[digest] != label:
                    previous = rows.pop(digest, None)
                    if previous:
                        (output / previous['path']).unlink()
                        rejected.append({'source': previous['source'], 'reason': 'conflicting duplicate labels'})
                    hashes[digest] = None
                reason = 'conflicting duplicate labels' if hashes[digest] is None else 'duplicate pixels'
                rejected.append({'source': name, 'reason': reason})
                continue
            hashes[digest] = label
            split = assignments[subject]
            category = ('closed', 'open')[label]
            relative = Path(split) / category / Path(name).name
            destination = output / relative
            if destination.exists():
                raise ValueError(f"Duplicate filename: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            rows[digest] = dict(path=relative.as_posix(), split=split, subject=subject,
                               label=label, source=name, sha256=digest)
            if index % 5000 == 0 or index == len(parsed):
                print(f'Validated {index:,}/{len(parsed):,} eye images...', flush=True)
        rows = list(rows.values())
        counts = Counter((r['split'], r['label']) for r in rows)
        for split in ('train', 'val', 'test'):
            if any(counts[split, label] == 0 for label in CLASSES.values()):
                raise ValueError(f"{split} lacks a class; inspect source or choose another seed.")
        with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = dict(task='eye_state', class_to_idx=CLASSES, seed=seed,
                        source=str(source.resolve()), subject_splits=assignments,
                        counts={s: {c: counts[s, i] for c, i in CLASSES.items()}
                                for s in ('train', 'val', 'test')}, rejected=rejected)
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(json.dumps(metadata['counts'], indent=2))
        print(f"Prepared {len(rows)} images; rejected {len(rejected)}. Output: {output}")
    finally:
        if archive:
            archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('dataset/mrlEyes_2018_01.zip'))
    parser.add_argument('--output', type=Path, default=Path('datasets/mrl_eyes_clean'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed, args.val_fraction, args.test_fraction)


if __name__ == '__main__':
    main()
