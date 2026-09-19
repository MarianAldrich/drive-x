"""Train MobileNetV3-Small on prepared MRL eye crops; run on the development PC."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def metrics(confusion):
    matrix = confusion.tolist()
    scores = []
    for i, name in enumerate(('closed', 'open')):
        tp = matrix[i][i]
        precision = tp / max(1, sum(row[i] for row in matrix))
        recall = tp / max(1, sum(matrix[i]))
        scores.append(dict(class_name=name, precision=precision, recall=recall,
                           f1=2 * precision * recall / max(1e-12, precision + recall)))
    return dict(accuracy=confusion.diag().sum().item() / max(1, confusion.sum().item()),
                macro_f1=sum(s['f1'] for s in scores) / 2,
                per_class=scores, confusion_matrix=matrix)


def run_epoch(model, loader, device, criterion, optimizer=None, frozen=False, progress=None):
    model.train(optimizer is not None)
    if frozen:
        model.features.eval()  # Keep frozen backbone BatchNorm statistics unchanged.
    confusion = torch.zeros(2, 2, dtype=torch.int64)
    loss_sum, count = 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        started = time.perf_counter()
        for batch_index, (images, labels) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
            indices = labels.detach().cpu() * 2 + logits.argmax(1).detach().cpu()
            confusion += torch.bincount(indices, minlength=4).reshape(2, 2)
            if progress and (batch_index % 25 == 0 or batch_index == len(loader)):
                elapsed = max(time.perf_counter() - started, 1e-6)
                rate = count / elapsed
                remaining = (len(loader) - batch_index) * loader.batch_size / max(rate, 1e-6)
                print(f'{progress}: batch {batch_index}/{len(loader)} | '
                      f'{rate:.0f} images/s | ETA {remaining / 60:.1f} min', flush=True)
    return dict(loss=loss_sum / count, **metrics(confusion))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('datasets/mrl_eyes_clean'))
    parser.add_argument('--output', type=Path, default=Path('models/original/eye_state'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--freeze-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--weights', choices=('imagenet', 'none'), default='imagenet')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0 or args.workers < 0 or args.freeze_epochs < 0:
        parser.error('Invalid training arguments.')
    if args.output.exists() and not args.resume:
        parser.error('Output already exists; choose a new run directory or use --resume.')
    if args.resume and not (args.output / 'last.pt').is_file():
        parser.error('--resume requires last.pt in the output directory.')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    metadata = json.loads((args.data / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['task'] != 'eye_state' or metadata['class_to_idx'] != {'closed': 0, 'open': 1}:
        raise ValueError('Expected prepared MRL eye-state dataset.')
    manifest_path = args.data / 'manifest.csv'
    with manifest_path.open(encoding='utf-8', newline='') as f:
        manifest = list(csv.DictReader(f))
    subjects, hashes = {}, {}
    for row in manifest:
        for mapping, key in ((subjects, row['subject']), (hashes, row['sha256'])):
            previous = mapping.setdefault(key, row['split'])
            if previous != row['split']:
                raise ValueError('Subject or image leakage across dataset splits.')
    evaluation = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    augmentation = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    data = {s: datasets.ImageFolder(args.data / s, transform=augmentation if s == 'train' else evaluation)
            for s in ('train', 'val', 'test')}
    for split, dataset in data.items():
        if dataset.class_to_idx != metadata['class_to_idx']:
            raise ValueError(f'Class mismatch in {split}.')
        actual = {(Path(p).relative_to(args.data).as_posix(), label) for p, label in dataset.samples}
        expected = {(r['path'], int(r['label'])) for r in manifest if r['split'] == split}
        if actual != expected:
            raise ValueError(f'{split} files do not match the preparation manifest.')
    print(f'Device: {device}; images: ' + ', '.join(f'{s}={len(d):,}' for s, d in data.items()), flush=True)
    loader_options = dict(num_workers=args.workers, pin_memory=device.type == 'cuda')
    if args.workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=s == 'train',
                            generator=torch.Generator().manual_seed(args.seed), **loader_options)
               for s, d in data.items()}
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                                      if args.weights == 'imagenet' and not args.resume else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': args.lr * 0.1},
        {'params': model.classifier.parameters(), 'lr': args.lr}], weight_decay=1e-4)
    counts = torch.bincount(torch.tensor(data['train'].targets), minlength=2)
    class_weights = (counts.sum() / (2 * counts.float())).to(device)
    # Unreduced loss keeps sample averaging correct with class weighting.
    train_criterion = lambda logits, labels: nn.functional.cross_entropy(
        logits, labels, weight=class_weights, reduction='none').mean()
    criterion = nn.CrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(architecture='mobilenet_v3_small', task='eye_state',
                  class_to_idx=metadata['class_to_idx'], torch_version=str(torch.__version__),
                  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  preprocessing=dict(color='RGB', resize_short_edge=256, center_crop=224,
                                     interpolation='bilinear', antialias=True, scale=1/255,
                                     mean=MEAN, std=STD, layout='NCHW'))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    best, stale, history, start_epoch = -1., 0, [], 0
    if args.resume:
        last = torch.load(args.output / 'last.pt', map_location=device, weights_only=True)
        model.load_state_dict(last['model_state_dict'])
        optimizer.load_state_dict(last['optimizer_state_dict'])
        start_epoch = int(last['epoch'])
        history_path = args.output / 'history.json'
        history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.is_file() else []
        if len(history) != start_epoch:
            raise ValueError('Checkpoint epoch and history length do not match.')
        scores = [entry['val']['macro_f1'] for entry in history]
        best = max(scores, default=-1.)
        best_index = max(range(len(scores)), key=scores.__getitem__) if scores else -1
        stale = len(scores) - best_index - 1
        print(f'Resuming after epoch {start_epoch}; best validation F1={best:.4f}.', flush=True)
    for epoch in range(start_epoch, args.epochs):
        frozen = args.weights == 'imagenet' and epoch < args.freeze_epochs
        for parameter in model.features.parameters():
            parameter.requires_grad_(not frozen)
        train = run_epoch(model, loaders['train'], device, train_criterion, optimizer, frozen,
                          progress=f'Epoch {epoch + 1} train')
        val = run_epoch(model, loaders['val'], device, criterion, progress=f'Epoch {epoch + 1} validation')
        history.append(dict(epoch=epoch + 1, train=train, val=val))
        checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                          epoch=epoch + 1, config=config, val=val)
        torch.save(checkpoint, args.output / 'last.pt')
        if val['macro_f1'] > best:
            best, stale = val['macro_f1'], 0
            torch.save(checkpoint, args.output / 'best.pt')
        else:
            stale += 1
        (args.output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f"Epoch {epoch + 1}: train loss={train['loss']:.4f}, val F1={val['macro_f1']:.4f}", flush=True)
        if stale >= args.patience and epoch >= args.freeze_epochs:
            break
    best_checkpoint = torch.load(args.output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    test = run_epoch(model, loaders['test'], device, criterion, progress='Test')
    test['selected_epoch'] = best_checkpoint['epoch']
    (args.output / 'test_metrics.json').write_text(json.dumps(test, indent=2), encoding='utf-8')
    print(json.dumps(test, indent=2))


if __name__ == '__main__':
    main()
