"""MobileNetV3-Small encoder with video-level pooling, including weak-label MIL."""
import math

import torch
from torch import nn
from torchvision import models, transforms

TASKS = {'drowsiness': ['alert', 'low_vigilance', 'drowsy'], 'yawn': ['no_yawn', 'yawn']}
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]


def image_transform(training=False):
    operations = [transforms.Resize(256), transforms.CenterCrop(224)]
    if training:
        operations += [transforms.RandomHorizontalFlip(),
                       transforms.ColorJitter(brightness=.15, contrast=.15)]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class VideoMobileNet(nn.Module):
    def __init__(self, task, pretrained=False, top_fraction=.25):
        super().__init__()
        if task not in TASKS or not 0 < top_fraction <= 1:
            raise ValueError('Invalid model task or pooling fraction.')
        self.task, self.top_fraction = task, top_fraction
        self.encoder = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.classifier[-1] = nn.Linear(self.encoder.classifier[-1].in_features, len(TASKS[task]))

    def pool(self, logits):
        if self.task == 'drowsiness':
            return logits.mean(dim=1)
        # Positive videos are bags that contain a yawn, not positive labels for every frame.
        # Pool the positive log-odds of the top quarter of frames in each bag.
        evidence = logits[..., 1] - logits[..., 0]
        count = max(1, math.ceil(evidence.shape[1] * self.top_fraction))
        score = evidence.topk(count, dim=1).values.mean(dim=1)
        return torch.stack((torch.zeros_like(score), score), dim=-1)

    def forward(self, clips):
        b, t, c, h, w = clips.shape
        logits = self.encoder(clips.reshape(b * t, c, h, w)).reshape(b, t, -1)
        return self.pool(logits)
