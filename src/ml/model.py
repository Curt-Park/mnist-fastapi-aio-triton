"""A simple image classification model for MNIST.

- Author: Jinwoo Park
- Email: www.jwpark.co.kr@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn


class ConvNet(nn.Module):
    """A simple image classification model for MNIST."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward."""
        feat1 = F.relu(self.conv1(inp))
        feat2 = F.relu(self.conv2(feat1))
        cnn_feat = self.dropout1(F.max_pool2d(feat2, 2))
        cnn_feat = torch.flatten(cnn_feat, 1)

        fc_feat = self.dropout2(F.relu(self.fc1(cnn_feat)))
        logits = self.fc2(fc_feat)
        output = F.log_softmax(logits, dim=1)
        return output
