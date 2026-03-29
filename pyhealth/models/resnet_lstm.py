from typing import Dict
import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel, PsdFeatureExtractor


RAW = "raw"
PSD = "psd"

FEATURE_EXTRACTORS = nn.ModuleDict([
    [RAW, None],
    [PSD, PsdFeatureExtractor()],
])


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=(1,9), stride=(1,stride), padding=(0,4), bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=(1,9), stride=1, padding=(0,4), bias=False),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=(1,stride), bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        out = self.net(x)
        out += self.shortcut(x)
        return torch.relu(out)


class Conv2dResNetLSTM(BaseModel):
    def __init__(
        self,
        dataset: SampleDataset,
        encoder: str,
        num_layers: int,
        in_channel: int,
        output_dim: int,
        batch_size: int,
        device: str,
        dropout: float = 0.5,
    ):
        super().__init__(dataset=dataset)

        self.encoder = encoder
        self.num_layers = num_layers
        self.in_channel = in_channel
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = 256

        self.feature_extractor = FEATURE_EXTRACTORS[self.encoder]
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if self.encoder == RAW:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, 64, kernel_size=(1,51), stride=(1,4), padding=(0,25)),
                nn.BatchNorm2d(64),
                self.activation,
            )
            self.maxpool = nn.MaxPool2d((1,4), (1,4))

        elif self.encoder == PSD:
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(7,21), stride=(7,2), padding=(0,10)),
                nn.BatchNorm2d(64),
                self.activation,
            )
            self.maxpool = nn.MaxPool2d((1,2), (1,2))

        self.in_planes = 64
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)

        self.agvpool = nn.AdaptiveAvgPool2d((1,1))

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Linear(64, self.output_dim),
        )

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1]*(num_blocks-1)

        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
            x = x.reshape(x.size(0), -1, x.size(3)).unsqueeze(1)
        else:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.agvpool(x)
        x = torch.squeeze(x, 2)
        x = x.permute(0, 2, 1)

        hidden = (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),
        )

        output, hidden = self.lstm(x, hidden)
        output = output[:, -1, :]

        output = self.classifier(output)

        return output, hidden