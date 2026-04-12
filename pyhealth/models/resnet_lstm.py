# Description: ResNetLSTM model implementation for PyHealth 2.0

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from .eeg_feature_extractors import FeatureExtractorManager, RESNET_LSTM


class BasicBlock(nn.Module):
    """Residual building block for the 2D convolutional ResNet encoder.

    This block applies two (1×9) convolutions with batch normalization and a
    residual (shortcut) connection. The shortcut uses a (1×1) convolution to
    match channel dimensions when they differ.

    Args:
        in_planes: number of input channels.
        planes: number of output channels.
        stride: stride applied to the first convolution and the shortcut
            projection. Default is 1.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: input tensor of shape [batch size, in_planes, height, width].

        Returns:
            output tensor of shape [batch size, planes, height, width // stride].
        """
        out = self.net(x)
        out += self.shortcut(x)
        return torch.relu(out)


class ResNetLSTM(BaseModel):
    """2-D ResNet + LSTM model for ECG / multivariate timeseries classification.

    The model processes each input sample through a configurable feature
    extractor, a stack of residual convolutional blocks (ResNet), and a
    multi-layer LSTM. The final LSTM hidden state is passed to a small
    classification head that produces the output logits.

    Pipeline:
        1. An optional pre-trained ``feature_extractor`` (e.g. a spectrogram
           or wavelet transform) reshapes the raw signal into a 2-D
           representation.
        2. ``feature_extractor_cnn`` applies two initial Conv2d layers to
           project the signal into a 64-channel feature map.
        3. Three ``BasicBlock`` residual stages progressively increase the
           channel depth (64 → 128 → 256) while halving the temporal
           resolution in stages 2 and 3.
        4. Adaptive average pooling collapses the spatial dimension to 1×1,
           yielding a per-frame feature vector of size 256.
        5. The LSTM models temporal dependencies across frames; only the last
           output step is retained.
        6. A two-layer fully-connected head maps the LSTM output to
           ``output_dim`` logits.

    Args:
        dataset (SampleDataset): dataset with fitted input and output processors.
        encoder (str): name of the pre-trained feature extractor backbone.
        num_layers (int): number of LSTM layers.
        in_channel (int): number of input channels (e.g. ECG leads).
        output_dim (int): number of output classes / regression targets.
        batch_size (int): batch size; used to initialise the LSTM hidden state.
        device (str): device string passed to hidden-state initialisation (e.g. ``"cuda"``).
        dropout (float): dropout probability applied after activations and
            between LSTM layers. Default is 0.5.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "ecg": [[0.1, 0.2, ...], ...],   # (leads, timesteps)
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "ecg": [[0.3, 0.1, ...], ...],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"ecg": "timeseries"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="toy",
        ... )
        >>> model = ResNetLSTM(
        ...     dataset,
        ...     encoder="resnet",
        ...     num_layers=2,
        ...     in_channel=12,
        ...     output_dim=2,
        ...     batch_size=32,
        ...     device="cpu",
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        encoder: str,
        num_layers: int,
        output_dim: int,
        batch_size: int,
        dropout: float = 0.5,
    ):
        super().__init__(dataset=dataset)

        self.encoder = encoder
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.hidden_dim = 256

        self.feature_manager = FeatureExtractorManager(model=RESNET_LSTM, encoder=self.encoder)
        self.feature_extractor = self.feature_manager.get_feature_extractor()
        self.feature_transformer = self.feature_manager.transform_features
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.feature_extractor_cnn = self.feature_manager.get_feature_extractor_cnn(
            activation        = self.activation,
            dropout           = self.dropout,
        )

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  num_blocks=2, stride=1)  
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

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Build a residual stage consisting of ``num_blocks`` BasicBlocks.

        The first block uses the supplied ``stride`` to optionally downsample;
        all subsequent blocks use stride=1. ``self.in_planes`` is updated in
        place so the next stage picks up the correct channel count.

        Args:
            planes: number of output channels for every block in this stage.
            num_blocks: number of BasicBlocks to stack.
            stride: stride applied only to the first block of the stage.

        Returns:
            nn.Sequential containing the stacked BasicBlocks.
        """
        layers = []
        strides = [stride] + [1]*(num_blocks-1)

        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Forward propagation.

        Args:
            x: input tensor of shape [batch size, sequence length, channels].

        Returns:
            output: logit tensor of shape [batch size, output_dim].
            hidden: tuple of final LSTM hidden and cell states, each of shape
                [num_layers, batch size, hidden_dim].
        """
        x = x.permute(0, 2, 1)

        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        x = self.feature_transformer(x)
        x = self.feature_extractor_cnn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.agvpool(x)
        x = torch.squeeze(x, 2)
        x = x.permute(0, 2, 1)

        hidden = (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        )

        output, hidden = self.lstm(x, hidden)
        output = output[:, -1, :]

        output = self.classifier(output)

        return output, hidden