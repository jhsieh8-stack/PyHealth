# Description: EEGCnnLstmModel implementation for PyHealth 2.0

import torch
from torch import nn
from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel, get_feature_extractor, get_feature_extractor_cnn, CNN_LSTM


class EEGCnnLstmModel(BaseModel):
    """CNN + LSTM model for EEG classification in PyHealth 2.0.

    The model processes raw multi-channel EEG signals through a pre-trained
    feature extractor, a three-stage 2-D convolutional encoder, and a
    multi-layer LSTM. The final LSTM output is passed to a small fully
    connected classification head.

    Pipeline:
        1. A pre-trained ``feature_extractor`` (e.g. a spectrogram or wavelet
           transform) converts the raw EEG signal into a 2-D time–frequency
           representation.
        2. ``feature_extractor_cnn`` applies three successive Conv2d stages
           to project the representation through 64 → 128 → 256 channels.
        3. Adaptive average pooling collapses the spatial dimension to 1×1,
           yielding a per-frame feature vector of size 256.
        4. The LSTM models temporal dependencies across frames; only the last
           output step is retained.
        5. A two-layer fully-connected head maps the LSTM output to
           ``output_dim`` logits.

    Args:
        dataset (SampleDataset): dataset with fitted input and output processors.
        encoder (str): name of the pre-trained feature extractor backbone.
        num_layers (int): number of LSTM layers.
        in_channel (int): number of input channels (i.e. EEG electrodes / leads).
        output_dim (int): number of output classes / regression targets.
        batch_size (int): batch size; used to initialise the LSTM hidden state.
        device (str): device string for hidden-state initialisation (e.g. ``"cuda"``).
        dropout (float): dropout probability applied after activations and
            between LSTM layers. Default is 0.5.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "eeg": [[0.1, 0.2, ...], ...],   # (electrodes, timesteps)
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "eeg": [[0.3, 0.1, ...], ...],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"eeg": "timeseries"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="toy",
        ... )
        >>> model = EEGCnnLstmModel(
        ...     dataset,
        ...     encoder="stft",
        ...     num_layers=2,
        ...     in_channel=64,
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
            in_channel: int,
            output_dim: int,
            batch_size: str,
            device: str,
            dropout: float = 0.5,
        ):
        super(EEGCnnLstmModel, self).__init__(
            dataset=dataset,
        )
        self.encoder = encoder
        self.num_layers = num_layers
        self.in_channel = in_channel
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = device

        self.feature_extractor = get_feature_extractor(self.encoder)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = 256

        self.feature_extractor_cnn = get_feature_extractor_cnn(
            model             = CNN_LSTM,
            encoder           = self.encoder,
            conv_in_channels  = [self.in_channel, 64, 128],
            conv_out_channels = [64, 128, 256],
            activation        = self.activation,
            dropout           = self.dropout,
        )

        self.agvpool = nn.AdaptiveAvgPool2d((1,1))

        self.lstm = nn.LSTM(
                input_size=256,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout)

        self.classifier = nn.Sequential(
                nn.Linear(in_features=self.hidden_dim, out_features=64, bias=True),
                nn.BatchNorm1d(64),
                self.activation,
                nn.Linear(in_features=64, out_features=self.output_dim, bias=True),
        )

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

        x = self.feature_extractor(x)
        x = x.reshape(x.size(0), -1, x.size(3)).unsqueeze(1)

        x = self.feature_extractor_cnn(x)

        x = self.agvpool(x)
        x = torch.squeeze(x, 2)
        x = x.permute(0, 2, 1)

        self.hidden = (
            (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),
             torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))
        )

        output, self.hidden = self.lstm(x, self.hidden)
        output = output[:, -1, :]

        output = self.classifier(output)

        return output, self.hidden