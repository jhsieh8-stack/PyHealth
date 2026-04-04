import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel, PsdFeatureExtractor


RAW = 'raw'
PSD = 'psd'
FEATURE_EXTRACTORS = nn.ModuleDict([
    [ RAW, None ],
    [ PSD, PsdFeatureExtractor() ],
])
CONV2D = {
    RAW: {
        'kernel_1': (1,51), 'stride_1': (1,4), 'padding_1': (0,25),
        'kernel_2': (1,21), 'stride_2': (1,2), 'padding_2': (0,10),
        'kernel_3': (1,9), 'stride_3': (1,2), 'padding_3': (0,4),
    },
    PSD: {
        'kernel_1': (7,21), 'stride_1': (7,2), 'padding_1': (0,10),
        'kernel_2': (1,21), 'stride_2': (1,2), 'padding_2': (0,10),
        'kernel_3': (1,9), 'stride_3': (1,1), 'padding_3': (0,4),
    }
}
MAX2D = {
    RAW: { 'kernel': (1,4), 'stride': (1,4) },
    PSD: { 'kernel': (1,2), 'stride': (1,2) },
}

CONV = 'conv'
MAXPOOL = 'maxpool'
ORDER2D = {
    RAW: { CONV, MAXPOOL, CONV, CONV},
    PSD: { CONV, CONV, MAXPOOL, CONV},
}

class EEGRealTimeDetectorModel(BaseModel):
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
        super(EEGRealTimeDetectorModel, self).__init__(
            dataset=dataset,
        )
        self.encoder = encoder
        self.num_layers = num_layers
        self.in_channel = in_channel
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = device
        self.feature_extractor = FEATURE_EXTRACTORS[self.encoder]
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        def conv2d(in_channel, out_channel, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channel),
                self.activation,
                nn.Dropout(self.dropout),
            )
        
        layers = []
        conv_count = 1
        conv_in_channels = [self.in_channel, 64, 128]
        conv_out_channels = [64, 128, 256]
        conv2d_pms = CONV2D[self.encoder]
        max2d_pms = MAX2D[self.encoder]
        for layer in ORDER2D[self.encoder]:
            if layer == CONV:
                layers.append(conv2d(
                    conv_in_channels[conv_count-1],
                    conv_out_channels[conv_count-1],
                    conv2d_pms[f"kernel_{conv_count}"],
                    conv2d_pms[f"stride_{conv_count}"],
                    conv2d_pms[f"padding_{conv_count}"],
                ))
                conv_count += 1
            elif layer == MAXPOOL:
                layers.append(nn.MaxPool2d(
                    kernel_size=max2d_pms['kernel'],
                    stride=max2d_pms['stride'],
                ))
        self.feature_extractor_cnn = nn.Sequential(*layers)

        self.agvpool = nn.AdaptiveAvgPool2d((1,1))

        self.hidden_dim = 256
        self.lstm = nn.LSTM(
                input_size=256,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout) 

        self.classifier = nn.Sequential(
                nn.Linear(in_features=self.hidden_dim, out_features= 64, bias=True),
                nn.BatchNorm1d(64),
                self.activation,
                nn.Linear(in_features=64, out_features= self.output_dim, bias=True),
        )
        
    def forward(self, x):
        """
        """
        # feature extractor
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = x.reshape(x.size(0), -1, x.size(3)).unsqueeze(1)

        # conv2d after feature extractor
        x = self.feature_extractor_cnn(x)

        # CNN2D_LSTM
        x = self.agvpool(x)
        x = torch.squeeze(x, 2)
        x = x.permute(0, 2, 1)

        self.hidden = (
            (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device), 
             torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))
        )
        output, self.hidden = self.lstm(x, self.hidden)    
        output = output[:,-1,:]
        output = self.classifier(output)
        return output, self.hidden
