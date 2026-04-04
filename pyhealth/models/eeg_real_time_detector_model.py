import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel, get_feature_extractor, get_feature_extractor_cnn, CNN_LSTM


class EEGCnnLstmModel(BaseModel):
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
