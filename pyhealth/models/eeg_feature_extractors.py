import torchaudio

import torch
from torch import nn

class PsdFeatureExtractor(nn.Module):
    def __init__(self,
            sample_rate: int = 200,
            frame_length: int = 16,
            frame_shift: int = 8,
            feature_extract_by: str = 'kaldi'):
        super(PsdFeatureExtractor, self).__init__()

        self.sample_rate = sample_rate
        self.feature_extract_by = feature_extract_by.lower()
        self.freq_resolution = 1
        self.n_fft = self.freq_resolution*self.sample_rate
        self.hop_length = frame_shift
        self.frame_length = frame_length

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.frame_length
        ) if feature_extract_by == 'kaldi' else torch.stft(
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            window=torch.hamming_window(self.frame_length),
            center=False,
            normalized=False,
            onesided=True
        )
        
    def psd(self, amp, begin, end):
        return torch.mean(amp[begin*self.freq_resolution:end*self.freq_resolution], 0)
        
    def forward(self, batch):
        psds_batch = []

        for signals in batch:
            psd_sample = []
            for signal in signals:
                stft = self.stft(signal)
                amp = (torch.log(torch.abs(stft) + 1e-10))

                psd1 = self.psd(amp,0,4)
                psd2 = self.psd(amp,4,8)
                psd3 = self.psd(amp,8,12)
                psd4 = self.psd(amp,12,30)
                psd5 = self.psd(amp,30,50)
                psd6 = self.psd(amp,50,70)
                psd7 = self.psd(amp,70,100)
                
                psds = torch.stack((psd1, psd2, psd3, psd4, psd5, psd6, psd7))
                psd_sample.append(psds)

            psds_batch.append(torch.stack(psd_sample))

        return torch.stack(psds_batch)


CNN_LSTM = 'cnn_lstm'
RESNET_LSTM = 'resnet_lstm'
RAW = 'raw'
PSD = 'psd'

FEATURE_EXTRACTORS = nn.ModuleDict([
    [ RAW, None ],
    [ PSD, PsdFeatureExtractor() ],
])

CONV2D = {
    RAW: {
        'in_1': 1,   'out_1': 64,   'kernel_1': (1,51), 'stride_1': (1,4), 'padding_1': (0,25),
        'in_2': 64,  'out_2': 128,  'kernel_2': (1,21), 'stride_2': (1,2), 'padding_2': (0,10),
        'in_3': 128, 'out_3': 256,  'kernel_3': (1,9),  'stride_3': (1,2), 'padding_3': (0,4),
    },
    PSD: {
        'in_1': 1,   'out_1': 64,   'kernel_1': (7,21), 'stride_1': (7,2), 'padding_1': (0,10),
        'in_2': 64,  'out_2': 128,  'kernel_2': (1,21), 'stride_2': (1,2), 'padding_2': (0,10),
        'in_3': 128, 'out_3': 256,  'kernel_3': (1,9), 'stride_3': (1,1), 'padding_3': (0,4),
    }
}

MAX2D = {
    RAW: { 'kernel': (1,4), 'stride': (1,4) },
    PSD: { 'kernel': (1,2), 'stride': (1,2) },
}

CONV = 'conv'
MAXPOOL = 'maxpool'
ORDER2D = {
    CNN_LSTM: {
        RAW: { CONV, MAXPOOL, CONV, CONV },
        PSD: { CONV, CONV, MAXPOOL, CONV },
    },
    RESNET_LSTM: {
        RAW: { CONV, MAXPOOL },
        PSD: { CONV, MAXPOOL },
    },
}


def get_feature_extractor(encoder):
    return FEATURE_EXTRACTORS[encoder]

def feature_extractor_conv2d(
    model,
    in_channel,
    out_channel,
    kernel_size,
    stride,
    padding,
    activation,
    dropout,
):
    if model == CNN_LSTM:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            activation,
            dropout,
        )
    elif model == RESNET_LSTM:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1),
            nn.BatchNorm2d(out_channel),
            activation,
        )

def get_feature_extractor_cnn(
    model,
    encoder,
    activation,
    dropout,
):
    layers = []
    conv_count = 1
    conv2d_pms = CONV2D[encoder]
    max2d_pms = MAX2D[encoder]
    for layer in ORDER2D[model][encoder]:
        if layer == CONV:
            layers.append(feature_extractor_conv2d(
                model,
                conv2d_pms[f"in_{conv_count}"],
                conv2d_pms[f"out_{conv_count}"],
                conv2d_pms[f"kernel_{conv_count}"],
                conv2d_pms[f"stride_{conv_count}"],
                conv2d_pms[f"padding_{conv_count}"],
                activation,
                dropout,
            ))
            conv_count += 1
        elif layer == MAXPOOL:
            layers.append(nn.MaxPool2d(
                kernel_size=max2d_pms['kernel'],
                stride=max2d_pms['stride'],
            ))
    return nn.Sequential(*layers)

def transform_features(
    model,
    encoder,
    x
):
    if encoder == RAW:
        return x.unsqueeze(1)
    elif encoder == PSD:
        return x.reshape(x.size(0), -1, x.size(3)).unsqueeze(1)
    return x