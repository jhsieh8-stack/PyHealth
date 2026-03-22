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
