import torchaudio.transforms as T, torch.nn as nn, torch.nn.functional as F

mfcc_transform = T.MFCC(
    sample_rate=16_000, 
    n_mfcc = 40, 
    melkwargs={
        "n_fft":400, 
        "hop_length":160,
        "n_mels":64
    }
)

class MFCC_CRNN(nn.Module):
    def __init__(self,n_classes=9):
        super().__init__()
        self.mfcc = mfcc_transform

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d((2,2)), # [B, 16, 20, T//2]

            nn.Conv2d(16,32, kernel_size=(3,3), padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d((2,2)),  # [B, 32, 10, T//4]
        )

        self.rnn = nn.GRU(
            input_size= 32*10, 
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*2, 64), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.mfcc(x)            # [B, 40, T]
        x = x.unsqueeze(1)          # [B, 1, 40, T]
        x = self.cnn(x)             # [B, 32, 10, T//4]
        x = x.permute(0, 3, 1, 2)   # [B, T//4, 32, 10]
        x = x.flatten(2)            # [B, T//4, 320]

        out,_ = self.rnn(x)         # [B, T//4, 128]
        out = out.mean(dim =1)      # [B, 128]
        return self.classifier(out) # [B, 9]  
    

if __name__ == '__main__':
    ...
