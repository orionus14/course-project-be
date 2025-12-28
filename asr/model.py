import torch
import torch.nn as nn


class ASRModel(nn.Module):
    def __init__(self, num_classes):
        super(ASRModel, self).__init__()
        # CNN блок
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        # BiLSTM
        self.lstm = nn.LSTM(input_size=64*80, hidden_size=256,
                            num_layers=2, bidirectional=True, batch_first=True)
        # Fully connected для CTC
        self.fc = nn.Linear(512, num_classes)  # 256*2 = 512

    def forward(self, x):
        # x: batch x time x mel
        x = x.unsqueeze(1)  # batch x 1 x time x mel
        x = self.cnn(x)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c*f)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
