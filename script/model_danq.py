"""DanQ (Quang & Xie, 2016) adapted to 101 bp ChIP-seq sequences.
Conv1d -> ReLU -> MaxPool -> Dropout(0.2) -> BiLSTM -> Dropout(0.5) -> Dense(925) -> ReLU -> output.
Differences from the original: 101 bp input (not 1000), 2 output logits (not 919 sigmoids).
Input: (batch, 4, seq_len) one-hot, order A,C,G,T.  Output: logits (batch, 2)."""
import torch
import torch.nn as nn


class DanQ(nn.Module):
    def __init__(self, seq_len=101, n_filters=320, kernel=26, pool=13,
                 lstm_hidden=320, dense=925, n_classes=2, p_conv=0.2, p_fc=0.5):
        super().__init__()
        self.conv = nn.Conv1d(4, n_filters, kernel_size=kernel)
        self.pool = nn.MaxPool1d(pool, stride=pool)
        self.drop1 = nn.Dropout(p_conv)
        self.lstm = nn.LSTM(n_filters, lstm_hidden, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(p_fc)
        steps = (seq_len - kernel + 1 - pool) // pool + 1
        if steps < 1:
            raise ValueError('seq_len too short for this kernel/pool combination')
        self.fc1 = nn.Linear(steps * 2 * lstm_hidden, dense)
        self.fc2 = nn.Linear(dense, n_classes)

    def forward(self, x):
        x = self.drop1(self.pool(torch.relu(self.conv(x))))
        x, _ = self.lstm(x.transpose(1, 2))
        x = self.drop2(x).flatten(1)
        return self.fc2(torch.relu(self.fc1(x)))
