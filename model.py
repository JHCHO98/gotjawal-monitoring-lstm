import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class GotjawalConvLSTM(nn.Module):
    def __init__(self, in_channels=7, hidden_channels=16):
        super().__init__()
        self.conv_lstm = ConvLSTMCell(in_channels, hidden_channels, kernel_size=3)
        self.decoder = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Time, Channel, H, W)
        b, t, _, h, w = x.size()
        h_t = torch.zeros(b, 16, h, w).to(x.device)
        c_t = torch.zeros(b, 16, h, w).to(x.device)

        for i in range(t):
            h_t, c_t = self.conv_lstm(x[:, i], (h_t, c_t))

        return torch.sigmoid(self.decoder(h_t))