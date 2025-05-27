import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.norm = LayerNormalization(filters)
        self.depthwise = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se = SEBlock(filters)

    def _forward(self, x):
        x_norm = self.norm(x)
        x1 = self.depthwise(x_norm)
        x2 = self.se(x_norm)
        return x + x1 * x2

    def forward(self, x):
        if x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 4, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x):
        B, T, C, H, W = x.size()
        h, c = (torch.zeros(B, C, H, W, device=x.device),
                torch.zeros(B, C, H, W, device=x.device))
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h)
        return outputs[T // 2]

class FeatureExtractor(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.process_y = nn.Conv2d(1, filters, 3, padding=1)
        self.process_cb = nn.Conv2d(1, filters, 3, padding=1)
        self.process_cr = nn.Conv2d(1, filters, 3, padding=1)
        self.denoise_cb = Denoiser(filters // 2)
        self.denoise_cr = Denoiser(filters // 2)

    def forward(self, x):
        ycbcr = self._rgb_to_ycbcr(x)
        y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]

        cb = self.denoise_cb(cb) + cb
        cr = self.denoise_cr(cr) + cr

        y_feat = F.relu(self.process_y(y))
        cb_feat = F.relu(self.process_cb(cb))
        cr_feat = F.relu(self.process_cr(cr))
        ref = torch.cat([cb_feat, cr_feat], dim=1)

        return y_feat, ref

    def _rgb_to_ycbcr(self, x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        return torch.stack([y, cb, cr], dim=1)

class Denoiser(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU()
        )
        self.bottleneck = nn.Conv2d(channels, channels, 1)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(),
        )
        self.out_conv = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.bottleneck(x1)
        x3 = self.decoder(x2)
        x3 = F.interpolate(x3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return torch.tanh(self.out_conv(x3))

class ColEn(nn.Module):
    def __init__(self, filters=32):
        super().__init__()
        self.extractor = FeatureExtractor(filters)
        self.lum_conv = nn.Conv2d(filters, filters * 2, 1)
        self.ref_conv = nn.Conv2d(filters * 2, filters, 1)
        self.fuse = MSEFBlock(filters * 2)
        self.concat = nn.Conv2d(filters * 3, filters, 3, padding=1)
        self.out_conv = nn.Conv2d(filters, 3, 3, padding=1)

        self.temporal_conv_lstm_y = ConvLSTMBlock(filters, filters)
        self.temporal_conv_lstm_ref = ConvLSTMBlock(filters * 2, filters * 2)

        self.pool = lambda x: F.interpolate(x, size=(32, 32), mode='bilinear')

    def forward(self, x):
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        y1, ref1 = self.extractor(x1)
        y2, ref2 = self.extractor(x2)
        y3, ref3 = self.extractor(x3)

        y_seq = torch.stack([y1, y2, y3], dim=1)
        ref_seq = torch.stack([ref1, ref2, ref3], dim=1)

        y_lstm = self.temporal_conv_lstm_y(y_seq)
        ref_lstm = self.temporal_conv_lstm_ref(ref_seq)

        context = self.pool(y_lstm)
        context = F.interpolate(context, size=y_lstm.shape[2:], mode='bilinear', align_corners=False)

        ref = ref_lstm + 0.2 * self.lum_conv(context)
        ref = self.fuse(ref)
        fused = self.concat(torch.cat([ref, y_lstm], dim=1))
        output = self.out_conv(fused)
        output = torch.sigmoid(output)
        output = torch.clamp(output, 0, 1)
        return [output]
