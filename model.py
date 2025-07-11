import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class FastNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels)
    def forward(self, x):
        return self.norm(x)

class FastDenoiser(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1)
        )
    def forward(self, x):
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        y = self.pool(x)
        y = torch.sigmoid(self.conv(y))
        return x * y

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.norm = FastNorm(filters)
        self.depthwise = DepthwiseSeparableConv(filters, filters, kernel_size=3, padding=1)
        self.se = SEBlock(filters)
    def forward(self, x):
        x_norm = self.norm(x)
        x1 = self.depthwise(x_norm)
        x2 = self.se(x_norm)
        return x + x1 * x2

class TemporalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = DepthwiseSeparableConv(channels, channels, kernel_size=kernel_size, padding=padding)
    def forward(self, x):
        outputs = []
        for t in range(x.shape[1]):
            outputs.append(self.conv(x[:, t]))
        mid_idx = x.shape[1] // 2
        return outputs[mid_idx]

class FeatureExtractor(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.y_conv = DepthwiseSeparableConv(1, filters, kernel_size=3, padding=1)
        self.cbcr_conv = DepthwiseSeparableConv(2, filters * 2, kernel_size=3, padding=1)
        self.denoise_cb = FastDenoiser(filters // 2)
        self.denoise_cr = FastDenoiser(filters // 2)
    def forward(self, x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y  = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        cb = self.denoise_cb(cb) + cb
        cr = self.denoise_cr(cr) + cr
        cbcr = torch.cat([cb, cr], dim=1)
        y_feat = F.relu(self.y_conv(y))
        cbcr_feat = F.relu(self.cbcr_conv(cbcr))
        return y_feat, cbcr_feat

class ColEn(nn.Module):
    def __init__(self, filters=16):
        super().__init__()
        self.extractor = FeatureExtractor(filters)
        self.lum_conv = DepthwiseSeparableConv(filters, filters * 2, kernel_size=1, padding=0)
        self.fuse = MSEFBlock(filters * 2)
        self.concat = DepthwiseSeparableConv(filters * 3, filters, kernel_size=3, padding=1)
        self.out_conv = DepthwiseSeparableConv(filters, 3, kernel_size=3, padding=1)
        self.temporal_y = TemporalConvBlock(filters)
        self.temporal_ref = TemporalConvBlock(filters * 2)
        self.pool = lambda x: F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)

    def forward(self, x):
        B, C, H, W = x.shape
        if not torch.onnx.is_in_onnx_export():
            assert C == 9, "Input must be 3 frames concatenated, [B,9,H,W]"
        x1, x2, x3 = x[:, :3], x[:, 3:6], x[:, 6:]
        y1, ref1 = self.extractor(x1)
        y2, ref2 = self.extractor(x2)
        y3, ref3 = self.extractor(x3)
        y_seq   = torch.stack([y1, y2, y3], dim=1)
        ref_seq = torch.stack([ref1, ref2, ref3], dim=1)
        y_temp   = self.temporal_y(y_seq)
        ref_temp = self.temporal_ref(ref_seq)
        context = self.pool(y_temp)
        context = F.interpolate(context, size=y_temp.shape[2:], mode='bilinear', align_corners=False)
        ref = ref_temp + 0.2 * self.lum_conv(context)
        ref = self.fuse(ref)
        fused = self.concat(torch.cat([ref, y_temp], dim=1))
        output = self.out_conv(fused)
        output = torch.sigmoid(output)
        output = torch.clamp(output, 0, 1)
        return [output]

if __name__ == "__main__":
    model = ColEn(filters=16)
    dummy = torch.randn(1, 9, 256, 256)
    with torch.no_grad():
        out = model(dummy)
    print("Output shape:", out[0].shape)
    
