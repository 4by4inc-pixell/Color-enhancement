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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, -1, self.num_heads * self.head_dim)
        out = self.proj(out)
        return out.permute(0, 2, 1).view(b, -1, h, w)

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
        self.process_y = nn.Conv2d(1, filters, 3, padding=1)
        self.process_cb = nn.Conv2d(1, filters, 3, padding=1)
        self.process_cr = nn.Conv2d(1, filters, 3, padding=1)

        self.denoise_cb = Denoiser(filters // 2)
        self.denoise_cr = Denoiser(filters // 2)

        self.pool = lambda x: F.interpolate(x, size=(32, 32), mode='bilinear')
        self.mhsa = MultiHeadSelfAttention(embed_dim=filters, num_heads=4)

        self.lum_conv = nn.Conv2d(filters, filters, 1)
        self.ref_conv = nn.Conv2d(filters * 2, filters, 1)
        self.fuse = MSEFBlock(filters)
        self.concat = nn.Conv2d(filters * 2, filters, 3, padding=1)
        self.out_conv = nn.Conv2d(filters, 3, 3, padding=1)

    def _rgb_to_ycbcr(self, x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        return torch.stack([y, cb, cr], dim=1)

    def forward(self, x):
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        
        out1 = self._forward_single(x1)
        out2 = self._forward_single(x2)
        out3 = self._forward_single(x3)
        
        return [out1, out2, out3]
    
    def _forward_single(self, x):
        residual = x
        ycbcr = self._rgb_to_ycbcr(x)
        y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]

        cb = self.denoise_cb(cb) + cb
        cr = self.denoise_cr(cr) + cr

        y_feat = F.relu(self.process_y(y))
        cb_feat = F.relu(self.process_cb(cb))
        cr_feat = F.relu(self.process_cr(cr))

        ref = torch.cat([cb_feat, cr_feat], dim=1)
        ref = self.ref_conv(ref)

        context = self.pool(y_feat)
        context = self.mhsa(context)
        context = F.interpolate(context, size=y_feat.shape[2:], mode='bilinear', align_corners=False)

        ref = ref + 0.2 * self.lum_conv(context)
        ref = self.fuse(ref)
        fused = self.concat(torch.cat([ref, y_feat], dim=1))
        output = torch.sigmoid(self.out_conv(fused) + residual)

        return output
