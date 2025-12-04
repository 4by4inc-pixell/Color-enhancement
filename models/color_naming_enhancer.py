import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, base_ch)
        self.conv2 = ConvBlock(base_ch, base_ch * 2, s=2)
        self.conv3 = ConvBlock(base_ch * 2, base_ch * 4, s=2)
        self.conv4 = ConvBlock(base_ch * 4, base_ch * 8, s=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x1, x2, x3, x4

class Decoder(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.dec4_3 = ConvBlock(base_ch * 12, base_ch * 4)
        self.dec3_2 = ConvBlock(base_ch * 6, base_ch * 2)
        self.dec2_1 = ConvBlock(base_ch * 3, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        x = F.interpolate(x4, size=x3.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.dec4_3(x)
        x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3_2(x)
        x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.dec2_1(x)
        x = self.out_conv(x)
        return x

class ColorNamingToneCurveHead(nn.Module):
    def __init__(self, in_feat_dim, num_colors=8, hidden_dim=256, num_hinges=4):
        super().__init__()
        self.num_colors = num_colors
        self.num_hinges = num_hinges
        self.fc = nn.Sequential(
            nn.Linear(in_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_slope = nn.Linear(hidden_dim, num_colors * 3)
        self.fc_bias = nn.Linear(hidden_dim, num_colors * 3)
        self.fc_hinge = nn.Linear(hidden_dim, num_colors * 3 * num_hinges)
        self.color_conv = nn.Conv2d(3, num_colors, kernel_size=1)

        hinges = torch.tensor([0.25, 0.5, 0.75, 0.9], dtype=torch.float32)
        if num_hinges != 4:
            hinges = torch.linspace(0.2, 0.9, steps=num_hinges)
        self.register_buffer("hinge_points", hinges.view(1, 1, num_hinges, 1, 1, 1))

    def compute_params(self, feat):
        b, c, _, _ = feat.shape
        pooled = F.adaptive_avg_pool2d(feat, 1).view(b, c)
        h_feat = self.fc(pooled)
        slope_raw = self.fc_slope(h_feat).view(b, self.num_colors, 3, 1, 1)
        bias_raw = self.fc_bias(h_feat).view(b, self.num_colors, 3, 1, 1)
        hinge_raw = self.fc_hinge(h_feat).view(
            b, self.num_colors, self.num_hinges, 3, 1, 1
        )

        slope = F.softplus(slope_raw) + 1e-4
        bias = torch.tanh(bias_raw) * 0.25
        hinge_w = F.softplus(hinge_raw)
        return slope, bias, hinge_w

    def forward(self, x_rgb, feat=None, params=None):
        b, _, h, w = x_rgb.shape
        k = self.num_colors

        color_logits = self.color_conv(x_rgb)
        color_prob = torch.softmax(color_logits, dim=1)

        if params is None:
            assert feat is not None, "feat or params must be provided."
            slope, bias, hinge_w = self.compute_params(feat)
        else:
            slope, bias, hinge_w = params

        x_exp = x_rgb.unsqueeze(1)                          
        x_exp_clamped = torch.clamp(x_exp, 1e-6, 1.0)

        slope_b = slope.expand(-1, -1, -1, h, w)
        bias_b = bias.expand(-1, -1, -1, h, w)
        y_k = slope_b * x_exp_clamped + bias_b

        hinge_points = self.hinge_points                    
        x_hp = x_exp_clamped.unsqueeze(2)                   
        relu_terms = F.relu(x_hp - hinge_points)

        hinge_w_b = hinge_w.expand(-1, -1, -1, -1, h, w)
        y_k = y_k + (hinge_w_b * relu_terms).sum(dim=2)

        color_weight = color_prob.unsqueeze(2)              
        y = (y_k * color_weight).sum(dim=1)                 
        return torch.clamp(y, 0.0, 1.0)
    
class GlobalLocalColorNamingEnhancer(nn.Module):
    def __init__(self, base_ch=32, num_colors=8):
        super().__init__()
        self.encoder = Encoder(in_ch=3, base_ch=base_ch)
        self.decoder = Decoder(base_ch=base_ch)
        self.global_head = ColorNamingToneCurveHead(
            in_feat_dim=base_ch * 8,
            num_colors=num_colors,
            hidden_dim=256,
            num_hinges=4,
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        global_out = self.global_head(x_rgb=x, feat=x4)  
        local_res = self.decoder(x1, x2, x3, x4)
        local_res = F.interpolate(local_res, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = global_out + local_res
        out = torch.clamp(out, 0.0, 1.0)
        return out