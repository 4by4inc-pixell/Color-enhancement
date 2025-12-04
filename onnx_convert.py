import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.color_naming_enhancer import GlobalLocalColorNamingEnhancer

class GlobalLocalWrapperSingle(nn.Module):
    def __init__(self, core_model):
        super().__init__()
        self.core = core_model

    def forward(self, full_img, patch):
        x1_full, x2_full, x3_full, x4_full = self.core.encoder(full_img)
        slope, bias, hinge_w = self.core.global_head.compute_params(x4_full)
        x1_p, x2_p, x3_p, x4_p = self.core.encoder(patch)
        params = (slope, bias, hinge_w)
        global_out = self.core.global_head(x_rgb=patch, feat=None, params=params)
        local_res = self.core.decoder(x1_p, x2_p, x3_p, x4_p)
        local_res = F.interpolate(local_res, size=patch.shape[2:], mode="bilinear", align_corners=False)
        out = torch.clamp(global_out + local_res, 0.0, 1.0)
        return out

def load_core_model(ckpt_path, device, base_channels=32, num_colors=8):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = GlobalLocalColorNamingEnhancer(
        base_ch=base_channels,
        num_colors=num_colors,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state[new_k] = v
    model.load_state_dict(new_state)
    model.eval()
    return model, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--num_colors", type=int, default=8)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    core_model, device = load_core_model(
        ckpt_path=args.ckpt,
        device=args.device,
        base_channels=args.base_channels,
        num_colors=args.num_colors,
    )

    wrapper = GlobalLocalWrapperSingle(core_model).to(device)

    dummy_full = torch.randn(1, 3, 512, 512, device=device)
    dummy_patch = torch.randn(1, 3, 512, 512, device=device)
    onnx_path = os.path.join(args.output_dir, "color_enhancer_1204.onnx")

    torch.onnx.export(
        wrapper,
        (dummy_full, dummy_patch),
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["full_img", "patch"],
        output_names=["output"],
        dynamic_axes={
            "full_img": {0: "batch_full", 2: "H_full", 3: "W_full"},
            "patch": {0: "batch_patch", 2: "H_patch", 3: "W_patch"},
            "output": {0: "batch_patch", 2: "H_patch", 3: "W_patch"},
        },
    )

if __name__ == "__main__":
    main()
