import os
import argparse
import torch
import torch.nn as nn

from model import EnhanceUNet, load_state_dict_compat

def load_ckpt_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt

class EnhanceUNetGlobalTileWrapper(nn.Module):
    def __init__(self, core: EnhanceUNet):
        super().__init__()
        self.core = core

    def _predict_params_from_global(self, x_global):
        s1, s2, s3, s4, m = self.core.encode(x_global)
        feat = self.core.head_feat_from_encoded(s1, s2, s3, s4, m)
        chroma_params = self.core.chroma_head(feat)
        tone_params = self.core.tone_head(feat)
        return chroma_params, tone_params

    def forward(self, global_input, tile_input):
        chroma_params, tone_params = self._predict_params_from_global(global_input)
        y = self.core.forward_with_params(tile_input, chroma_params=chroma_params, tone_params=tone_params)
        return y

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--onnx_out", type=str, required=True)

    p.add_argument("--base", type=int, default=48)
    p.add_argument("--residual_scale", type=float, default=0.10)
    p.add_argument("--head_from", type=str, default="mid", choices=["mid", "s4", "s3", "s2", "s1"])

    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dynamic", action="store_true")

    p.add_argument("--dummy_global_h", type=int, default=512)
    p.add_argument("--dummy_global_w", type=int, default=512)
    p.add_argument("--dummy_tile_h", type=int, default=512)
    p.add_argument("--dummy_tile_w", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()

    core = EnhanceUNet(base=int(args.base), residual_scale=float(args.residual_scale), head_from=str(args.head_from))
    state = load_ckpt_state(args.ckpt)
    load_state_dict_compat(core, {"model": state} if isinstance(state, dict) else state, strict=True)
    core.eval()

    model = EnhanceUNetGlobalTileWrapper(core).eval()

    dummy_global = torch.randn(int(args.batch), 3, int(args.dummy_global_h), int(args.dummy_global_w), dtype=torch.float32)
    dummy_tile = torch.randn(int(args.batch), 3, int(args.dummy_tile_h), int(args.dummy_tile_w), dtype=torch.float32)

    input_names = ["global_input", "tile_input"]
    output_names = ["output"]

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "global_input": {0: "batch", 2: "global_h", 3: "global_w"},
            "tile_input": {0: "batch", 2: "tile_h", 3: "tile_w"},
            "output": {0: "batch", 2: "tile_h", 3: "tile_w"},
        }

    os.makedirs(os.path.dirname(args.onnx_out) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_global, dummy_tile),
        args.onnx_out,
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

if __name__ == "__main__":
    main()
