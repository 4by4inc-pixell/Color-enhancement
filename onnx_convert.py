import torch
import torch.onnx
import argparse
import os
from model import FusionLYT

class MiddleFrameWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]  

def convert_to_onnx(model_path, onnx_path, input_height=256, input_width=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionLYT().to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    model.eval()

    wrapped_model = MiddleFrameWrapper(model)

    dummy_input = torch.randn(1, 9, input_height, input_width).to(device)  

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TemporalLYT PyTorch model to ONNX.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--onnx_path", type=str, default="onnx_models/Fusion_Enhance.onnx")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    convert_to_onnx(args.model_path, args.onnx_path, args.height, args.width)
