import torch
import torch.onnx
from model import ColEn
import os

def convert_to_onnx(
    model_path, 
    save_path="onnx_models/0710/Color_Enhance_0002_epoch0002.onnx", 
    input_size=(1, 9, 256, 256),
    filters=16
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColEn(filters=filters).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    print(f"ONNX model saved to: {save_path}")

if __name__ == "__main__":
    convert_to_onnx(
        model_path="./saved_train_models/Color_Enhancement_0710/ColorEnhance_epoch0002_valpsnr19.6860.pth",
        save_path="onnx_models/0710/Color_Enhance_0002_epoch0002.onnx",
        input_size=(1, 9, 256, 256),
        filters=16
    )
