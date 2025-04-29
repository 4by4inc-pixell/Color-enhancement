import torch
from torchprofile import profile_macs
from model import ColEn

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ColEn().to(device)
    model.eval()

    input_tensor = torch.randn(1, 9, 256, 256).to(device)

    macs = profile_macs(model, input_tensor)

    flops = macs * 2  
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tflops = flops / (1024 ** 3)  

    print(f"Model FLOPs (G): {tflops:.4f} G")
    print(f"Model FLOPs (M): {tflops*1024:.2f} M")
    print(f"Model MACs (G): {macs / (1024 ** 3):.4f} G")
    print(f"Model params (M): {num_params / 1e6:.2f} M")
    print(f"Model params (exact): {num_params}")

if __name__ == "__main__":
    main()
