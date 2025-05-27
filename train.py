import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import structural_similarity_index_measure
from torch.utils.tensorboard import SummaryWriter
from model import ColEn
from losses import CombinedLoss
from dataloader import create_dataloaders
import os

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, max_pixel_value=1.0):
    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()

def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_seq = [img.to(device) for img in batch['low']]
            target_seq = [img.to(device) for img in batch['high']]

            input_tensor = torch.cat(input_seq, dim=1)
            target_tensor = target_seq[1]

            pred_seq = model(input_tensor)
            pred = torch.clamp(pred_seq[0], 0, 1)
            target_tensor = torch.clamp(target_tensor, 0, 1)

            psnr = calculate_psnr(pred, target_tensor)
            ssim = calculate_ssim(pred, target_tensor)

            total_psnr += psnr
            total_ssim += ssim
            num_samples += 1

    if num_samples > 0:
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
    else:
        avg_psnr, avg_ssim = 0, 0

    return avg_psnr, avg_ssim

def main():
    train_dirs = {
        'input': './data/Custom_triplet/Train/input',
        'target': './data/Custom_triplet/Train/target'
    }
    val_dirs = {
        'input': './data/Custom_triplet/Val/input',
        'target': './data/Custom_triplet/Val/target'
    }

    learning_rate = 2e-4
    num_epochs = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LR: {learning_rate}, Epochs: {num_epochs}')

    train_loader, _, val_loader = create_dataloaders(
        train_low=train_dirs['input'],
        train_high=train_dirs['target'],
        val_low=val_dirs['input'],
        val_high=val_dirs['target'],
        crop_size=256,
        batch_size=1
    )

    print(f'Train loader: {len(train_loader)}, Val loader: {len(val_loader)}')

    model = ColEn().to(device)
    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler()

    best_val_psnr = 0.0
    writer = SummaryWriter(log_dir="logs/Color_Enhancement")
    print('Training started.')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i, batch in enumerate(train_loader):
            input_seq = [img.to(device) for img in batch['low']]
            target_seq = [img.to(device) for img in batch['high']]

            input_tensor = torch.cat(input_seq, dim=1)

            optimizer.zero_grad()

            output_seq = model(input_tensor)
            loss = criterion(target_seq, output_seq, input_seq)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if i % 10 == 0:
                writer.add_scalar("Loss/Train", loss.item(), epoch * len(train_loader) + i)

        train_psnr, train_ssim = validate(model, train_loader, device)
        val_psnr, val_ssim = validate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  ▶ Train PSNR: {train_psnr:.4f}, SSIM: {train_ssim:.4f}")
        print(f"  ▶ Val   PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")

        writer.add_scalar("PSNR/Train", train_psnr, epoch)
        writer.add_scalar("SSIM/Train", train_ssim, epoch)
        writer.add_scalar("PSNR/Val", val_psnr, epoch)
        writer.add_scalar("SSIM/Val", val_ssim, epoch)
        writer.add_scalar("Loss/Epoch", train_loss / len(train_loader), epoch)

        scheduler.step()

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            save_path = f'saved_train_models/Color_Enhancement_0514/ColorEnhance_epoch{epoch + 1:04d}_valpsnr{best_val_psnr:.4f}.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'Saving new best model at epoch {epoch + 1} with Val PSNR: {best_val_psnr:.4f}')

    writer.close()

if __name__ == '__main__':
    main()
