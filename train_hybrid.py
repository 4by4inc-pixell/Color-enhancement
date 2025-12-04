import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import HybridConfig, StageConfig, hybrid_cfg
from datasets.enhancement_dataset import EnhancementDataset, MultiExposureFiveKDataset
from models.color_naming_enhancer import GlobalLocalColorNamingEnhancer
from utils.transforms import PairTransform
from utils.metrics import EnhancedColorLoss, psnr, ssim
from utils.logger import create_logger

def get_dataloaders(stage_cfg: StageConfig, img_size: int):
    transform = PairTransform(img_size=img_size)

    if stage_cfg.kind == "public":
        train_ds = MultiExposureFiveKDataset(
            input_dir=stage_cfg.train_input_dir,
            gt_dir=stage_cfg.train_target_dir,
            transform=transform,
        )
        val_ds = MultiExposureFiveKDataset(
            input_dir=stage_cfg.val_input_dir,
            gt_dir=stage_cfg.val_target_dir,
            transform=transform,
        )
    else:
        train_ds = EnhancementDataset(
            input_dir=stage_cfg.train_input_dir,
            target_dir=stage_cfg.train_target_dir,
            transform=transform,
        )
        val_ds = EnhancementDataset(
            input_dir=stage_cfg.val_input_dir,
            target_dir=stage_cfg.val_target_dir,
            transform=transform,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=stage_cfg.batch_size,
        shuffle=True,
        num_workers=stage_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        num_workers=stage_cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, logger, stage_name):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    pbar = tqdm(loader, desc=f"{stage_name} Train Epoch {epoch}")
    for step, batch in enumerate(pbar):
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)

        optimizer.zero_grad()
        pred = model(inp)
        loss = loss_fn(pred, tgt, inp)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            psnr_val = psnr(pred, tgt)
            ssim_val = ssim(pred, tgt)

        running_loss += loss.item()
        running_psnr += psnr_val.item()
        running_ssim += ssim_val.item()

        pbar.set_postfix(loss=loss.item(), psnr=psnr_val.item(), ssim=ssim_val.item())

        global_step = epoch * len(loader) + step
        logger.add_scalar(f"{stage_name}/step/train_loss", loss.item(), global_step)
        logger.add_scalar(f"{stage_name}/step/train_psnr", psnr_val.item(), global_step)
        logger.add_scalar(f"{stage_name}/step/train_ssim", ssim_val.item(), global_step)

    avg_loss = running_loss / len(loader)
    avg_psnr = running_psnr / len(loader)
    avg_ssim = running_ssim / len(loader)

    logger.add_scalar(f"{stage_name}/epoch/train_loss", avg_loss, epoch)
    logger.add_scalar(f"{stage_name}/epoch/train_psnr", avg_psnr, epoch)
    logger.add_scalar(f"{stage_name}/epoch/train_ssim", avg_ssim, epoch)

    return avg_loss, avg_psnr, avg_ssim

@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch, logger, stage_name):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    pbar = tqdm(loader, desc=f"{stage_name} Val Epoch {epoch}")
    for step, batch in enumerate(pbar):
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)

        pred = model(inp)
        loss = loss_fn(pred, tgt, inp)
        psnr_val = psnr(pred, tgt)
        ssim_val = ssim(pred, tgt)

        running_loss += loss.item()
        running_psnr += psnr_val.item()
        running_ssim += ssim_val.item()

        pbar.set_postfix(loss=loss.item(), psnr=psnr_val.item(), ssim=ssim_val.item())

        global_step = epoch * len(loader) + step
        logger.add_scalar(f"{stage_name}/step/val_loss", loss.item(), global_step)
        logger.add_scalar(f"{stage_name}/step/val_psnr", psnr_val.item(), global_step)
        logger.add_scalar(f"{stage_name}/step/val_ssim", ssim_val.item(), global_step)

    avg_loss = running_loss / len(loader)
    avg_psnr = running_psnr / len(loader)
    avg_ssim = running_ssim / len(loader)

    logger.add_scalar(f"{stage_name}/epoch/val_loss", avg_loss, epoch)
    logger.add_scalar(f"{stage_name}/epoch/val_psnr", avg_psnr, epoch)
    logger.add_scalar(f"{stage_name}/epoch/val_ssim", avg_ssim, epoch)

    return avg_loss, avg_psnr, avg_ssim

def run_stage(
    stage_name: str,
    stage_cfg: StageConfig,
    cfg: HybridConfig,
    model,
    device,
    ckpt_start_path=None,
):
    os.makedirs(cfg.save_dir, exist_ok=True)
    stage_log_dir = os.path.join(cfg.log_dir, stage_name)
    logger = create_logger(stage_log_dir)

    train_loader, val_loader = get_dataloaders(stage_cfg, cfg.img_size)

    loss_fn = EnhancedColorLoss(
        alpha_l1=1.0,
        alpha_ssim=1.0,
        alpha_perc=0.1,
        alpha_tv=0.01,
        alpha_id=0.1,
        use_perceptual=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=stage_cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    start_epoch = 0
    best_val_psnr = -1e9
    best_ckpt_path = None

    if ckpt_start_path is not None and os.path.isfile(ckpt_start_path):
        ckpt = torch.load(ckpt_start_path, map_location=device, weights_only=True)
        state_dict = ckpt["model"]
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        print(f"[{stage_name}] Loaded checkpoint {ckpt_start_path}")

    for epoch in range(start_epoch, stage_cfg.num_epochs):
        train_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, logger, stage_name
        )

        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, loss_fn, device, epoch, logger, stage_name
        )

        print(
            f"[{stage_name}] Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_psnr={train_psnr:.4f}, train_ssim={train_ssim:.4f}, "
            f"val_loss={val_loss:.4f}, val_psnr={val_psnr:.4f}, val_ssim={val_ssim:.4f}"
        )

        if val_psnr > best_val_psnr:
            prev_psnr = best_val_psnr
            best_val_psnr = val_psnr

            save_path = os.path.join(
                cfg.save_dir,
                f"{stage_name}_best_psnr_epoch_{epoch}.pth",
            )

            if isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            torch.save(
                {
                    "epoch": epoch,
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "best_val_psnr": best_val_psnr,
                    "val_loss": val_loss,
                    "val_ssim": val_ssim,
                },
                save_path,
            )
            best_ckpt_path = save_path

            print(
                f"[{stage_name}] PSNR improved from {prev_psnr:.4f} to {best_val_psnr:.4f}. "
                f"Saved model to {save_path}"
            )

    return best_ckpt_path

def main():
    cfg = hybrid_cfg
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = GlobalLocalColorNamingEnhancer(
        base_ch=cfg.base_channels,
        num_colors=cfg.num_colors,
    )

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    model = model.to(device)

    custom_ckpt = run_stage(
        stage_name="custom",
        stage_cfg=cfg.custom,
        cfg=cfg,
        model=model,
        device=device,
        ckpt_start_path=None,  
    )

    print(f"Custom stage best ckpt: {custom_ckpt}")

if __name__ == "__main__":
    main()
