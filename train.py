import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from dataset import PairedFolderDataset
from model import EnhanceUNet
from losses import CharbonnierLoss, VGGPerceptualLoss, YCbCrChromaLoss
from utils import set_seed, ensure_dir, psnr, ssim, tensor_to_pil

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_input_dir", type=str, default="./train_datasets/MiT/train/input/")
    p.add_argument("--train_target_dir", type=str, default="./train_datasets/MiT/train/target/")
    p.add_argument("--val_input_dir", type=str, default="./train_datasets/MiT/val/input/")
    p.add_argument("--val_target_dir", type=str, default="./train_datasets/MiT/val/target/")
    p.add_argument("--train_names_txt", type=str, default="")
    p.add_argument("--val_names_txt", type=str, default="")

    p.add_argument("--save_dir", type=str, default="runs/mit_color")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--base", type=int, default=48)

    p.add_argument("--lambda_pix", type=float, default=1.0)
    p.add_argument("--lambda_perc", type=float, default=0.12)
    p.add_argument("--lambda_chroma", type=float, default=0.12)

    p.add_argument("--resume", type=str, default="")
    p.add_argument("--val_vis_n", type=int, default=10)
    p.add_argument("--tb_log_dir", type=str, default="")

    p.add_argument("--train_crop", type=int, default=512)
    p.add_argument("--val_resize", type=int, default=0)

    p.add_argument("--pad_stride", type=int, default=16)
    p.add_argument("--pad_mode", type=str, default="replicate", choices=["reflect", "replicate", "constant"])
    p.add_argument("--pad_value", type=float, default=0.0)

    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--save_best_as_latest", action="store_true")

    p.add_argument("--vgg_max_side", type=int, default=256)

    p.add_argument("--residual_scale", type=float, default=0.30)
    p.add_argument("--head_from", type=str, default="mid", choices=["mid", "s4", "s3", "s2", "s1"])
    return p.parse_args()

def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def forward_pad_to_stride(model, x, stride=16, pad_mode="reflect", pad_value=0.0, use_amp=False):
    b, c, h, w = x.shape
    ph = (stride - (h % stride)) % stride
    pw = (stride - (w % stride)) % stride
    if ph != 0 or pw != 0:
        pad = (0, pw, 0, ph)
        if pad_mode == "constant":
            x_pad = F.pad(x, pad, mode=pad_mode, value=pad_value)
        else:
            x_pad = F.pad(x, pad, mode=pad_mode)
    else:
        x_pad = x

    device = "cuda" if x_pad.is_cuda else "cpu"
    autocast_ctx = torch.amp.autocast(device_type=device, enabled=use_amp and device == "cuda") if device == "cuda" else torch.amp.autocast(device_type="cpu", enabled=False)
    with autocast_ctx:
        y_pad = model(x_pad)

    y = y_pad[:, :, :h, :w]
    return y

@torch.no_grad()
def validate(model, loader, device, use_amp: bool, pad_stride: int, pad_mode: str, pad_value: float):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0
    for x, y, _ in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = forward_pad_to_stride(
            model=model,
            x=x,
            stride=pad_stride,
            pad_mode=pad_mode,
            pad_value=pad_value,
            use_amp=use_amp and device.startswith("cuda"),
        )

        total_psnr += psnr(pred[0], y[0])
        total_ssim += ssim(pred, y)
        n += 1
    return total_psnr / max(n, 1), total_ssim / max(n, 1)

@torch.no_grad()
def build_val_grid_tensor(model, loader, device, use_amp: bool, take=10, pad_stride=16, pad_mode="reflect", pad_value=0.0):
    model.eval()
    imgs = []
    count = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        pred = forward_pad_to_stride(
            model=model,
            x=x,
            stride=pad_stride,
            pad_mode=pad_mode,
            pad_value=pad_value,
            use_amp=use_amp and device.startswith("cuda"),
        )

        imgs.append(x[0].detach().float().clamp(0, 1).cpu())
        imgs.append(pred[0].detach().float().clamp(0, 1).cpu())
        imgs.append(y[0].detach().float().clamp(0, 1).cpu())

        count += 1
        if count >= take:
            break

    max_h = max(t.shape[1] for t in imgs)
    max_w = max(t.shape[2] for t in imgs)

    padded = []
    for t in imgs:
        _, h, w = t.shape
        pad_h = max_h - h
        pad_w = max_w - w
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        padded.append(t)

    grid = make_grid(padded, nrow=3, padding=6)
    return grid

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.save_dir)
    ensure_dir(os.path.join(args.save_dir, "val_vis"))

    if not args.tb_log_dir:
        args.tb_log_dir = os.path.join(args.save_dir, "tb_logs")
    ensure_dir(args.tb_log_dir)
    writer = SummaryWriter(log_dir=args.tb_log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = PairedFolderDataset(
        input_dir=args.train_input_dir,
        target_dir=args.train_target_dir,
        names_txt=args.train_names_txt,
        train=True,
        align_sizes=True,
        train_crop=args.train_crop,
        val_resize=0,
    )
    val_ds = PairedFolderDataset(
        input_dir=args.val_input_dir,
        target_dir=args.val_target_dir,
        names_txt=args.val_names_txt,
        train=False,
        align_sizes=True,
        train_crop=0,
        val_resize=args.val_resize,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    model = EnhanceUNet(base=args.base, residual_scale=args.residual_scale, head_from=args.head_from).to(device)

    if device.startswith("cuda"):
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            model = nn.DataParallel(model)
            print(f"DataParallel enabled: {n_gpus} GPUs")

    opt = torch.optim.AdamW(unwrap_model(model).parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    pix_loss = CharbonnierLoss()
    perc_loss = VGGPerceptualLoss(max_vgg_side=args.vgg_max_side).to(device)
    chroma_loss = YCbCrChromaLoss(loss="l1")

    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.startswith("cuda")))

    start_epoch = 0
    best_psnr = -1.0
    global_step = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("best_psnr", -1.0)
        global_step = ckpt.get("global_step", 0)

    writer.add_text("run/config", str(vars(args)), 0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        run_pix = 0.0
        run_perc = 0.0
        run_chr = 0.0

        pbar = tqdm(train_loader, desc=f"train {epoch+1}/{args.epochs}", leave=False)
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.startswith("cuda"))) if device.startswith("cuda") else torch.amp.autocast(device_type="cpu", enabled=False)

            try:
                with autocast_ctx:
                    pred = model(x)
                    l_pix = pix_loss(pred, y) * args.lambda_pix

                l_perc = perc_loss(pred, y) * args.lambda_perc
                l_chr = chroma_loss(pred, y) * args.lambda_chroma
                loss = l_pix + l_perc + l_chr

                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=args.grad_clip_norm)

                scaler.step(opt)
                scaler.update()

            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            run_loss += float(loss.item())
            run_pix += float(l_pix.item())
            run_perc += float(l_perc.item())
            run_chr += float(l_chr.item())

            lr = opt.param_groups[0]["lr"]
            writer.add_scalar("train/loss_total_step", float(loss.item()), global_step)
            writer.add_scalar("train/loss_pix_step", float(l_pix.item()), global_step)
            writer.add_scalar("train/loss_perc_step", float(l_perc.item()), global_step)
            writer.add_scalar("train/loss_chroma_step", float(l_chr.item()), global_step)
            writer.add_scalar("train/lr_step", float(lr), global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
            global_step += 1

        sched.step()

        steps = max(len(train_loader), 1)
        avg_loss = run_loss / steps
        avg_pix = run_pix / steps
        avg_perc = run_perc / steps
        avg_chr = run_chr / steps
        lr = opt.param_groups[0]["lr"]

        v_psnr, v_ssim = validate(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=(args.amp and device.startswith("cuda")),
            pad_stride=args.pad_stride,
            pad_mode=args.pad_mode,
            pad_value=args.pad_value,
        )

        is_best = (v_psnr > best_psnr) and (not (torch.isnan(torch.tensor(v_psnr)) or torch.isnan(torch.tensor(v_ssim))))
        if is_best:
            best_psnr = v_psnr

        grid = build_val_grid_tensor(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=(args.amp and device.startswith("cuda")),
            take=args.val_vis_n,
            pad_stride=args.pad_stride,
            pad_mode=args.pad_mode,
            pad_value=args.pad_value,
        )
        vis_path = os.path.join(args.save_dir, "val_vis", f"epoch_{epoch+1:03d}_psnr_{v_psnr:.2f}_ssim_{v_ssim:.4f}.jpg")
        tensor_to_pil(grid).save(vis_path)

        writer.add_scalar("epoch/train_loss_total", float(avg_loss), epoch + 1)
        writer.add_scalar("epoch/train_loss_pix", float(avg_pix), epoch + 1)
        writer.add_scalar("epoch/train_loss_perc", float(avg_perc), epoch + 1)
        writer.add_scalar("epoch/train_loss_chroma", float(avg_chr), epoch + 1)
        writer.add_scalar("epoch/lr", float(lr), epoch + 1)
        writer.add_scalar("epoch/val_psnr", float(v_psnr), epoch + 1)
        writer.add_scalar("epoch/val_ssim", float(v_ssim), epoch + 1)
        writer.add_image("epoch/val_grid_input_pred_gt", grid, epoch + 1)
        writer.flush()

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": unwrap_model(model).state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": best_psnr,
            "args": vars(args),
        }

        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        if is_best:
            best_name = f"best_epoch_{epoch+1:04d}_psnr_{v_psnr:.3f}_ssim_{v_ssim:.4f}.pt"
            torch.save(ckpt, os.path.join(args.save_dir, best_name))
            if args.save_best_as_latest:
                torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))

        dt = time.time() - t0
        print(
            f"epoch {epoch+1}/{args.epochs}  loss {avg_loss:.5f}  val_psnr {v_psnr:.3f}  val_ssim {v_ssim:.4f}  lr {lr:.6e}  time {dt:.1f}s  vis {vis_path}  tb {args.tb_log_dir}"
        )

    writer.close()

if __name__ == "__main__":
    main()
