import os
import math
import random
import glob
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataset import create_dataloaders
from model import RetinexEnhancer
from losses import enhancement_loss, chroma_std, luma_std, VGGPerceptualLoss  
from utils import calculate_psnr, calculate_ssim
import torch._dynamo
import numpy as np

torch._dynamo.reset()
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.optimize_ddp = False

USE_SYNTH_SEQ = True
T_SYN = 3
MAX_ANGLE = 2.0
MAX_SHIFT = 0.03
MIN_SCALE = 0.98
MAX_SCALE = 1.02
PHOTOMETRIC_DRIFT = True
DRIFT_GAMMA = (0.95, 1.05)
DRIFT_GAIN = (0.97, 1.03)
DRIFT_BIAS = (-0.01, 0.01)
LAMBDA_TEMP_KNOWN = 0.5
LAMBDA_TEMP_SIMPLE = 0.25
WINDOW_SIZE = 3
AUTO_RESUME = True

def _affine_matrix(angle_deg, tx, ty, scale):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return torch.tensor([[scale * ca, -scale * sa, tx],
                         [scale * sa, scale * ca, ty]], dtype=torch.float32)

def _to_3x3(theta_2x3):
    B = theta_2x3.size(0)
    pad = theta_2x3.new_zeros((B, 1, 3))
    pad[:, :, 2] = 1.0
    return torch.cat([theta_2x3, pad], dim=1)

def _inv_2x3(theta_2x3):
    T = _to_3x3(theta_2x3)
    Tinv = torch.inverse(T)
    return Tinv[:, :2, :]

def _apply_photometric_drift(x, gamma_range, gain_range, bias_range):
    B = x.size(0)
    device = x.device
    gammas = torch.empty(B, device=device).uniform_(*gamma_range).view(B, 1, 1, 1)
    gains = torch.empty(B, 3, device=device).uniform_(*gain_range).view(B, 3, 1, 1)
    bias = torch.empty(B, 3, device=device).uniform_(*bias_range).view(B, 3, 1, 1)
    x = torch.clamp(x, 0, 1)
    x = torch.clamp(x.pow(gammas), 0, 1)
    x = torch.clamp(x * gains + bias, 0, 1)
    return x

def build_synthetic_seq(x, y, T=3,
                        max_angle=2.0, max_shift=0.03,
                        min_scale=0.98, max_scale=1.02,
                        photometric_drift=True):
    B, C, H, W = x.shape
    device = x.device
    thetas = []
    x_seq = []
    y_seq = []
    for t in range(T):
        angles = torch.empty(B, device=device).uniform_(-max_angle, max_angle)
        scales = torch.empty(B, device=device).uniform_(min_scale, max_scale)
        txs = torch.empty(B, device=device).uniform_(-max_shift, max_shift)
        tys = torch.empty(B, device=device).uniform_(-max_shift, max_shift)
        theta_bt = []
        for b in range(B):
            th = _affine_matrix(angles[b].item(), txs[b].item(), tys[b].item(), scales[b].item()).to(device)
            theta_bt.append(th)
        theta_bt = torch.stack(theta_bt, dim=0)
        thetas.append(theta_bt)
        grid = F.affine_grid(theta_bt, size=x.size(), align_corners=False)
        x_t = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        y_t = F.grid_sample(y, grid, mode="bilinear", padding_mode="border", align_corners=False)
        if photometric_drift:
            x_t = _apply_photometric_drift(x_t, DRIFT_GAMMA, DRIFT_GAIN, DRIFT_BIAS)
        x_seq.append(x_t)
        y_seq.append(y_t)
    x_seq = torch.stack(x_seq, dim=1)
    y_seq = torch.stack(y_seq, dim=1)
    rel_grids = []
    for t in range(1, T):
        theta_t = thetas[t]
        theta_tm1 = thetas[t - 1]
        Tt = _to_3x3(theta_t)
        Tm1i = torch.inverse(_to_3x3(theta_tm1))
        Trel = torch.bmm(Tt, Tm1i)[:, :2, :]
        grid_rel = F.affine_grid(Trel, size=x.size(), align_corners=False)
        rel_grids.append(grid_rel)
    return x_seq, y_seq, rel_grids

def temporal_loss_by_known_warp(y_seq, rel_grids, weight_l1=1.0, weight_ssim=0.0):
    B, T, C, H, W = y_seq.shape
    if T < 2:
        return y_seq.new_tensor(0.0)
    loss = y_seq.new_tensor(0.0)
    cnt = 0
    for t in range(1, T):
        y_t = y_seq[:, t]
        y_tm1 = y_seq[:, t - 1]
        grid = rel_grids[t - 1]
        warp = F.grid_sample(y_tm1, grid, mode="bilinear", padding_mode="border", align_corners=False)
        loss = loss + weight_l1 * F.l1_loss(y_t, warp)
        if weight_ssim > 0:
            try:
                import torchmetrics
                ssim = 1.0 - torchmetrics.functional.structural_similarity_index_measure(
                    y_t, warp, data_range=1.0, gaussian_kernel=True
                )
            except Exception:
                ssim = (y_t - warp).abs().mean() * 0.0
            loss = loss + weight_ssim * ssim
        cnt += 1
    return loss / max(1, cnt)

def temporal_consistency_loss_simple(y_seq, pool=4):
    B, T, C, H, W = y_seq.shape
    if T < 2:
        return y_seq.new_tensor(0.0)
    loss = y_seq.new_tensor(0.0)
    cnt = 0
    for t in range(1, T):
        ya = F.avg_pool2d(y_seq[:, t - 1], kernel_size=pool, stride=pool)
        yb = F.avg_pool2d(y_seq[:, t], kernel_size=pool, stride=pool)
        loss = loss + F.l1_loss(ya, yb)
        cnt += 1
    return loss / max(1, cnt)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def _save_last_ckpt(path, model, optimizer, scheduler, scaler, ema, epoch, best_psnr, use_dp):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_state = (model.module if use_dp else model).state_dict()
    torch.save({
        "epoch": epoch,
        "best_psnr": best_psnr,
        "model_state": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "ema_shadow": ema.shadow,
        "rng": {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }, path)

def _load_last_ckpt(path, model, optimizer, scheduler, scaler, ema, device, use_dp):
    ckpt = torch.load(path, map_location=device)
    (model.module if use_dp else model).load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    _optimizer_to_device(optimizer, device)
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    if "ema_shadow" in ckpt and isinstance(ckpt["ema_shadow"], dict):
        ema.shadow = ckpt["ema_shadow"]
    try:
        rng = ckpt.get("rng", {})
        if "random" in rng:
            random.setstate(rng["random"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "torch_cuda" in rng and rng["torch_cuda"] is not None and torch.cuda.is_available():
            for i, s in enumerate(rng["torch_cuda"]):
                try:
                    torch.cuda.set_rng_state(s, device=i)
                except Exception:
                    pass
    except Exception:
        pass
    start_epoch = int(ckpt.get("epoch", 0))
    best_psnr = float(ckpt.get("best_psnr", -1e9))
    print(f"[Resume] Loaded last checkpoint from {path} (epoch={start_epoch}, best_psnr={best_psnr:.3f})")
    return start_epoch, best_psnr

def _try_auto_resume(ckpt_dir, model, optimizer, scheduler, scaler, ema, device, use_dp):
    last_path = os.path.join(ckpt_dir, "last.pth")
    if os.path.isfile(last_path):
        return _load_last_ckpt(last_path, model, optimizer, scheduler, scaler, ema, device, use_dp)
    cand = sorted(glob.glob(os.path.join(ckpt_dir, "BEST_PSNR_epoch*.pth")), key=os.path.getmtime)
    if cand:
        best_path = cand[-1]
        state = torch.load(best_path, map_location=device)
        (model.module if use_dp else model).load_state_dict(state, strict=True)
        print(f"[Warmstart] Loaded weights from best checkpoint: {best_path}")
        return 0, -1e9
    print("[Start] No checkpoint found. Training from scratch.")
    return 0, -1e9

def main():
    set_seed(20250822)
    train_in = "./data/LCDP_dataset_+/input"
    train_gt = "./data/LCDP_dataset_+/gt"
    val_in = "./data/LCDP_dataset_+/valid-input"
    val_gt = "./data/LCDP_dataset_+/valid-gt"
    crop = 256
    batch_size = 16
    lr = 2e-4
    epochs = 1000
    max_grad_norm = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device} | #GPUs: {n_gpus} | LR: {lr} | Epochs: {epochs} | SynthSeq={USE_SYNTH_SEQ} (T={T_SYN})")

    train_loader, val_loader = create_dataloaders(
        train_in, train_gt, val_in, val_gt,
        crop_size=crop, batch_size=batch_size, workers=4
    )

    base_model = RetinexEnhancer(window_size=WINDOW_SIZE).to(device)
    use_dp = (n_gpus > 1 and device.type == "cuda")
    model = torch.nn.DataParallel(base_model) if use_dp else base_model

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema = EMA(model.module if use_dp else model, decay=0.998)

    vgg_loss = VGGPerceptualLoss().to(device)

    log_dir = "logs/Color_enhancement_1118"
    writer = SummaryWriter(log_dir=log_dir)
    ckpt_dir = "saved_models/Color_enhancement_1118"
    os.makedirs(ckpt_dir, exist_ok=True)

    if AUTO_RESUME:
        start_epoch, best_psnr = _try_auto_resume(
            ckpt_dir, model, optimizer, scheduler, scaler, ema, device, use_dp
        )
    else:
        start_epoch, best_psnr = 0, -1e9

    for ep in range(start_epoch, epochs):
        model.train()
        running = 0.0
        running_temp_known = 0.0
        running_temp_simple = 0.0

        for i, batch in enumerate(train_loader):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                if USE_SYNTH_SEQ:
                    x_seq, y_seq, rel_grids = build_synthetic_seq(
                        x, y, T=T_SYN, max_angle=MAX_ANGLE, max_shift=MAX_SHIFT,
                        min_scale=MIN_SCALE, max_scale=MAX_SCALE,
                        photometric_drift=PHOTOMETRIC_DRIFT
                    )
                    B, T, C, H, W = x_seq.shape
                    preds = []
                    per_losses = []
                    half_w = WINDOW_SIZE // 2
                    for t in range(T):
                        idxs = []
                        for dt in range(-half_w, half_w + 1):
                            tt = min(max(t + dt, 0), T - 1)
                            idxs.append(tt)
                        x_window = torch.cat([x_seq[:, ti] for ti in idxs], dim=1)
                        pred_t, extras_t, _ = model(x_window)
                        preds.append(pred_t)
                        per_losses.append(
                            enhancement_loss(
                                pred_t,
                                y_seq[:, t],
                                extras_t,
                                x.device,
                                x_input=x_seq[:, t],
                                vgg_loss=vgg_loss,  
                            )
                        )
                    y_pred_seq = torch.stack(preds, dim=1)
                    per_frame = torch.stack(per_losses).mean()
                    L_temp_known = temporal_loss_by_known_warp(y_pred_seq, rel_grids, weight_l1=1.0, weight_ssim=0.0)
                    L_temp_simple = temporal_consistency_loss_simple(y_pred_seq, pool=4)
                    loss = per_frame + LAMBDA_TEMP_KNOWN * L_temp_known + LAMBDA_TEMP_SIMPLE * L_temp_simple
                    running_temp_known += float(L_temp_known.item())
                    running_temp_simple += float(L_temp_simple.item())
                else:
                    x_window = torch.cat([x for _ in range(WINDOW_SIZE)], dim=1)
                    pred, extras, _ = model(x_window)
                    loss = enhancement_loss(
                        pred,
                        y,
                        extras,
                        x.device,
                        x_input=x,
                        vgg_loss=vgg_loss, 
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model.module if use_dp else model)
            running += float(loss.item())

            if i % 20 == 0:
                global_step = ep * len(train_loader) + i
                writer.add_scalar("Loss/TrainIter", float(loss.item()), global_step)
                if USE_SYNTH_SEQ:
                    writer.add_scalar("Loss/TempKnownIter", float(L_temp_known.item()), global_step)
                    writer.add_scalar("Loss/TempSimpleIter", float(L_temp_simple.item()), global_step)

        eval_model = model.module if use_dp else model
        backup = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
        ema.copy_to(eval_model)
        model.eval()

        psnr_vals, ssim_vals, dchroma_vals, dluma_vals = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True)
                x_window = torch.cat([x for _ in range(WINDOW_SIZE)], dim=1)
                pred, _, _ = model(x_window)
                pred = torch.clamp(pred, 0, 1)
                y = torch.clamp(y, 0, 1)
                ps = calculate_psnr(pred, y)
                ss = calculate_ssim(pred, y)
                dc = float((chroma_std(pred) - chroma_std(y)).mean().item())
                dl = float((luma_std(pred) - luma_std(y)).mean().item())
                if math.isfinite(ps):
                    psnr_vals.append(ps)
                if math.isfinite(ss):
                    ssim_vals.append(ss)
                if math.isfinite(dc):
                    dchroma_vals.append(dc)
                if math.isfinite(dl):
                    dluma_vals.append(dl)

        eval_model.load_state_dict(backup, strict=True)

        val_psnr = sum(psnr_vals) / len(psnr_vals) if psnr_vals else 0.0
        val_ssim = sum(ssim_vals) / len(ssim_vals) if ssim_vals else 0.0
        val_dchr = sum(dchroma_vals) / len(dchroma_vals) if dchroma_vals else 0.0
        val_dluma = sum(dluma_vals) / len(dluma_vals) if dluma_vals else 0.0

        writer.add_scalar("Loss/TrainEpoch", running / max(1, len(train_loader)), ep)
        if USE_SYNTH_SEQ:
            writer.add_scalar("Loss/TempKnownEpoch", running_temp_known / max(1, len(train_loader)), ep)
            writer.add_scalar("Loss/TempSimpleEpoch", running_temp_simple / max(1, len(train_loader)), ep)
        writer.add_scalar("PSNR/Val", val_psnr, ep)
        writer.add_scalar("SSIM/Val", val_ssim, ep)
        writer.add_scalar("DeltaChroma/Val", val_dchr, ep)
        writer.add_scalar("DeltaLumaStd/Val", val_dluma, ep)

        print(f"[{ep+1:03d}/{epochs}] Train {running/len(train_loader):.4f} | "
              f"TempK {running_temp_known/max(1,len(train_loader)):.4f} | "
              f"TempS {running_temp_simple/max(1,len(train_loader)):.4f} | "
              f"Val PSNR {val_psnr:.3f} SSIM {val_ssim:.4f} ΔChroma {val_dchr:.4f} ΔLumaStd {val_dluma:.4f}")

        if math.isfinite(val_psnr) and val_psnr > best_psnr:
            best_psnr = val_psnr
            state = (model.module if use_dp else model).state_dict()
            path = os.path.join(ckpt_dir, f"BEST_PSNR_epoch{ep+1:04d}_{val_psnr:.3f}.pth")
            torch.save(state, path)
            print(f"  -> Saved (best PSNR {best_psnr:.3f}): {path}")

        scheduler.step()
        _save_last_ckpt(
            os.path.join(ckpt_dir, "last.pth"),
            model, optimizer, scheduler, scaler, ema,
            epoch=ep + 1, best_psnr=best_psnr, use_dp=use_dp
        )

    writer.close()

if __name__ == "__main__":
    main()
