"""GPU-optimized training loop with mixed precision and early stopping."""
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


def train_epoch(model, dataloader, optimizer, scaler, scheduler, cfg, device):
    model.train()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    optimizer.zero_grad(set_to_none=True)
    accum = cfg.gradient_accumulation_steps

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=cfg.mixed_precision and device.type == 'cuda'):
            pred = model(x)
            loss = nn.functional.l1_loss(pred, y)

        loss = loss / accum
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        loss_meter.update(loss.item() * accum, x.size(0))
        mae_meter.update(compute_mae(pred, y), x.size(0))

    return loss_meter.avg, mae_meter.avg


@torch.no_grad()
def validate(model, dataloader, cfg, device):
    model.eval()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    all_preds = []
    all_targets = []

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=cfg.mixed_precision and device.type == 'cuda'):
            pred = model(x)
            loss = nn.functional.l1_loss(pred, y)

        loss_meter.update(loss.item(), x.size(0))
        mae_meter.update(compute_mae(pred, y), x.size(0))
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Per-hour MAE (24 values)
    per_hour_mae = np.mean(np.abs(all_preds - all_targets), axis=0)

    return loss_meter.avg, mae_meter.avg, per_hour_mae, all_preds, all_targets


def train_model(model, train_loader, val_loader, cfg, device):
    model = model.to(device)

    if cfg.compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("  [INFO] torch.compile enabled")
        except Exception as e:
            print(f"  [WARN] torch.compile failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.max_epochs
    warmup_steps = len(train_loader) * cfg.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.mixed_precision)

    best_val_mae = float('inf')
    best_state = None
    best_per_hour = None
    patience_counter = 0

    for epoch in range(cfg.max_epochs):
        t0 = time.time()

        train_loss, train_mae = train_epoch(model, train_loader, optimizer, scaler, scheduler, cfg, device)
        val_loss, val_mae, per_hour_mae, preds, targets = validate(model, val_loader, cfg, device)

        epoch_time = time.time() - t0

        # Per-hour bins
        early = np.mean(per_hour_mae[:8])
        mid = np.mean(per_hour_mae[8:16])
        late = np.mean(per_hour_mae[16:])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg.max_epochs} | "
                  f"t_loss={train_mae:.0f} v_loss={val_mae:.0f} | "
                  f"per-hr {early:.0f}/{mid:.0f}/{late:.0f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                  f"{epoch_time:.0f}s", flush=True)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_per_hour = per_hour_mae.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, best_val_mae, best_per_hour
