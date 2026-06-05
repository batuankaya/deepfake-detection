"""Deepfake Tespit Sistemi - Egitim Scripti.

Kullanim:
    python -m src.train --config configs/config.yaml
    python -m src.train --config configs/config.yaml --module spatial
    python -m src.train --config configs/config.yaml --module frequency
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.audio.audio_model import AudioModel
from src.models.frequency.frequency_model import FrequencyModel
from src.models.spatial.spatial_model import SpatialModel
from src.preprocessing.dataset import (
    DeepfakeAudioDataset,
    DeepfakeFrameDataset,
    DeepfakeVideoDataset,
)
from src.utils.device import get_device
from src.utils.metrics import MetricsTracker


SEED = 42
WARMUP_STEPS = 500


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017). Class imbalance + hard example mining.

    gamma=2 standart, alpha=0.25 default (positive class weight).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        p = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - p) ** self.gamma * ce
        return loss.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # hiz icin (deterministik degil)


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config dosyasi bulunamadi: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def unpack_batch(batch, module_type, device):
    if module_type in ("spatial", "full"):
        frames, audio, labels = batch
        return frames.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    inputs, labels = batch
    return inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, warmup,
                    device, module_type, base_lr, epoch_label=""):
    """Tek bir epoch egitim (AMP + warmup + tqdm ile)."""
    model.train()
    tracker = MetricsTracker()

    pbar = tqdm(dataloader, desc=f"Train {epoch_label}", unit="batch",
                ncols=100, leave=True)
    for batch in pbar:
        inputs, labels = unpack_batch(batch, module_type, device)

        if warmup["step"] < warmup["total"]:
            lr = base_lr * (warmup["step"] + 1) / warmup["total"]
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        warmup["step"] += 1

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tracker.update(outputs, labels, loss.item())
        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.1e}")
    pbar.close()
    return tracker.compute()


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, module_type, epoch_label=""):
    """Degerlendirme (AMP autocast + tqdm)."""
    model.eval()
    tracker = MetricsTracker()

    pbar = tqdm(dataloader, desc=f"Val   {epoch_label}", unit="batch",
                ncols=100, leave=True)
    for batch in pbar:
        inputs, labels = unpack_batch(batch, module_type, device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        tracker.update(outputs, labels, loss.item())
    pbar.close()
    return tracker.compute()


def create_model(module_type: str, config: dict, device: torch.device):
    """Modul tipine gore model olusturur."""
    if module_type == "spatial":
        model = SpatialModel(
            lstm_hidden=config["spatial"]["lstm_hidden"],
            lstm_layers=config["spatial"]["lstm_layers"],
            dropout=config["spatial"]["dropout"],
        )
    elif module_type == "audio":
        model = AudioModel(n_mels=config["audio"]["n_mels"])
    elif module_type == "frequency":
        model = FrequencyModel(
            in_channels=3,
            feature_dim=config["frequency"]["feature_dim"],
        )
    else:
        raise ValueError(f"Bilinmeyen modul: {module_type}")

    return model.to(device)


def create_dataloader(module_type: str, config: dict, split: str, num_workers: int,
                      seq_len: int | None = None):
    """Modul tipine gore dataloader olusturur."""
    if module_type == "spatial":
        dataset = DeepfakeVideoDataset(
            root_dir="data/processed", split=split,
            seq_len=seq_len if seq_len is not None else 10,
        )
    elif module_type == "frequency":
        dataset = DeepfakeFrameDataset(
            root_dir="data/processed", split=split,
        )
    elif module_type == "audio":
        dataset = DeepfakeAudioDataset(
            root_dir="data/processed/audio", split=split,
        )
    else:
        raise ValueError(f"Bilinmeyen modul: {module_type}")

    batch_size = config["training"]["batch_size"]

    # Egitim icin class-imbalance sampler; val/test'te shuffle yok
    sampler = None
    shuffle = False
    if split == "train" and hasattr(dataset, "make_weighted_sampler"):
        sampler = dataset.make_weighted_sampler()
    elif split == "train":
        shuffle = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Deepfake Tespit - Egitim")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--module", type=str, default="frequency",
                        choices=["spatial", "audio", "frequency"],
                        help="Egitilecek modul")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Islenmis veri klasoru")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker sayisi (CPU)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Mixed precision'i devre disi birak")
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "focal"],
                        help="Kayip fonksiyonu: ce (CrossEntropy) veya focal")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch boyutu override (config.yaml ezilir)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Spatial modul icin video dizi uzunlugu override")
    args = parser.parse_args()

    set_seed(SEED)
    config = load_config(args.config)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    device = get_device(config["device"])
    print(f"Cihaz: {device} | Seed: {SEED} | AMP: {not args.no_amp} | "
          f"Batch: {config['training']['batch_size']}")

    # Model
    model = create_model(args.module, config, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.module}")
    print(f"Toplam parametre: {total_params:,}")
    print(f"Egitilir parametre: {trainable_params:,}")

    # Dataloader
    train_loader = create_dataloader(args.module, config, "train", args.num_workers,
                                      seq_len=args.seq_len)
    val_loader = create_dataloader(args.module, config, "val", args.num_workers,
                                    seq_len=args.seq_len)
    print(f"Egitim ornekleri: {len(train_loader.dataset)}")
    print(f"Dogrulama ornekleri: {len(val_loader.dataset)}")

    if len(train_loader.dataset) == 0:
        print("\n[HATA] Egitim verisi bulunamadi!")
        print(f"Lutfen verileri '{args.data_dir}/train/real/' ve "
              f"'{args.data_dir}/train/fake/' klasorlerine yerlestirin.")
        return

    # Egitim ayarlari
    criterion = FocalLoss() if args.loss == "focal" else nn.CrossEntropyLoss()
    print(f"Loss: {args.loss}")
    base_lr = config["training"]["learning_rate"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp and torch.cuda.is_available())
    warmup = {"step": 0, "total": WARMUP_STEPS}

    # Egitim dongusu
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    total_epochs = config["training"]["epochs"]
    print(f"\nEgitim basliyor ({total_epochs} epoch, "
          f"warmup={WARMUP_STEPS} step, batch_size={config['training']['batch_size']})")
    print("=" * 80)

    for epoch in range(total_epochs):
        epoch_label = f"[{epoch + 1}/{total_epochs}]"
        print(f"\n{'='*80}")
        print(f"  EPOCH {epoch + 1} / {total_epochs}   "
              f"(best_val_acc={best_val_acc:.4f}, patience={patience_counter}/"
              f"{config['training']['early_stopping_patience']})")
        print("=" * 80)

        # Egitim
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, warmup,
            device, args.module, base_lr, epoch_label=epoch_label,
        )
        MetricsTracker.print_metrics(train_metrics, prefix=f"Epoch {epoch+1} Train - ")

        # Degerlendirme
        val_metrics = evaluate(model, val_loader, criterion, device, args.module,
                                epoch_label=epoch_label)
        MetricsTracker.print_metrics(val_metrics, prefix=f"Epoch {epoch+1} Val   - ")

        if warmup["step"] >= warmup["total"]:
            scheduler.step()

        # En iyi modeli kaydet
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            save_path = checkpoint_dir / f"best_{args.module}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": config,
            }, save_path)
            print(f"  >> En iyi model kaydedildi: {save_path} "
                  f"(val_acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  Iyilesme yok (patience: {patience_counter}/"
                  f"{config['training']['early_stopping_patience']})")
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"\nErken durdurma! {patience_counter} epoch iyilesme yok.")
                break

    print(f"\n{'='*80}")
    print(f"  EGITIM TAMAMLANDI")
    print(f"  En iyi val accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoint: checkpoints/best_{args.module}.pt")
    print("=" * 80)


if __name__ == "__main__":
    main()
