"""Deepfake Tespit Sistemi - Egitim Scripti.

Kullanim:
    python -m src.train --config configs/config.yaml
    python -m src.train --config configs/config.yaml --module spatial
    python -m src.train --config configs/config.yaml --module frequency
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.spatial.spatial_model import SpatialModel
from src.models.audio.audio_model import AudioModel
from src.models.frequency.frequency_model import FrequencyModel
from src.models.fusion.fusion_model import HierarchicalFusionModel
from src.preprocessing.dataset import DeepfakeFrameDataset, DeepfakeVideoDataset
from src.utils.device import get_device
from src.utils.metrics import MetricsTracker


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config dosyasi bulunamadi: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, criterion, optimizer, device, module_type):
    """Tek bir epoch egitim."""
    model.train()
    tracker = MetricsTracker()

    for batch_idx, batch in enumerate(dataloader):
        if module_type in ("spatial", "full"):
            frames, audio, labels = batch
            inputs = frames.to(device)
            labels = labels.to(device)
        elif module_type == "frequency":
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
        elif module_type == "audio":
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        tracker.update(outputs, labels, loss.item())

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")

    return tracker.compute()


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, module_type):
    """Degerlendirme."""
    model.eval()
    tracker = MetricsTracker()

    for batch in dataloader:
        if module_type in ("spatial", "full"):
            frames, audio, labels = batch
            inputs = frames.to(device)
            labels = labels.to(device)
        elif module_type == "frequency":
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
        elif module_type == "audio":
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        tracker.update(outputs, labels, loss.item())

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


def create_dataloader(module_type: str, config: dict, split: str):
    """Modul tipine gore dataloader olusturur."""
    data_dir = "data/processed"

    if module_type == "spatial":
        dataset = DeepfakeVideoDataset(
            root_dir=data_dir, split=split,
            seq_len=10,
        )
    elif module_type in ("frequency", "audio"):
        dataset = DeepfakeFrameDataset(
            root_dir=data_dir, split=split,
        )
    else:
        raise ValueError(f"Bilinmeyen modul: {module_type}")

    batch_size = config["training"]["batch_size"]
    shuffle = split == "train"

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def main():
    parser = argparse.ArgumentParser(description="Deepfake Tespit - Egitim")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--module", type=str, default="frequency",
                        choices=["spatial", "audio", "frequency"],
                        help="Egitilecek modul")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Islenmis veri klasoru")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["device"])
    print(f"Cihaz: {device}")

    # Model
    model = create_model(args.module, config, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.module}")
    print(f"Toplam parametre: {total_params:,}")
    print(f"Egitilir parametre: {trainable_params:,}")

    # Dataloader
    train_loader = create_dataloader(args.module, config, "train")
    val_loader = create_dataloader(args.module, config, "val")
    print(f"Egitim ornekleri: {len(train_loader.dataset)}")
    print(f"Dogrulama ornekleri: {len(val_loader.dataset)}")

    if len(train_loader.dataset) == 0:
        print("\n[HATA] Egitim verisi bulunamadi!")
        print(f"Lutfen verileri '{args.data_dir}/train/real/' ve "
              f"'{args.data_dir}/train/fake/' klasorlerine yerlestirin.")
        return

    # Egitim ayarlari
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"],
    )

    # Egitim dongusu
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"\nEgitim basliyor ({config['training']['epochs']} epoch)...")
    print("=" * 60)

    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch [{epoch + 1}/{config['training']['epochs']}]")

        # Egitim
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.module
        )
        MetricsTracker.print_metrics(train_metrics, prefix="Train - ")

        # Degerlendirme
        val_metrics = evaluate(model, val_loader, criterion, device, args.module)
        MetricsTracker.print_metrics(val_metrics, prefix="Val   - ")

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
            print(f"  -> En iyi model kaydedildi: {save_path} "
                  f"(acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"\nErken durdurma! {patience_counter} epoch iyilesme yok.")
                break

    print(f"\nEgitim tamamlandi! En iyi val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
