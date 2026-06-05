"""Egitilmis modeli test seti uzerinde calistirir, metrik + grafik uretir.

Kullanim:
    python -m src.evaluate --checkpoint checkpoints/best_frequency.pt
    python -m src.evaluate --checkpoint checkpoints/best_frequency.pt --split test
    python -m src.evaluate --checkpoint checkpoints/best_frequency.pt --split cross_celebdf
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
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


def build_model(module_type: str, config: dict, device: torch.device):
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


def build_dataloader(module_type: str, split: str, batch_size: int, num_workers: int):
    if module_type == "spatial":
        dataset = DeepfakeVideoDataset(root_dir="data/processed", split=split, seq_len=10)
    elif module_type == "audio":
        dataset = DeepfakeAudioDataset(root_dir="data/processed/audio", split=split)
    else:
        dataset = DeepfakeFrameDataset(root_dir="data/processed", split=split)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    ), dataset


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, module_type, split: str):
    model.eval()
    tracker = MetricsTracker()
    misclassified = []
    pbar = tqdm(dataloader, desc=f"Eval {split}", unit="batch")
    for batch in pbar:
        if module_type == "spatial":
            inputs, _, labels = batch
        else:
            inputs, labels = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        tracker.update(outputs, labels, loss.item())

        preds = outputs.argmax(dim=1)
        wrong = (preds != labels).nonzero(as_tuple=True)[0].cpu().tolist()
        for w in wrong:
            misclassified.append({"label": int(labels[w].item()),
                                  "pred": int(preds[w].item())})
    pbar.close()
    return tracker, misclassified


def main():
    parser = argparse.ArgumentParser(description="Model degerlendirme")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="best_<module>.pt dosyasi")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test", "cross_celebdf"],
                        help="Hangi split'te degerlendirilsin")
    parser.add_argument("--module", type=str, default=None,
                        choices=["spatial", "audio", "frequency"],
                        help="Modul tipi (checkpoint adindan da turetilir)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Rapor klasoru (varsayilan: reports/<ckpt_stem>_<split>)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint yok: {ckpt_path}")

    # Modul tipini checkpoint isminden cikar (best_frequency.pt -> frequency)
    module = args.module
    if module is None:
        stem = ckpt_path.stem  # best_frequency
        for m in ("frequency", "spatial", "audio"):
            if m in stem:
                module = m
                break
    if module is None:
        raise ValueError("--module belirtilmedi ve checkpoint adindan turetilemedi")

    out_dir = Path(args.out_dir) if args.out_dir \
              else Path("reports") / f"{ckpt_path.stem}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = get_device(config.get("device", "auto"))
    print(f"Cihaz: {device} | Modul: {module} | Split: {args.split}")

    model = build_model(module, config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint epoch={checkpoint.get('epoch')} "
          f"val_acc={checkpoint.get('val_accuracy', 0):.4f}")

    dataloader, dataset = build_dataloader(module, args.split, args.batch_size, args.num_workers)
    if len(dataset) == 0:
        print(f"[HATA] {args.split} split bos. Preprocessing tamamlandi mi?")
        return

    criterion = nn.CrossEntropyLoss()
    tracker, misclassified = evaluate(model, dataloader, criterion, device,
                                       module, args.split)
    metrics = tracker.compute()
    MetricsTracker.print_metrics(metrics, prefix=f"{args.split} - ")

    tracker.save_plots(str(out_dir), prefix=args.split)
    print(f"Plotlar -> {out_dir}/")

    # EER threshold ile yeniden Acc/F1 hesapla
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    probs = np.array(tracker.probabilities)
    targets = np.array(tracker.targets)
    eer_thresh = metrics.get("eer_threshold", 0.5)
    preds_at_eer = (probs >= eer_thresh).astype(int)
    eer_metrics = {
        "threshold": float(eer_thresh),
        "accuracy": float(accuracy_score(targets, preds_at_eer)),
        "f1": float(f1_score(targets, preds_at_eer, zero_division=0)),
    }
    print(f"\n  EER threshold ({eer_thresh:.4f}) ile:")
    print(f"  Acc: {eer_metrics['accuracy']:.4f} | F1: {eer_metrics['f1']:.4f}")

    report = {
        "checkpoint": str(ckpt_path),
        "module": module,
        "split": args.split,
        "num_samples": len(dataset),
        "num_misclassified": len(misclassified),
        "metrics": metrics,
        "metrics_at_eer_threshold": eer_metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Rapor -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
