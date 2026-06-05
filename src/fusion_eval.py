"""EER-kalibreli karar fuzyonunu FF++ test setinde sayisal dogrular.

inference.py'deki AYNI esik + oylama mantigi kullanilir. FF++ sessiz
oldugundan audio modulu pasif; 2-modul (spatial + frequency) fuzyonu
raporlanir. Cikti: reports/fusion_eval/metrics.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from torchvision import transforms

from src.inference import EER_THRESHOLDS, fuse
from src.models.frequency.frequency_model import FrequencyModel
from src.models.spatial.spatial_model import SpatialModel
from src.preprocessing.dataset import DeepfakeVideoDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 8


def _tf():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _load(name, model):
    ckpt = torch.load(f"checkpoints/best_{name}.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()


@torch.no_grad()
def main():
    tf = _tf()
    ds = DeepfakeVideoDataset(root_dir="data/processed", split="test")
    print(f"Cihaz: {DEVICE} | Test videolari: {len(ds)}", flush=True)

    spatial = _load("spatial", SpatialModel(lstm_hidden=256, lstm_layers=2, dropout=0.3))
    freq = _load("frequency", FrequencyModel(in_channels=3, feature_dim=256))

    y_true, p_sp, p_fr, y_fusion = [], [], [], []

    for i in range(len(ds)):
        _, paths, label = ds.samples[i]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        if not imgs:
            continue
        t = torch.stack([tf(im) for im in imgs]).to(DEVICE)

        # spatial: SEQ_LEN'lik dizi
        n = len(imgs)
        if n >= SEQ_LEN:
            idx = np.linspace(0, n - 1, SEQ_LEN, dtype=int)
            seq = t[idx]
        else:
            seq = torch.cat([t, t[-1:].repeat(SEQ_LEN - n, 1, 1, 1)], 0)
        sp = float(torch.softmax(spatial(seq.unsqueeze(0)), 1)[0, 1])

        # frequency: kare ortalamasi
        fr_probs = []
        for j in range(0, len(t), 16):
            fr_probs += torch.softmax(freq(t[j:j + 16]), 1)[:, 1].tolist()
        fr = float(np.mean(fr_probs))

        # inference.py ile AYNI fuze() (audio yok -> 2 modul)
        f = fuse({"spatial": sp, "frequency": fr, "audio": None})
        verdict = 1 if f["verdict"] == "FAKE" else 0

        y_true.append(label)
        p_sp.append(sp)
        p_fr.append(fr)
        y_fusion.append(verdict)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ds)}", flush=True)

    y_true = np.array(y_true)
    metrics = {
        "n_videos": int(len(y_true)),
        "modules_used": ["spatial", "frequency"],
        "note": "FF++ sessiz oldugundan audio pasif; decision-level fusion.",
        "fusion": {
            "accuracy": float(accuracy_score(y_true, y_fusion)),
            "precision": float(precision_score(y_true, y_fusion, zero_division=0)),
            "recall": float(recall_score(y_true, y_fusion, zero_division=0)),
            "f1": float(f1_score(y_true, y_fusion, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_fusion).tolist(),
        },
        "spatial_only": {
            "auc": float(roc_auc_score(y_true, p_sp)),
            "accuracy": float(accuracy_score(
                y_true, (np.array(p_sp) >= EER_THRESHOLDS["spatial"]).astype(int))),
        },
        "frequency_only": {
            "auc": float(roc_auc_score(y_true, p_fr)),
            "accuracy": float(accuracy_score(
                y_true, (np.array(p_fr) >= EER_THRESHOLDS["frequency"]).astype(int))),
        },
    }

    out = Path("reports/fusion_eval")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nRapor -> {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
