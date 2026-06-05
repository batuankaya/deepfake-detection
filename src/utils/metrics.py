"""Degerlendirme metrikleri."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(targets: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate (EER) ve esik degerini hesaplar.

    EER: FAR (false accept rate) == FRR (false reject rate) noktasi.
    Ses deepfake literaturunde standart metrik (ornek: ASVspoof).
    """
    if len(np.unique(targets)) < 2:
        return 0.0, 0.5
    fpr, tpr, thresholds = roc_curve(targets, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thresholds[idx])


class MetricsTracker:
    """Egitim ve degerlendirme metriklerini takip eder."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.losses = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float):
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.probabilities.extend(probs[:, 1].detach().cpu().numpy())
        self.losses.append(loss)

    def compute(self) -> dict:
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        probs = np.array(self.probabilities)

        metrics = {
            "loss": float(np.mean(self.losses)) if self.losses else 0.0,
            "accuracy": accuracy_score(targets, preds),
            "precision": precision_score(targets, preds, zero_division=0),
            "recall": recall_score(targets, preds, zero_division=0),
            "f1": f1_score(targets, preds, zero_division=0),
        }

        if len(np.unique(targets)) > 1:
            metrics["auc_roc"] = roc_auc_score(targets, probs)
            metrics["eer"], metrics["eer_threshold"] = compute_eer(targets, probs)
        else:
            metrics["auc_roc"] = 0.0
            metrics["eer"] = 0.0
            metrics["eer_threshold"] = 0.5

        return metrics

    @staticmethod
    def print_metrics(metrics: dict, prefix: str = ""):
        print(
            f"  {prefix}Loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"AUC: {metrics['auc_roc']:.4f} | "
            f"EER: {metrics['eer']:.4f}"
        )

    def save_plots(self, out_dir: str, prefix: str = "eval") -> None:
        """ROC ve confusion matrix gorsellerini kaydeder."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        targets = np.array(self.targets)
        probs = np.array(self.probabilities)
        preds = np.array(self.predictions)

        if len(np.unique(targets)) > 1:
            fpr, tpr, _ = roc_curve(targets, probs)
            auc = roc_auc_score(targets, probs)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(out_path / f"{prefix}_roc.png", dpi=120)
            plt.close(fig)

        cm = confusion_matrix(targets, preds, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Real", "Fake"])
        ax.set_yticks([0, 1], ["Real", "Fake"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_path / f"{prefix}_cm.png", dpi=120)
        plt.close(fig)
