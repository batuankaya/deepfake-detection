"""Degerlendirme metrikleri."""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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
            "loss": np.mean(self.losses),
            "accuracy": accuracy_score(targets, preds),
            "precision": precision_score(targets, preds, zero_division=0),
            "recall": recall_score(targets, preds, zero_division=0),
            "f1": f1_score(targets, preds, zero_division=0),
        }

        # AUC - en az 2 sinif olmali
        if len(np.unique(targets)) > 1:
            metrics["auc_roc"] = roc_auc_score(targets, probs)
        else:
            metrics["auc_roc"] = 0.0

        return metrics

    @staticmethod
    def print_metrics(metrics: dict, prefix: str = ""):
        print(f"  {prefix}Loss: {metrics['loss']:.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"AUC: {metrics['auc_roc']:.4f}")
