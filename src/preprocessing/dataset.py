"""Deepfake veri seti yukleyicileri (DataLoader)."""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms


class DeepfakeFrameDataset(Dataset):
    """Onceden cikarilmis yuz kareleri icin veri seti.

    Beklenen klasor yapisi:
        data/processed/
            train/
                real/
                    video001_frame00.png
                    video001_frame01.png
                fake/
                    video002_frame00.png
            val/
                real/
                fake/
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform or self._default_transform(split)

        self.samples = []
        self.labels = []

        # real=0, fake=1
        for label, class_name in enumerate(["real", "fake"]):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append(str(img_path))
                self.labels.append(label)
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append(str(img_path))
                self.labels.append(label)

    @staticmethod
    def _default_transform(split: str):
        if split == "train":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


class DeepfakeVideoDataset(Dataset):
    """Video seviyesinde veri seti - kare dizisi + ses + frekans.

    Her ornek icin:
    - frames: (seq_len, C, H, W) - yuz kareleri dizisi
    - audio: (1, n_mels, T) - mel-spektrogram
    - label: 0 (gercek) veya 1 (sahte)

    Beklenen klasor yapisi:
        data/processed/
            train/
                real/
                    video001/
                        frames/  (frame_00.png, frame_01.png, ...)
                        audio.npy  (mel-spektrogram)
                fake/
                    video002/
                        frames/
                        audio.npy
    """

    def __init__(self, root_dir: str, split: str = "train",
                 seq_len: int = 10, transform=None):
        self.root_dir = Path(root_dir) / split
        self.seq_len = seq_len
        self.transform = transform or DeepfakeFrameDataset._default_transform(split)

        self.samples = []  # (video_dir, label)

        for label, class_name in enumerate(["real", "fake"]):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for video_dir in sorted(class_dir.iterdir()):
                if video_dir.is_dir() and (video_dir / "frames").exists():
                    self.samples.append((str(video_dir), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        video_dir = Path(video_dir)

        # Kareleri yukle
        frame_dir = video_dir / "frames"
        frame_paths = sorted(frame_dir.glob("*.png")) + sorted(frame_dir.glob("*.jpg"))

        # Esit aralikla seq_len kare sec
        if len(frame_paths) >= self.seq_len:
            indices = np.linspace(0, len(frame_paths) - 1, self.seq_len, dtype=int)
        else:
            indices = list(range(len(frame_paths)))
            # Eksik kareleri son kareyle doldur
            while len(indices) < self.seq_len:
                indices.append(indices[-1])

        frames = []
        for i in indices:
            img = Image.open(frame_paths[i]).convert("RGB")
            img = self.transform(img)
            frames.append(img)
        frames = torch.stack(frames)  # (seq_len, C, H, W)

        # Ses spektrogrami yukle
        audio_path = video_dir / "audio.npy"
        if audio_path.exists():
            audio = np.load(str(audio_path))
            audio = torch.from_numpy(audio).float().unsqueeze(0)  # (1, n_mels, T)
        else:
            audio = torch.zeros(1, 128, 300)  # Bos spektrogram

        return frames, audio, label
