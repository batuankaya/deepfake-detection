"""Deepfake veri seti yukleyicileri (DataLoader)."""

import io
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


class SpecAugment:
    """SpecAugment: mel-spektrogramda zaman/frekans bandı maskeleme.

    Park et al. (2019), ASVspoof literatüründe cross-attack generalization
    için standart. Tek bir mel tensoru üzerinde freq ve time mask uygular.
    """

    def __init__(self, freq_mask=27, time_mask=70, n_freq_masks=2, n_time_masks=2):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (1, n_mels, T)
        _, n_mels, t = mel.shape
        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask, n_mels - 1))
            if f == 0:
                continue
            f0 = random.randint(0, n_mels - f)
            mel[:, f0:f0 + f, :] = mel.min()
        for _ in range(self.n_time_masks):
            tm = random.randint(0, min(self.time_mask, t - 1))
            if tm == 0:
                continue
            t0 = random.randint(0, t - tm)
            mel[:, :, t0:t0 + tm] = mel.min()
        return mel


class DeepfakeAudioDataset(Dataset):
    """Mel-spektrogram .npy dosyalari icin veri seti.

    Beklenen yapi:
        data/processed/audio/
            train/{real,fake}/*.npy   (her dosya: float32 (n_mels, T))
            val/{real,fake}/*.npy
            test/{real,fake}/*.npy

    Train split'inde SpecAugment uygulanir (augmentation).
    """

    def __init__(self, root_dir: str = "data/processed/audio", split: str = "train",
                 use_specaugment: bool = True):
        self.root = Path(root_dir) / split
        self.samples: list[str] = []
        self.labels: list[int] = []
        self.augment = SpecAugment() if (split == "train" and use_specaugment) else None
        for label, cls in enumerate(["real", "fake"]):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.glob("*.npy")):
                self.samples.append(str(f))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        mel = np.load(self.samples[idx], allow_pickle=False).astype(np.float32)
        # (n_mels, T) -> (1, n_mels, T) (channel ekle)
        tensor = torch.from_numpy(mel).unsqueeze(0)
        if self.augment is not None:
            tensor = self.augment(tensor)
        return tensor, self.labels[idx]

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        labels = np.array(self.labels)
        class_count = np.bincount(labels, minlength=2)
        class_weight = 1.0 / np.maximum(class_count, 1)
        sample_weight = class_weight[labels]
        return WeightedRandomSampler(
            weights=sample_weight.tolist(),
            num_samples=len(sample_weight),
            replacement=True,
        )


class RandomJPEGCompression:
    """JPEG sikistirma augmentation'i (deepfake robustlugu icin)."""

    def __init__(self, quality_range=(40, 90), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


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
        root_resolved = self.root_dir.resolve()
        for label, class_name in enumerate(["real", "fake"]):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                if not img_path.resolve().is_relative_to(root_resolved):
                    continue  # path traversal korunmasi
                self.samples.append(str(img_path))
                self.labels.append(label)
            for img_path in sorted(class_dir.glob("*.jpg")):
                if not img_path.resolve().is_relative_to(root_resolved):
                    continue
                self.samples.append(str(img_path))
                self.labels.append(label)

    @staticmethod
    def _default_transform(split: str):
        if split == "train":
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                RandomJPEGCompression(quality_range=(40, 90), p=0.3),
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

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        """Sinif dengesizligi icin ornek agirlikli sampler.

        FF++ 1:4 (real:fake) dengesizligi var; bu sampler her sinifi
        esit olasilikla cekmeyi saglar.
        """
        labels = np.array(self.labels)
        class_count = np.bincount(labels, minlength=2)
        class_weight = 1.0 / np.maximum(class_count, 1)
        sample_weight = class_weight[labels]
        return WeightedRandomSampler(
            weights=sample_weight.tolist(),
            num_samples=len(sample_weight),
            replacement=True,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


class DeepfakeVideoDataset(Dataset):
    """Video seviyesinde veri seti - kare dizisi (spatial modul icin).

    build_dataset.py duz yapida kareler uretir:
        data/processed/train/real/<video_id>__frame000.jpg
        data/processed/train/real/<video_id>__frame001.jpg

    Bu dataset kareleri video_id'ye gore gruplandirip her video icin
    seq_len uzunlukta kare dizisi olusturur.

    Donus: (frames, audio_placeholder, label)
      frames: (seq_len, C, H, W)
      audio_placeholder: (1, 128, 300)  # spatial-only egitim icin bos
      label: 0 (gercek) / 1 (sahte)
    """

    def __init__(self, root_dir: str, split: str = "train",
                 seq_len: int = 10, transform=None):
        self.root_dir = Path(root_dir) / split
        self.seq_len = seq_len
        self.transform = transform or DeepfakeFrameDataset._default_transform(split)

        # (video_id, [frame_paths], label)
        self.samples: list[tuple[str, list[str], int]] = []
        self.labels: list[int] = []

        from collections import defaultdict
        for label, class_name in enumerate(["real", "fake"]):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            # video_id'ye gore grupla: dosya adi "<id>__frameNNN.jpg"
            groups: dict[str, list[Path]] = defaultdict(list)
            for f in class_dir.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".jpg", ".png"):
                    continue
                name = f.stem
                if "__frame" in name:
                    vid = name.rsplit("__frame", 1)[0]
                else:
                    vid = name
                groups[vid].append(f)
            for vid, frames in groups.items():
                if len(frames) < 2:
                    continue
                frames.sort()
                self.samples.append((vid, [str(p) for p in frames], label))
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        labels = np.array(self.labels)
        class_count = np.bincount(labels, minlength=2)
        class_weight = 1.0 / np.maximum(class_count, 1)
        sample_weight = class_weight[labels]
        return WeightedRandomSampler(
            weights=sample_weight.tolist(),
            num_samples=len(sample_weight),
            replacement=True,
        )

    def __getitem__(self, idx):
        _, frame_paths, label = self.samples[idx]

        # Esit aralikla seq_len kare sec
        n = len(frame_paths)
        if n >= self.seq_len:
            indices = np.linspace(0, n - 1, self.seq_len, dtype=int)
        else:
            indices = list(range(n)) + [n - 1] * (self.seq_len - n)

        frames = []
        for i in indices:
            img = Image.open(frame_paths[i]).convert("RGB")
            img = self.transform(img)
            frames.append(img)
        frames = torch.stack(frames)  # (seq_len, C, H, W)

        audio_placeholder = torch.zeros(1, 128, 300)
        return frames, audio_placeholder, label
