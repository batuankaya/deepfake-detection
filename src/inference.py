"""Deepfake Tespit Sistemi - Inference Pipeline.

Video dosyasini alip uctan uca analiz yapar.
Egitilmis model varsa kullanir, yoksa sadece frekans analizini gorsellestir.
"""

import numpy as np
import torch
import cv2
from pathlib import Path

from src.preprocessing.video_processor import VideoProcessor
from src.models.frequency.frequency_model import FrequencyAnalyzer
from src.utils.device import get_device


class DeepfakeDetector:
    """Uctan uca deepfake tespit pipeline'i."""

    def __init__(self, config: dict = None, checkpoint_dir: str = "checkpoints"):
        self.config = config or {}
        self.device = get_device(self.config.get("device", "auto"))
        self.checkpoint_dir = Path(checkpoint_dir)
        self.video_processor = VideoProcessor(
            frame_rate=self.config.get("preprocessing", {}).get("frame_rate", 5),
            max_frames=self.config.get("preprocessing", {}).get("max_frames", 150),
        )
        self.freq_analyzer = FrequencyAnalyzer()
        self.models_loaded = False

    def extract_frames(self, video_path: str) -> list[np.ndarray]:
        """Videodan kareleri cikarir."""
        return self.video_processor.extract_frames(video_path)

    def analyze_frequency(self, frame: np.ndarray) -> dict:
        """Tek bir kare uzerinde frekans analizi yapar (egitim gerektirmez)."""
        fft_spectrum = self.freq_analyzer.compute_fft_spectrum(frame)
        azimuthal = self.freq_analyzer.compute_azimuthal_average(fft_spectrum)

        # DCT analizi
        dct_spectrum = self.freq_analyzer.compute_dct_spectrum(frame)

        # Basit istatistiksel skor (egitilmis model olmadan)
        high_freq_energy = np.mean(azimuthal[len(azimuthal) // 2:])
        low_freq_energy = np.mean(azimuthal[:len(azimuthal) // 2])
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-8)

        return {
            "fft_spectrum": fft_spectrum,
            "dct_spectrum": dct_spectrum,
            "azimuthal_profile": azimuthal,
            "high_freq_energy": float(high_freq_energy),
            "low_freq_energy": float(low_freq_energy),
            "freq_ratio": float(freq_ratio),
        }

    def analyze_video(self, video_path: str) -> dict:
        """Video dosyasini analiz eder.

        Returns:
            dict: Analiz sonuclari (kareler, frekans haritasi, skor vb.)
        """
        results = {
            "video_path": video_path,
            "frames_analyzed": 0,
            "frequency_analysis": None,
            "sample_frames": [],
        }

        # Kareleri cikar
        frames = self.extract_frames(video_path)
        results["frames_analyzed"] = len(frames)

        if len(frames) == 0:
            results["error"] = "Videodan kare cikarilamadi"
            return results

        # Ornek kareleri sakla (ilk, orta, son)
        indices = [0, len(frames) // 2, len(frames) - 1]
        results["sample_frames"] = [frames[i] for i in indices if i < len(frames)]

        # Frekans analizi (ortadaki kare uzerinde)
        mid_frame = frames[len(frames) // 2]
        results["frequency_analysis"] = self.analyze_frequency(mid_frame)

        # Tum karelerin ortalama frekans skoru
        freq_ratios = []
        for frame in frames[::max(1, len(frames) // 10)]:  # Her 10 kareden 1
            analysis = self.analyze_frequency(frame)
            freq_ratios.append(analysis["freq_ratio"])

        results["avg_freq_ratio"] = float(np.mean(freq_ratios))
        results["std_freq_ratio"] = float(np.std(freq_ratios))

        return results
