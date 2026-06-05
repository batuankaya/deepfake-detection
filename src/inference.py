"""Deepfake Tespit Sistemi - End-to-end Inference Pipeline.

Egitilmis 4 modulun (spatial, audio, frequency, fusion) ortak kullanimi.
Eksik checkpoint'ler graceful handle edilir — mevcut moduller calismaya devam eder.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.audio.audio_model import AudioModel
from src.models.frequency.frequency_model import FrequencyAnalyzer, FrequencyModel
from src.models.spatial.spatial_model import SpatialModel
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.video_processor import VideoProcessor
from src.utils.device import get_device


CHECKPOINTS = {
    "audio": "checkpoints/best_audio.pt",
    "frequency": "checkpoints/best_frequency.pt",
    "spatial": "checkpoints/best_spatial.pt",
}

# Eval seti uzerinden hesaplanmis EER threshold'lar (reports/*_test/metrics.json).
# Her modul kendi optimum karar siniri ile kalibre edilir.
EER_THRESHOLDS = {
    "audio": 0.2002,
    "frequency": 0.7031,
    "spatial": 0.7852,
}

# Test seti AUC'lari (reports/*_test, reports/fusion_eval). Fuzyonda her modul
# kendi guvenilirligiyle orantili agirlik alir; zayif modul gucluyu sulandirmaz.
MODULE_AUC = {
    "audio": 0.9806,
    "frequency": 0.8927,
    "spatial": 0.9812,
}

# Bir modulun kalibre olasiligi bu esigi gecerse, agirlikli ortalama
# "gercek" dese bile SAHTE alarmi verilir (yuksek-recall guvenlik kurali).
STRONG_FAKE = 0.85


def _calibrate(prob: float, threshold: float) -> float:
    """Bir modulun ham olasiligini EER threshold'una gore [0,1]'e kalibre eder.

    prob == threshold -> 0.5 (karar sinirinda)
    prob < threshold -> [0, 0.5) (gercek olasiligi)
    prob > threshold -> (0.5, 1] (sahte olasiligi)
    """
    if prob <= threshold:
        return 0.5 * prob / max(threshold, 1e-6)
    return 0.5 + 0.5 * (prob - threshold) / max(1 - threshold, 1e-6)


def fuse(module_probs: dict) -> dict:
    """Modul ham olasiliklarini AUC-agirlikli karar fuzyonu ile birlestirir.

    module_probs: {"spatial": p|None, "frequency": p|None, "audio": p|None}

    Her modul once kendi EER esigiyle [0,1]'e kalibre edilir, sonra
    guvenilirligiyle (AUC) orantili agirlikla ortalanir. Esit oylamanin
    aksine zayif modul gucluyu sulandirmaz.
    """
    calibrated, weights = {}, {}
    for name, p in module_probs.items():
        if p is None:
            continue
        calibrated[name] = _calibrate(p, EER_THRESHOLDS[name])
        # (AUC-0.5)^2 -> rastgele modul 0 agirlik, guclu modul baskin
        weights[name] = max(MODULE_AUC[name] - 0.5, 1e-3) ** 2

    out = {
        "verdict": "UNKNOWN",
        "confidence": 0.0,
        "fake_probability": None,
        "calibrated": calibrated,
        "active_modules": len(calibrated),
        "module_weights": {},
    }
    if not calibrated:
        return out

    wsum = sum(weights.values())
    fused = sum(calibrated[n] * weights[n] for n in calibrated) / wsum
    verdict = "FAKE" if fused >= 0.5 else "REAL"

    # Guclu tek-modul kanit override: kacirilan sahte (false negative)
    # en maliyetli hata. Bir modul cok emin "sahte" diyorsa (kalibre >=
    # STRONG_FAKE) agirlikli ortalama gercek dese bile alarm ver.
    strong = max(calibrated.values())
    if strong >= STRONG_FAKE:
        verdict = "FAKE"

    out["verdict"] = verdict
    out["fake_probability"] = max(fused, strong) if verdict == "FAKE" else fused
    out["confidence"] = abs(out["fake_probability"] - 0.5) * 2
    out["module_weights"] = {n: weights[n] / wsum for n in calibrated}
    return out


def _eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def _find_ffmpeg() -> Optional[str]:
    import os
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


class DeepfakeDetector:
    """End-to-end deepfake detector.

    Kullanim:
        det = DeepfakeDetector()
        result = det.predict("video.mp4")
    """

    def __init__(self, config: dict | None = None,
                 checkpoint_dir: str = "checkpoints"):
        self.config = config or {}
        self.device = get_device(self.config.get("device", "auto"))
        self.checkpoint_dir = Path(checkpoint_dir)
        self.video_processor = VideoProcessor(
            frame_rate=self.config.get("frame_rate", 2),
            max_frames=self.config.get("max_frames", 30),
        )
        self.freq_analyzer = FrequencyAnalyzer()
        self.transform = _eval_transform()

        # Lazy-load
        self._face_detector: FaceDetector | None = None
        self._audio_model: AudioModel | None = None
        self._frequency_model: FrequencyModel | None = None
        self._spatial_model: SpatialModel | None = None
        self._loaded: dict[str, bool] = {}

    def _try_load_checkpoint(self, name: str, model: torch.nn.Module) -> bool:
        path = self.checkpoint_dir / Path(CHECKPOINTS[name]).name
        if not path.exists():
            return False
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        return True

    def _ensure_face_detector(self):
        if self._face_detector is None:
            # MTCNN'i CPU'da tutarak GPU VRAM'ini siniflandirici modellere birak
            self._face_detector = FaceDetector(face_size=224, device="cpu")

    def _ensure_audio(self) -> bool:
        if "audio" in self._loaded:
            return self._loaded["audio"]
        model = AudioModel(n_mels=128)
        ok = self._try_load_checkpoint("audio", model)
        self._audio_model = model if ok else None
        self._loaded["audio"] = ok
        return ok

    def _ensure_frequency(self) -> bool:
        if "frequency" in self._loaded:
            return self._loaded["frequency"]
        model = FrequencyModel(in_channels=3, feature_dim=256)
        ok = self._try_load_checkpoint("frequency", model)
        self._frequency_model = model if ok else None
        self._loaded["frequency"] = ok
        return ok

    def _ensure_spatial(self) -> bool:
        if "spatial" in self._loaded:
            return self._loaded["spatial"]
        model = SpatialModel(lstm_hidden=256, lstm_layers=2, dropout=0.3)
        ok = self._try_load_checkpoint("spatial", model)
        self._spatial_model = model if ok else None
        self._loaded["spatial"] = ok
        return ok

    @torch.no_grad()
    def _spatial_predict(self, faces: list[np.ndarray]) -> float | None:
        if not self._ensure_spatial() or not faces:
            return None
        seq_len = 8
        n = len(faces)
        if n >= seq_len:
            idx = np.linspace(0, n - 1, seq_len, dtype=int)
            sel = [faces[i] for i in idx]
        else:
            sel = faces + [faces[-1]] * (seq_len - n)
        tensors = [self.transform(Image.fromarray(f)) for f in sel]
        batch = torch.stack(tensors).unsqueeze(0).to(self.device)
        logits = self._spatial_model(batch)
        return float(torch.softmax(logits, dim=1)[0, 1].item())

    @torch.no_grad()
    def _frequency_predict(self, faces: list[np.ndarray]) -> float | None:
        if not self._ensure_frequency() or not faces:
            return None
        tensors = torch.stack([self.transform(Image.fromarray(f)) for f in faces])
        tensors = tensors.to(self.device)
        probs = []
        for i in range(0, len(tensors), 16):
            chunk = tensors[i:i + 16]
            logits = self._frequency_model(chunk)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
        return float(np.mean(probs))

    @torch.no_grad()
    def _audio_predict(self, video_path: str) -> float | None:
        if not self._ensure_audio():
            return None
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            return None
        import librosa
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "audio.wav"
                result = subprocess.run(
                    [ffmpeg, "-y", "-i", video_path,
                     "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                     str(wav)],
                    capture_output=True,
                )
                if result.returncode != 0 or not wav.exists():
                    return None  # ses kanali yok
                y, _ = librosa.load(str(wav), sr=16000, mono=True)
            mel = librosa.feature.melspectrogram(
                y=y, sr=16000, n_mels=128, n_fft=2048, hop_length=512,
            )
            log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            if log_mel.shape[1] >= 300:
                log_mel = log_mel[:, :300]
            else:
                pad = 300 - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0, 0), (0, pad)),
                                 mode="constant", constant_values=log_mel.min())
            tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self._audio_model(tensor)
            return float(torch.softmax(logits, dim=1)[0, 1].item())
        except Exception:
            return None

    def predict(self, video_path: str) -> dict:
        """End-to-end tahmin."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = {
            "video_path": video_path,
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "fake_probability": None,
            "modules": {"audio": None, "frequency": None, "spatial": None},
            "active_modules": 0,
            "frames_analyzed": 0,
            "faces_detected": 0,
        }
        try:
            frames = self.video_processor.extract_frames(video_path)
        except Exception as exc:
            result["error"] = f"Kare cikarilamadi: {exc}"
            return result
        result["frames_analyzed"] = len(frames)
        if not frames:
            result["error"] = "Videodan kare cikarilamadi"
            return result

        self._ensure_face_detector()
        faces = self._face_detector.detect_faces(frames)
        result["faces_detected"] = len(faces)

        if faces:
            result["modules"]["spatial"] = self._spatial_predict(faces)
            result["modules"]["frequency"] = self._frequency_predict(faces)
        result["modules"]["audio"] = self._audio_predict(video_path)

        # AUC-agirlikli karar fuzyonu (ortak fuse() ile fusion_eval ayni mantik)
        fused = fuse(result["modules"])
        result["calibrated"] = fused["calibrated"]
        result["active_modules"] = fused["active_modules"]
        result["module_weights"] = fused["module_weights"]
        result["verdict"] = fused["verdict"]
        result["fake_probability"] = fused["fake_probability"]
        result["confidence"] = fused["confidence"]
        return result

    # ---------- Eski API (gorsellestirme) ----------

    def extract_frames(self, video_path: str) -> list[np.ndarray]:
        return self.video_processor.extract_frames(video_path)

    def analyze_frequency(self, frame: np.ndarray) -> dict:
        fft_spectrum = self.freq_analyzer.compute_fft_spectrum(frame)
        azimuthal = self.freq_analyzer.compute_azimuthal_average(fft_spectrum)
        dct_spectrum = self.freq_analyzer.compute_dct_spectrum(frame)
        high = np.mean(azimuthal[len(azimuthal) // 2:])
        low = np.mean(azimuthal[:len(azimuthal) // 2])
        return {
            "fft_spectrum": fft_spectrum,
            "dct_spectrum": dct_spectrum,
            "azimuthal_profile": azimuthal,
            "high_freq_energy": float(high),
            "low_freq_energy": float(low),
            "freq_ratio": float(high / (low + 1e-8)),
        }

    def analyze_video(self, video_path: str, run_prediction: bool = True) -> dict:
        out = {
            "video_path": video_path,
            "frames_analyzed": 0,
            "sample_frames": [],
            "frequency_analysis": None,
            "avg_freq_ratio": 0.0,
            "std_freq_ratio": 0.0,
        }
        frames = self.extract_frames(video_path)
        out["frames_analyzed"] = len(frames)
        if not frames:
            out["error"] = "Videodan kare cikarilamadi"
            return out
        indices = [0, len(frames) // 2, len(frames) - 1]
        out["sample_frames"] = [frames[i] for i in indices if i < len(frames)]
        mid = frames[len(frames) // 2]
        out["frequency_analysis"] = self.analyze_frequency(mid)
        ratios = [self.analyze_frequency(f)["freq_ratio"]
                  for f in frames[::max(1, len(frames) // 10)]]
        out["avg_freq_ratio"] = float(np.mean(ratios))
        out["std_freq_ratio"] = float(np.std(ratios))
        if run_prediction:
            out["prediction"] = self.predict(video_path)
        return out
