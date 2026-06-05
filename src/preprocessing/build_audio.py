"""ASVspoof 2019 LA + FF++ video seslerinden mel-spektrogram .npy uretir.

ASVspoof: FLAC -> mel (.npy) ; etiket protocol dosyasindan okunur (bonafide=0, spoof=1)
FF++: video -> ffmpeg wav -> mel (.npy) ; etiket klasor adindan (original=0, manipulation=1)

Cikti:
    data/processed/audio/
        train/{real,fake}/<utt_id>.npy
        val/{real,fake}/<utt_id>.npy
        test/{real,fake}/<utt_id>.npy

Kullanim:
    python -m src.preprocessing.build_audio --dry-run
    python -m src.preprocessing.build_audio
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


def _find_ffmpeg() -> str:
    """ffmpeg.exe'yi PATH'te veya bilinen yerlerde arar."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    candidates = [
        Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe"),
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "ffmpeg"  # fallback, hata aciklayici olur


FFMPEG = _find_ffmpeg()

RAW_AUDIO_DIR = Path("data/raw/asvspoof-2019-la")
_RAW = Path("data/raw")
RAW_FF_DIR = _RAW / "faceforensics" if (_RAW / "faceforensics").exists() else _RAW / "ff++"
OUT_DIR = Path("data/processed/audio")

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 300  # ~9.6 sn @ hop=512, sr=16000


def to_mel(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Waveform -> log-mel-spektrogram (n_mels, T) -> sabit T uzunluga padle/kes."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # Fix-length: (N_MELS, TARGET_FRAMES)
    if log_mel.shape[1] >= TARGET_FRAMES:
        log_mel = log_mel[:, :TARGET_FRAMES]
    else:
        pad = TARGET_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant",
                         constant_values=log_mel.min())
    return log_mel


def process_flac(flac_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(flac_path), sr=SAMPLE_RATE, mono=True)
    return to_mel(y, sr)


def process_video_audio(video_path: Path) -> np.ndarray:
    """Video'dan ffmpeg ile ses cek, mel uret."""
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        subprocess.run(
            [FFMPEG, "-y", "-i", str(video_path),
             "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE),
             str(wav_path)],
            check=True, capture_output=True,
        )
        y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    return to_mel(y, sr)


def read_protocol(path: Path) -> list[tuple[str, str]]:
    """Protocol satirlari: '<speaker> <utt_id> - - <bonafide|spoof>'."""
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                pairs.append((parts[1], parts[4]))
    return pairs


def build_asvspoof_split(split: str, protocol_file: str, flac_subdir: str,
                         limit: int | None = None) -> int:
    """ASVspoof bir split'i isle. train/dev/eval -> train/val/test'e map."""
    out_split = {"train": "train", "dev": "val", "eval": "test"}[split]
    pairs = read_protocol(RAW_AUDIO_DIR / "ASVspoof2019_LA_cm_protocols" / protocol_file)
    if limit:
        pairs = pairs[:limit]
    flac_dir = RAW_AUDIO_DIR / flac_subdir / "flac"

    n_real = n_fake = 0
    pbar = tqdm(pairs, desc=f"ASVspoof {split}", unit="utt", leave=False)
    for utt_id, label in pbar:
        flac_path = flac_dir / f"{utt_id}.flac"
        if not flac_path.exists():
            continue
        cls = "real" if label == "bonafide" else "fake"
        out_dir = OUT_DIR / out_split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"asv_{utt_id}.npy"
        if out_file.exists():
            continue
        try:
            mel = process_flac(flac_path)
            np.save(out_file, mel, allow_pickle=False)
            if cls == "real":
                n_real += 1
            else:
                n_fake += 1
            pbar.set_postfix(real=n_real, fake=n_fake)
        except Exception as exc:
            print(f"  [HATA] {utt_id}: {exc}")
    pbar.close()
    print(f"[ASVspoof {split}] real={n_real} fake={n_fake}")
    return n_real + n_fake


def build_ff_audio(limit: int | None = None) -> None:
    """FF++ original (real) + manipulasyon (fake) videolarindan ses cek.

    NOT: FF++ orijinal dataseti SESSIZ video'lar icerir (yuz odakli, ses
    kanali yok). Bu fonksiyon literatur ve veri seti acisindan opsiyonel;
    pratik olarak FF++ ses uretmez. Audio modulu icin ASVspoof yeterli.
    """
    print("[FF++ ses] UYARI: FF++ video'larinda ses kanali yok (dataset boyle "
          "tasarlanmis). Atlanıyor. Audio modulu ASVspoof ile eğitilecek.")
    return
    import random
    rng = random.Random(42)

    for split_name, ratio in [("train", 0.70), ("val", 0.15), ("test", 0.15)]:
        pass  # sirayla asagida islenecek

    # Real (original)
    real_videos = sorted((RAW_FF_DIR / "original").glob("*.mp4"))
    rng.shuffle(real_videos)
    if limit:
        real_videos = real_videos[:limit]
    n = len(real_videos)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    split_map = (["train"] * n_tr + ["val"] * n_va +
                 ["test"] * (n - n_tr - n_va))
    for vp, split in zip(real_videos, split_map):
        out_split = {"train": "train", "val": "val", "test": "test"}[split]
        out_dir = OUT_DIR / out_split / "real"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"ff_real_{vp.stem}.npy"
        if out_file.exists():
            continue
        try:
            mel = process_video_audio(vp)
            np.save(out_file, mel, allow_pickle=False)
        except Exception as exc:
            print(f"  [HATA] {vp.name}: {exc}")
    print(f"[FF++ real] {len(real_videos)} video islendi")

    # Fake (manipulasyonlar)
    for manip in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
        fake_videos = sorted((RAW_FF_DIR / manip).glob("*.mp4"))
        rng.shuffle(fake_videos)
        if limit:
            fake_videos = fake_videos[:limit]
        n = len(fake_videos)
        n_tr, n_va = int(n * 0.70), int(n * 0.15)
        split_map = (["train"] * n_tr + ["val"] * n_va +
                     ["test"] * (n - n_tr - n_va))
        for vp, split in zip(fake_videos, split_map):
            out_dir = OUT_DIR / split / "fake"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"ff_{manip}_{vp.stem}.npy"
            if out_file.exists():
                continue
            try:
                mel = process_video_audio(vp)
                np.save(out_file, mel, allow_pickle=False)
            except Exception as exc:
                print(f"  [HATA] {vp.name}: {exc}")
        print(f"[FF++ {manip}] {len(fake_videos)} video islendi")


def main():
    parser = argparse.ArgumentParser(description="Audio preprocessing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-asv", action="store_true")
    parser.add_argument("--skip-ff", action="store_true")
    args = parser.parse_args()

    if args.dry_run and args.limit is None:
        args.limit = 20

    if not args.skip_asv:
        build_asvspoof_split("train",
                             "ASVspoof2019.LA.cm.train.trn.txt",
                             "ASVspoof2019_LA_train",
                             limit=args.limit)
        build_asvspoof_split("dev",
                             "ASVspoof2019.LA.cm.dev.trl.txt",
                             "ASVspoof2019_LA_dev",
                             limit=args.limit)
        build_asvspoof_split("eval",
                             "ASVspoof2019.LA.cm.eval.trl.txt",
                             "ASVspoof2019_LA_eval",
                             limit=args.limit)

    if not args.skip_ff:
        build_ff_audio(limit=args.limit)

    print("\n=== Audio preprocessing tamamlandi ===")


if __name__ == "__main__":
    main()
