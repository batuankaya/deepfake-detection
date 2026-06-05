"""Ham video klasorlerinden data/processed/ yapisini olusturur.

FF++ icin 70/15/15 train/val/test split (deterministik, seed=42).
Celeb-DF v2 icin tum video'lar cross-dataset test'e gider.

Cikti yapisi:
    data/processed/
        train/{real,fake}/<video_id>__frame<NN>.png
        val/{real,fake}/...
        test/{real,fake}/...
        cross_celebdf/{real,fake}/...

Kullanim:
    python -m src.preprocessing.build_dataset --dry-run --limit 10
    python -m src.preprocessing.build_dataset            # tam calistir
"""

import argparse
import random
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm

from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.video_processor import VideoProcessor

SEED = 42
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

FF_REAL_SUBDIR = "original"
FF_FAKE_SUBDIRS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")

CELEB_REAL_SUBDIRS = ("Celeb-real", "YouTube-real")
CELEB_FAKE_SUBDIRS = ("Celeb-synthesis",)

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def split_videos(videos: list[Path], seed: int = SEED) -> dict[str, list[Path]]:
    """Listeyi 70/15/15 train/val/test'e ayirir (deterministik)."""
    rng = random.Random(seed)
    shuffled = videos.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def list_videos(directory: Path, exts=(".mp4", ".avi", ".mov")) -> list[Path]:
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in exts])


def process_video(video_path: Path, out_dir: Path, video_id: str,
                  processor: VideoProcessor, detector: FaceDetector,
                  jpeg_format: bool = True) -> int:
    """Tek video icin: kare cikar -> yuz tespit -> kaydet. Kaydedilen kare sayisini doner."""
    # Resume: bu video icin zaten kare uretilmis mi?
    ext = "jpg" if jpeg_format else "png"
    if out_dir.exists():
        existing = list(out_dir.glob(f"{video_id}__frame*.{ext}"))
        if existing:
            return len(existing)  # zaten islenmiş, atla

    try:
        frames = processor.extract_frames(str(video_path))
    except Exception as exc:
        print(f"  [HATA] {video_path.name}: kare cikarilamadi ({exc})")
        return 0

    if not frames:
        return 0

    faces = detector.detect_faces(frames)
    if not faces:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, face in enumerate(faces):
        img = Image.fromarray(face)
        img.save(out_dir / f"{video_id}__frame{i:03d}.{ext}",
                 quality=92 if jpeg_format else None)
    return len(faces)


def process_set(name: str, videos: list[Path], out_dir: Path, label: str,
                processor: VideoProcessor, detector: FaceDetector,
                limit: int | None = None) -> int:
    """Bir video grubunu islemekle gorevli."""
    if limit is not None:
        videos = videos[:limit]
    total_frames = 0
    pbar = tqdm(videos, desc=name, unit="video", leave=False)
    for vp in pbar:
        video_id = f"{vp.parent.name}_{vp.stem}"
        n = process_video(vp, out_dir / label, video_id, processor, detector)
        total_frames += n
        pbar.set_postfix(frames=total_frames)
    pbar.close()
    print(f"[{name}] tamamlandi: {len(videos)} video, {total_frames} kare")
    return total_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Sadece ilk birkac video ile test")
    parser.add_argument("--limit", type=int, default=None,
                        help="Set basina maksimum video sayisi (test icin)")
    parser.add_argument("--frame-rate", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--face-size", type=int, default=224)
    parser.add_argument("--skip-celeb", action="store_true",
                        help="Celeb-DF v2'yi atla")
    args = parser.parse_args()

    if args.dry_run and args.limit is None:
        args.limit = 5

    processor = VideoProcessor(frame_rate=args.frame_rate,
                               max_frames=args.max_frames)
    detector = FaceDetector(face_size=args.face_size)

    print(f"Frame rate: {args.frame_rate} fps | Max frame: {args.max_frames}")
    print(f"Face size: {args.face_size}x{args.face_size}")
    print(f"Dry-run: {args.dry_run} | Limit: {args.limit}")

    # FF++ klasor adi (eski: ff++, yeni: faceforensics — cv2 + ile sorun)
    ff_root_name = "faceforensics" if (RAW_DIR / "faceforensics").exists() else "ff++"
    print(f"FF++ klasoru: {ff_root_name}")

    # FF++ real
    ff_real_dir = RAW_DIR / ff_root_name / FF_REAL_SUBDIR
    real_videos = list_videos(ff_real_dir)
    real_splits = split_videos(real_videos)
    for split, videos in real_splits.items():
        process_set(f"FF++ real/{split}", videos,
                    PROCESSED_DIR / split, "real",
                    processor, detector, limit=args.limit)

    # FF++ fake (her manipulasyon icin ayri split)
    for fake_subdir in FF_FAKE_SUBDIRS:
        fake_dir = RAW_DIR / ff_root_name / fake_subdir
        fake_videos = list_videos(fake_dir)
        fake_splits = split_videos(fake_videos)
        for split, videos in fake_splits.items():
            process_set(f"FF++ {fake_subdir}/{split}", videos,
                        PROCESSED_DIR / split, "fake",
                        processor, detector, limit=args.limit)

    # Celeb-DF v2 -> cross_celebdf
    if not args.skip_celeb:
        for sub in CELEB_REAL_SUBDIRS:
            videos = list_videos(RAW_DIR / "celeb-df-v2" / sub)
            process_set(f"Celeb-DF {sub}", videos,
                        PROCESSED_DIR / "cross_celebdf", "real",
                        processor, detector, limit=args.limit)
        for sub in CELEB_FAKE_SUBDIRS:
            videos = list_videos(RAW_DIR / "celeb-df-v2" / sub)
            process_set(f"Celeb-DF {sub}", videos,
                        PROCESSED_DIR / "cross_celebdf", "fake",
                        processor, detector, limit=args.limit)

    print("\n=== Preprocessing tamamlandi ===")
    for split in ("train", "val", "test", "cross_celebdf"):
        d = PROCESSED_DIR / split
        if not d.exists():
            continue
        for cls in ("real", "fake"):
            cd = d / cls
            if cd.exists():
                n = sum(1 for _ in cd.glob("*.jpg")) + sum(1 for _ in cd.glob("*.png"))
                print(f"  {split}/{cls}: {n} kare")


if __name__ == "__main__":
    main()
