"""Video on isleme modulu - Video dosyasindan kare ve ses cikartma."""

import cv2
import numpy as np
from pathlib import Path


class VideoProcessor:
    """MP4 video dosyasini karelere ve ses kanalina ayirir."""

    def __init__(self, frame_rate=5, max_frames=150):
        self.frame_rate = frame_rate
        self.max_frames = max_frames

    def extract_frames(self, video_path: str) -> list[np.ndarray]:
        """Videodan belirli araliklarla kare cikarir."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Video acilamadi: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / self.frame_rate))
        frames = []
        frame_idx = 0

        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_idx += 1

        cap.release()
        return frames

    def extract_audio(self, video_path: str, output_path: str) -> str:
        """Videodan ses kanalini WAV olarak cikarir (ffmpeg ile)."""
        import ffmpeg

        output_path = str(Path(output_path))
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec="pcm_s16le", ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
