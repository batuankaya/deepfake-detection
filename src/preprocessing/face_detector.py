"""Yuz tespit modulu - MTCNN ile yuz kirpma ve hizalama."""

import numpy as np
import torch
from PIL import Image


class FaceDetector:
    """MTCNN tabanli yuz tespiti ve kirpma."""

    def __init__(self, face_size=224, device=None):
        from facenet_pytorch import MTCNN

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mtcnn = MTCNN(
            image_size=face_size,
            margin=40,
            device=device,
            post_process=False,
        )
        self.face_size = face_size

    def detect_faces(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Kare listesinden yuzleri tespit edip kirpar."""
        face_crops = []
        for frame in frames:
            img = Image.fromarray(frame)
            face = self.mtcnn(img)
            if face is not None:
                # (C, H, W) -> (H, W, C) ve 0-255 araligina
                face_np = face.permute(1, 2, 0).numpy().astype(np.uint8)
                face_crops.append(face_np)
        return face_crops
