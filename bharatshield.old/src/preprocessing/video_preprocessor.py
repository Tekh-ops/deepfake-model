# src/preprocessing/video_preprocessor.py

import cv2
import torch
from facenet_pytorch import MTCNN
from pathlib import Path
import json

class VideoPreprocessor:
    def __init__(self, output_size=224, margin=32, fps_target=10):
        self.detector = MTCNN(
            image_size=output_size,
            margin=margin,
            keep_all=False,      # only largest face per frame
            post_process=True,   # normalize to [-1, 1]
            device='cuda'
        )
        self.output_size = output_size
        self.fps_target = fps_target

    def process_video(self, video_path: Path, output_dir: Path):
        """
        Extract aligned face crops from video at target FPS.
        Returns list of saved crop paths and a metadata dict.
        """
        cap = cv2.VideoCapture(str(video_path))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(orig_fps / self.fps_target))
        
        crops = []
        frame_idx = 0
        saved_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop = self.detector(rgb)  # returns tensor or None
                
                if crop is not None:
                    save_path = output_dir / f"frame_{saved_idx:05d}.pt"
                    torch.save(crop, save_path)
                    crops.append(str(save_path))
                    saved_idx += 1
            
            frame_idx += 1
        
        cap.release()
        
        metadata = {
            "video_path": str(video_path),
            "orig_fps": orig_fps,
            "frames_extracted": saved_idx,
            "output_size": self.output_size
        }
        return crops, metadata