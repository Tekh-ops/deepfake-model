# src/preprocessing/metadata_extractor.py

import subprocess, json
import numpy as np
from pathlib import Path

FEATURES = [
    "FileSize", "ImageWidth", "ImageHeight",
    "VideoFrameRate", "VideoScanType",
    "AudioSampleRate", "AudioChannels",
    "CompressorID", "BitDepth",
    "CreateDate", "ModifyDate", "TrackCreateDate"
]

class MetadataExtractor:
    def extract(self, file_path: Path) -> np.ndarray:
        result = subprocess.run([
            "exiftool", "-j", str(file_path)
        ], capture_output=True, text=True)
        
        data = json.loads(result.stdout)[0]
        feature_vec = []
        
        for feat in FEATURES:
            val = data.get(feat, None)
            if val is None:
                feature_vec.append(0.0)
            elif isinstance(val, (int, float)):
                feature_vec.append(float(val))
            else:
                # Hash strings to a float in [0, 1]
                feature_vec.append(float(hash(str(val)) % 10000) / 10000.0)
        
        # Timestamp consistency check
        create = data.get("CreateDate", "")
        modify = data.get("ModifyDate", "")
        feature_vec.append(0.0 if create == modify else 1.0)
        
        return np.array(feature_vec, dtype=np.float32)