# src/serving/spatial_handler.py

from ts.torch_handler.base_handler import BaseHandler
import torch
import onnxruntime as ort
import numpy as np
import base64, io
from PIL import Image

class SpatialDetectorHandler(BaseHandler):
    def initialize(self, context):
        self.manifest = context.manifest
        model_dir = context.system_properties.get("model_dir")
        
        self.session = ort.InferenceSession(
            f"{model_dir}/spatial_detector.onnx",
            providers=['CUDAExecutionProvider', 
                      'CPUExecutionProvider']
        )
        self.initialized = True
    
    def preprocess(self, data):
        images = []
        for item in data:
            img_bytes = item.get("data") or item.get("body")
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            arr = (arr - mean) / std
            images.append(arr.transpose(2, 0, 1))  # HWC → CHW
        return np.stack(images, axis=0).astype(np.float32)
    
    def inference(self, model_input):
        logit, embedding = self.session.run(
            None, {'image': model_input}
        )
        score = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        return score, embedding
    
    def postprocess(self, inference_output):
        scores, embeddings = inference_output
        return [{"score": float(s), 
                 "embedding": e.tolist()} 
                for s, e in zip(scores, embeddings)]