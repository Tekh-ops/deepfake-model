# src/explainability/gradcam.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Register hooks on the target layer
        target_layer = dict(model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, image: torch.Tensor) -> np.ndarray:
        """
        Returns a heatmap of same spatial size as input image.
        """
        self.model.eval()
        image.requires_grad_(True)
        
        logits, _ = self.model(image.unsqueeze(0))
        
        self.model.zero_grad()
        logits.backward()
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize and resize to input resolution
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (image.shape[-1], image.shape[-2]))
        
        return cam  # values in [0, 1]
    
    def overlay_on_image(self, image: np.ndarray, 
                         cam: np.ndarray) -> np.ndarray:
        """Create a red-highlighted overlay on the original image."""
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.6 * image + 0.4 * heatmap).astype(np.uint8)
        return overlay