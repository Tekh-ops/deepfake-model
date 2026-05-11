# src/explainability/shap_explainer.py

import shap
import numpy as np
import matplotlib.pyplot as plt

class FusionExplainer:
    def __init__(self, fusion_model, feature_names: list):
        self.explainer = shap.TreeExplainer(fusion_model.model)
        self.feature_names = feature_names
    
    def explain(self, features: np.ndarray) -> dict:
        shap_values = self.explainer.shap_values(
            features.reshape(1, -1)
        )
        
        contributions = {
            name: float(val) 
            for name, val in zip(
                self.feature_names, shap_values[0]
            )
        }
        
        # Sort by absolute contribution
        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            "top_features": sorted_contribs[:5],
            "base_value": float(self.explainer.expected_value),
            "all_contributions": contributions
        }
    
    def generate_waterfall_plot(self, features: np.ndarray,
                                 save_path: str):
        shap_values = self.explainer(features.reshape(1, -1))
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()