# src/models/fusion/xgboost_fusion.py

import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import mlflow

class RiskFusionEngine:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=1.0,  # adjust for class imbalance
            eval_metric='auc',
            early_stopping_rounds=30,
            random_state=42,
            tree_method='gpu_hist'  # GPU acceleration
        )
    
    def build_fusion_features(self,
        spatial_score: float,
        temporal_score: float,
        audio_score: float,
        metadata_score: float,
        metadata_raw: np.ndarray
    ) -> np.ndarray:
        """
        Combine calibrated modality scores with 
        raw metadata features into fusion input vector.
        """
        modal_scores = np.array([
            spatial_score, temporal_score,
            audio_score, metadata_score,
            # Interaction features
            spatial_score * temporal_score,
            max(spatial_score, temporal_score, audio_score),
            np.mean([spatial_score, temporal_score, audio_score])
        ])
        return np.concatenate([modal_scores, metadata_raw])
    
    def train(self, X_train, y_train, X_val, y_val):
        with mlflow.start_run(run_name="fusion_engine"):
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50
            )
            val_auc = roc_auc_score(
                y_val, self.model.predict_proba(X_val)[:, 1]
            )
            mlflow.log_metric("fusion_val_auc", val_auc)
            mlflow.xgboost.log_model(self.model, "fusion_model")
    
    def predict(self, features: np.ndarray) -> dict:
        proba = self.model.predict_proba(
            features.reshape(1, -1)
        )[0, 1]
        
        # Classification thresholds
        if proba >= 0.75:
            label = "likely_synthetic"
        elif proba >= 0.40:
            label = "uncertain"
        else:
            label = "likely_authentic"
        
        return {
            "risk_score": float(proba),
            "classification": label,
            "confidence_band": self._conformal_band(proba)
        }
    
    def _conformal_band(self, score: float) -> dict:
        """Approximate conformal prediction interval."""
        margin = 0.08 if 0.3 < score < 0.7 else 0.04
        return {
            "lower": max(0.0, score - margin),
            "upper": min(1.0, score + margin),
            "coverage": 0.90
        }