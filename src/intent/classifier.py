from __future__ import annotations
import os
import joblib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from src.intent.taxonomy import ALL_INTENTS, TAXONOMY, IntentName

class EmbeddingIntentClassifier:
    """
    Production-grade Intent Classifier using Sentence-Transformers (all-MiniLM-L6-v2)
    and a regularized linear probe on L2-normalized embeddings.
    Provides semantic generalization across phrasing variations, typos, and syntax.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model_path: Optional[str] = None):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=1.5,
            random_state=42
        )
        self.is_fitted = False
        self.classes_: List[str] = []
        self.prototype_embeddings: Optional[np.ndarray] = None
        self.prototype_labels: List[str] = []
        
        # Precompute canonical exemplar prototypes for zero-shot fallback
        self._init_exemplar_prototypes()
        
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def _init_exemplar_prototypes(self):
        """Builds exemplar prototype centroids from the taxonomy definitions."""
        texts = []
        labels = []
        for name, defn in TAXONOMY.items():
            for ex in defn.exemplars:
                texts.append(ex)
                labels.append(name.value)
            # Include definition description as an anchor
            texts.append(defn.description)
            labels.append(name.value)
            
        embs = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.prototype_embeddings = embs
        self.prototype_labels = labels

    def fit(self, texts: List[str], intents: List[str]) -> EmbeddingIntentClassifier:
        """Trains linear probe on sentence embeddings."""
        print(f"Encoding {len(texts)} training texts with {self.model_name}...")
        X = self.encoder.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        print("Fitting regularized linear probe...")
        self.classifier.fit(X, intents)
        self.classes_ = list(self.classifier.classes_)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts intent labels for given texts."""
        res = self.predict_with_confidence(texts)
        return [r[0] for r in res]

    def predict_with_confidence(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predicts intent and calibrated confidence score in [0.0, 1.0].
        If linear probe is not fitted, falls back to prototype cosine similarity.
        """
        if not texts:
            return []
            
        X = self.encoder.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        
        if self.is_fitted:
            probs = self.classifier.predict_proba(X)
            preds = self.classifier.predict(X)
            results = []
            for pred, prob_dist in zip(preds, probs):
                class_idx = self.classes_.index(pred)
                conf = float(prob_dist[class_idx])
                results.append((pred, round(conf, 4)))
            return results
        else:
            # Fallback: Cosine similarity to exemplar prototypes
            sims = np.dot(X, self.prototype_embeddings.T) # shape: (N, num_prototypes)
            results = []
            for i in range(len(texts)):
                row_sims = sims[i]
                best_proto_idx = int(np.argmax(row_sims))
                pred_label = self.prototype_labels[best_proto_idx]
                sim_score = float(row_sims[best_proto_idx])
                # Scale cosine similarity [-1, 1] to a calibrated probability [0, 1]
                conf = max(0.0, min(1.0, (sim_score + 1.0) / 2.0))
                results.append((pred_label, round(conf, 4)))
            return results

    def save(self, filepath: str):
        """Serializes classifier state to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classifier": self.classifier,
            "classes": self.classes_,
            "is_fitted": self.is_fitted
        }
        joblib.dump(payload, filepath)
        print(f"Saved intent classifier to {filepath}")

    def load(self, filepath: str):
        """Loads serialized classifier state."""
        payload = joblib.load(filepath)
        self.classifier = payload["classifier"]
        self.classes_ = payload["classes"]
        self.is_fitted = payload["is_fitted"]
        print(f"Loaded intent classifier from {filepath}")
