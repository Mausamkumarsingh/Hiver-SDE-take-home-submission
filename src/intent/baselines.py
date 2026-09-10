from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class MajorityClassBaseline:
    """
    Baseline 1: Always predicts the most common class observed during training.
    Serves as the foundational lower bound benchmark.
    """
    def __init__(self):
        self.majority_intent: Optional[str] = None
        self.majority_action: Optional[str] = None
        self.intent_distribution: Dict[str, float] = {}

    def fit(self, texts: List[str], intents: List[str], actions: Optional[List[str]] = None) -> MajorityClassBaseline:
        intent_counts = Counter(intents)
        total = len(intents)
        self.majority_intent = intent_counts.most_common(1)[0][0]
        self.intent_distribution = {k: v / total for k, v in intent_counts.items()}
        
        if actions:
            action_counts = Counter(actions)
            self.majority_action = action_counts.most_common(1)[0][0]
        else:
            self.majority_action = "AUTO_HANDLE"
            
        return self

    def predict_intent(self, texts: List[str]) -> List[str]:
        if not self.majority_intent:
            raise ValueError("Baseline not fitted yet.")
        return [self.majority_intent] * len(texts)

    def predict_intent_with_confidence(self, texts: List[str]) -> List[Tuple[str, float]]:
        conf = self.intent_distribution.get(self.majority_intent, 0.5)
        return [(self.majority_intent, float(conf)) for _ in texts]

    def predict_action(self, texts: List[str]) -> List[str]:
        return [self.majority_action or "AUTO_HANDLE"] * len(texts)


class TfidfLogisticRegressionBaseline:
    """
    Baseline 2: Classical Machine Learning pipeline using Sublinear TF-IDF
    (unigrams + bigrams) and Balanced Logistic Regression.
    """
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            stop_words='english'
        )
        self.intent_classifier = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=1.0,
            random_state=42
        )
        self.action_classifier = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=1.0,
            random_state=42
        )
        self.is_fitted = False
        self.classes_: List[str] = []

    def fit(
        self,
        texts: List[str],
        intents: List[str],
        actions: Optional[List[str]] = None
    ) -> TfidfLogisticRegressionBaseline:
        X = self.vectorizer.fit_transform(texts)
        self.intent_classifier.fit(X, intents)
        self.classes_ = list(self.intent_classifier.classes_)
        
        if actions:
            self.action_classifier.fit(X, actions)
            
        self.is_fitted = True
        return self

    def predict_intent(self, texts: List[str]) -> List[str]:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        X = self.vectorizer.transform(texts)
        return list(self.intent_classifier.predict(X))

    def predict_intent_with_confidence(self, texts: List[str]) -> List[Tuple[str, float]]:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        X = self.vectorizer.transform(texts)
        probs = self.intent_classifier.predict_proba(X)
        preds = self.intent_classifier.predict(X)
        
        results = []
        for pred, prob_dist in zip(preds, probs):
            class_idx = self.classes_.index(pred)
            conf = float(prob_dist[class_idx])
            results.append((pred, conf))
        return results

    def predict_action(self, texts: List[str]) -> List[str]:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        X = self.vectorizer.transform(texts)
        return list(self.action_classifier.predict(X))
