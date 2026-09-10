import pytest
from src.intent.taxonomy import ALL_INTENTS, TAXONOMY, IntentName
from src.intent.classifier import EmbeddingIntentClassifier

def test_taxonomy_integrity():
    assert len(ALL_INTENTS) == 10
    for name in ALL_INTENTS:
        defn = TAXONOMY[IntentName(name)]
        assert len(defn.keywords) > 0
        assert len(defn.exemplars) > 0

def test_embedding_classifier_predictions():
    clf = EmbeddingIntentClassifier()
    texts = [
        "Where is my package? It is late.",
        "How do I return this shirt?",
        "Someone hacked my account and changed the password!"
    ]
    preds = clf.predict_with_confidence(texts)
    assert len(preds) == 3
    
    # Test intent labels
    assert preds[0][0] == IntentName.ORDER_TRACKING_AND_DELIVERY.value
    assert preds[1][0] == IntentName.RETURNS_AND_EXCHANGES.value
    assert preds[2][0] == IntentName.ACCOUNT_ACCESS_AND_SECURITY.value
    
    # Test confidence scores in [0.0, 1.0]
    for _, conf in preds:
        assert 0.0 <= conf <= 1.0
