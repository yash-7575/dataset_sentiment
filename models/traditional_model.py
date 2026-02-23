"""
Traditional ML models (SVM + Random Forest) for ABSA.
Uses TF-IDF features with aspect-context window augmentation.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TFIDF_MAX_FEATURES, CONTEXT_WINDOW, RANDOM_STATE


def create_combined_text(sentences, aspects, context_window=CONTEXT_WINDOW):
    """Create combined feature text: full sentence + context window around aspect."""
    combined = []
    for sent, asp in zip(sentences, aspects):
        words = sent.lower().split()
        asp_lower = asp.lower()

        # Find aspect position
        asp_start = sent.lower().find(asp_lower)
        if asp_start != -1:
            # Get words before aspect start
            before_text = sent[:asp_start].lower().split()
            after_text = sent[asp_start + len(asp) :].lower().split()
            context_before = " ".join(before_text[-context_window:])
            context_after = " ".join(after_text[:context_window])
            context = f"{context_before} {asp_lower} {context_after}"
        else:
            context = asp_lower

        # Combine full sentence with localized context
        combined_text = f"{sent.lower()} [SEP] {context} [SEP] {asp_lower}"
        combined.append(combined_text)

    return combined


class TraditionalABSA:
    """Traditional ML models for ABSA using TF-IDF features."""

    def __init__(self):
        self.svm_pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=TFIDF_MAX_FEATURES,
                        ngram_range=(1, 3),
                        sublinear_tf=True,
                        min_df=2,
                    ),
                ),
                ("scaler", MaxAbsScaler()),
                (
                    "svm",
                    LinearSVC(
                        C=1.0,
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        self.rf_pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=TFIDF_MAX_FEATURES,
                        ngram_range=(1, 3),
                        sublinear_tf=True,
                        min_df=2,
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=50,
                        min_samples_split=5,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit_svm(self, texts, labels):
        """Train SVM pipeline."""
        self.svm_pipeline.fit(texts, labels)

    def fit_rf(self, texts, labels):
        """Train Random Forest pipeline."""
        self.rf_pipeline.fit(texts, labels)

    def predict_svm(self, texts):
        return self.svm_pipeline.predict(texts)

    def predict_rf(self, texts):
        return self.rf_pipeline.predict(texts)
