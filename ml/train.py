"""Train the sentence rhetorical-role classifier.

Pipeline: TF-IDF (word 1-2 grams) -> Logistic Regression.
Saves the fitted pipeline to ml/model.joblib and prints evaluation metrics.

Run from the project root:
    python -m ml.train
"""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from ml.dataset import DATA

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


def build_pipeline() -> Pipeline:
    # Word 1-2 grams + character 3-5 grams capture both keywords ("we propose")
    # and morphology ("achiev-", "conclud-"), which helps on a small dataset.
    from sklearn.pipeline import FeatureUnion
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=2000, C=6.0, class_weight="balanced")),
    ])


def main():
    texts = [t for t, _ in DATA]
    labels = [y for _, y in DATA]

    pipe = build_pipeline()

    # Stratified 5-fold cross-validation - a fair estimate on a small dataset.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, texts, labels, cv=cv)
    preds = cross_val_predict(pipe, texts, labels, cv=cv)

    print(f"Dataset: {len(texts)} sentences across {len(set(labels))} roles")
    print(f"5-fold CV accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})\n")
    print(classification_report(labels, preds, zero_division=0))

    # Fit on ALL data and persist (small dataset - use every example).
    pipe.fit(texts, labels)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
