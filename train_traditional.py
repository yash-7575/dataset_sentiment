"""
Training script for Traditional ML ABSA models (SVM + Random Forest).
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, SEED
from models.traditional_model import TraditionalABSA, create_combined_text
from utils.data_loader import get_traditional_data
from utils.helpers import (
    compute_metrics,
    plot_confusion_matrix,
    save_results_json,
)


def run_traditional_training(domain="laptops"):
    """Main Traditional ML training pipeline."""
    print(f"\n{'=' * 60}")
    print(f"  📊 Traditional ML ABSA Training — Domain: {domain.upper()}")
    print(f"{'=' * 60}\n")

    # Load data
    print("📦 Loading data...")
    data = get_traditional_data(domain)
    print(f"  Train samples: {len(data['train_sentences'])}")
    print(f"  Val samples:   {len(data['val_sentences'])}")

    # Create feature texts
    print("\n🔧 Extracting features...")
    train_texts = create_combined_text(data["train_sentences"], data["train_aspects"])
    val_texts = create_combined_text(data["val_sentences"], data["val_aspects"])

    results_dir = os.path.join(RESULTS_DIR, "traditional", domain)
    os.makedirs(results_dir, exist_ok=True)

    model = TraditionalABSA()

    # ── SVM ──
    print("\n🔹 Training SVM...")
    start = time.time()
    model.fit_svm(train_texts, data["train_labels"])
    svm_time = time.time() - start

    svm_preds = model.predict_svm(val_texts)
    svm_metrics = compute_metrics(data["val_labels"], svm_preds)

    print(f"\n  SVM Results (trained in {svm_time:.2f}s):")
    print(
        f"  Accuracy: {svm_metrics['accuracy']:.4f} | Macro-F1: {svm_metrics['macro_f1']:.4f}"
    )
    print(svm_metrics["report"])

    plot_confusion_matrix(
        svm_metrics["confusion_matrix"],
        svm_metrics["target_names"],
        os.path.join(results_dir, "svm_confusion_matrix.png"),
    )
    save_results_json(
        {
            "model": "SVM (LinearSVC)",
            "domain": domain,
            "accuracy": svm_metrics["accuracy"],
            "macro_f1": svm_metrics["macro_f1"],
            "training_time_seconds": svm_time,
        },
        os.path.join(results_dir, "svm_results.json"),
    )

    # ── Random Forest ──
    print("\n🔹 Training Random Forest...")
    start = time.time()
    model.fit_rf(train_texts, data["train_labels"])
    rf_time = time.time() - start

    rf_preds = model.predict_rf(val_texts)
    rf_metrics = compute_metrics(data["val_labels"], rf_preds)

    print(f"\n  Random Forest Results (trained in {rf_time:.2f}s):")
    print(
        f"  Accuracy: {rf_metrics['accuracy']:.4f} | Macro-F1: {rf_metrics['macro_f1']:.4f}"
    )
    print(rf_metrics["report"])

    plot_confusion_matrix(
        rf_metrics["confusion_matrix"],
        rf_metrics["target_names"],
        os.path.join(results_dir, "rf_confusion_matrix.png"),
    )
    save_results_json(
        {
            "model": "Random Forest",
            "domain": domain,
            "accuracy": rf_metrics["accuracy"],
            "macro_f1": rf_metrics["macro_f1"],
            "training_time_seconds": rf_time,
        },
        os.path.join(results_dir, "rf_results.json"),
    )

    print(f"\n{'=' * 60}")
    print(f"  ✅ Traditional ML training complete!")
    print(f"{'=' * 60}")

    return {"svm": svm_metrics, "rf": rf_metrics}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", default="laptops", choices=["laptops", "restaurants"]
    )
    args = parser.parse_args()
    run_traditional_training(args.domain)
