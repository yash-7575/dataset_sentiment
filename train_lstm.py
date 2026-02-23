"""
Training script for Bi-LSTM + Attention ABSA model.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    LSTM_BATCH_SIZE,
    LSTM_LR,
    LSTM_EPOCHS,
    LSTM_MAX_LEN,
    DEVICE,
    RESULTS_DIR,
    EARLY_STOP_PATIENCE,
    SEED,
)
from models.lstm_model import LSTMABSAModel, load_glove_embeddings
from utils.data_loader import load_dataset, build_vocab, create_lstm_dataloaders
from utils.helpers import (
    compute_metrics,
    EarlyStopping,
    plot_training_curves,
    plot_confusion_matrix,
    save_results_json,
)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="  Training", leave=False):
        sentence = batch["sentence"].to(device)
        aspect = batch["aspect"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(sentence, aspect)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            sentence = batch["sentence"].to(device)
            aspect = batch["aspect"].to(device)
            labels = batch["label"].to(device)

            logits = model(sentence, aspect)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_preds, all_labels


def run_lstm_training(domain="laptops"):
    """Main LSTM training pipeline."""
    print(f"\n{'=' * 60}")
    print(f"  🧠 LSTM + Attention ABSA Training — Domain: {domain.upper()}")
    print(f"  Device: {DEVICE}")
    print(f"{'=' * 60}\n")

    torch.manual_seed(SEED)

    # Build vocabulary
    print("📦 Loading data and building vocabulary...")
    train_df, _ = load_dataset(domain)
    all_texts = train_df["sentence"].tolist() + train_df["aspect_term"].tolist()
    vocab = build_vocab(all_texts)
    print(f"  Vocabulary size: {len(vocab)}")

    # Load GloVe embeddings
    pretrained_embeddings = load_glove_embeddings(vocab)

    # Create dataloaders
    train_loader, val_loader = create_lstm_dataloaders(
        domain, vocab, LSTM_BATCH_SIZE, LSTM_MAX_LEN
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = LSTMABSAModel(
        vocab_size=len(vocab),
        pretrained_embeddings=pretrained_embeddings,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # Results directory
    results_dir = os.path.join(RESULTS_DIR, "lstm", domain)
    os.makedirs(results_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_f1 = 0
    start_time = time.time()

    print("\n🚀 Starting training...\n")
    for epoch in range(LSTM_EPOCHS):
        print(f"Epoch {epoch + 1}/{LSTM_EPOCHS}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(
            f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pt"))
            print(f"  ✅ New best model saved (F1: {best_f1:.4f})")

        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
            break
        print()

    elapsed = time.time() - start_time
    print(f"\n⏱️  Training completed in {elapsed:.1f}s")

    # Final evaluation
    print("\n📊 Final Validation Results:")
    model.load_state_dict(
        torch.load(os.path.join(results_dir, "best_model.pt"), weights_only=True)
    )
    _, final_metrics, preds, labels = evaluate(model, val_loader, criterion, DEVICE)
    print(final_metrics["report"])

    # Save plots & results
    plot_training_curves(history, os.path.join(results_dir, "training_curves.png"))
    plot_confusion_matrix(
        final_metrics["confusion_matrix"],
        final_metrics["target_names"],
        os.path.join(results_dir, "confusion_matrix.png"),
    )
    save_results_json(
        {
            "model": "LSTM+Attention",
            "domain": domain,
            "accuracy": final_metrics["accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "epochs_trained": len(history["train_loss"]),
            "training_time_seconds": elapsed,
            "history": history,
        },
        os.path.join(results_dir, "results.json"),
    )

    return final_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", default="laptops", choices=["laptops", "restaurants"]
    )
    args = parser.parse_args()
    run_lstm_training(args.domain)
