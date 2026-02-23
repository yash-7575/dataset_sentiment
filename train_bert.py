"""
Training script for BERT-based ABSA model.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    BERT_MODEL_NAME,
    BERT_MAX_LEN,
    BERT_BATCH_SIZE,
    BERT_LR,
    BERT_EPOCHS,
    BERT_WARMUP_RATIO,
    DEVICE,
    RESULTS_DIR,
    EARLY_STOP_PATIENCE,
    SEED,
)
from models.bert_model import BertABSA
from utils.data_loader import create_bert_dataloaders
from utils.helpers import (
    compute_metrics,
    EarlyStopping,
    plot_training_curves,
    plot_confusion_matrix,
    save_results_json,
)


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="  Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_preds, all_labels


def run_bert_training(domain="laptops"):
    """Main BERT training pipeline."""
    print(f"\n{'=' * 60}")
    print(f"  🤖 BERT ABSA Training — Domain: {domain.upper()}")
    print(f"  Device: {DEVICE}")
    print(f"{'=' * 60}\n")

    torch.manual_seed(SEED)

    # Tokenizer & Data
    print("📦 Loading data and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_loader, val_loader, test_loader = create_bert_dataloaders(
        domain, tokenizer, BERT_BATCH_SIZE, BERT_MAX_LEN
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = BertABSA().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)

    total_steps = len(train_loader) * BERT_EPOCHS
    warmup_steps = int(total_steps * BERT_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # Results directory
    results_dir = os.path.join(RESULTS_DIR, "bert", domain)
    os.makedirs(results_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_f1 = 0
    start_time = time.time()

    print("\n🚀 Starting training...\n")
    for epoch in range(BERT_EPOCHS):
        print(f"Epoch {epoch + 1}/{BERT_EPOCHS}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE
        )
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(
            f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}"
        )

        # Save best model
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
            "model": "BERT",
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
    run_bert_training(args.domain)
