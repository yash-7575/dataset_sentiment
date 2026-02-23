"""
Data loading utilities for ABSA.
Parses XML/CSV files and provides PyTorch Dataset classes.
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LABEL2ID, VAL_SPLIT, SEED


def parse_xml(filepath):
    """Parse SemEval-2014 XML and return list of dicts with sentence, term, polarity."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    samples = []

    for sentence in root.findall("sentence"):
        text_elem = sentence.find("text")
        if text_elem is None:
            continue
        text = text_elem.text or ""

        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is not None:
            for at in aspect_terms.findall("aspectTerm"):
                term = at.get("term", "")
                polarity = at.get("polarity", None)
                from_idx = int(at.get("from", 0))
                to_idx = int(at.get("to", 0))
                sample = {
                    "sentence": text,
                    "aspect_term": term,
                    "polarity": polarity,
                    "from": from_idx,
                    "to": to_idx,
                }
                samples.append(sample)

    return samples


def load_dataset(domain="laptops"):
    """Load train and test data for a domain. Returns DataFrames."""
    from config import TRAIN_FILES, TEST_FILES

    train_samples = parse_xml(TRAIN_FILES[domain])
    test_samples = parse_xml(TEST_FILES[domain])

    train_df = pd.DataFrame(train_samples)
    test_df = pd.DataFrame(test_samples)

    # Filter only samples with valid polarity labels for training
    valid_labels = set(LABEL2ID.keys())
    train_df = train_df[train_df["polarity"].isin(valid_labels)].reset_index(drop=True)

    return train_df, test_df


def build_vocab(texts, min_freq=1):
    """Build word-to-index vocabulary from list of texts."""
    counter = Counter()
    for text in texts:
        tokens = text.lower().split()
        counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def tokenize_simple(text):
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.split()


# ============================================================
# PyTorch Dataset for BERT
# ============================================================
class BertABSADataset(Dataset):
    """Dataset for BERT-based ABSA. Pairs sentence with aspect term."""

    def __init__(self, sentences, aspects, labels, tokenizer, max_len):
        self.sentences = sentences
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        aspect = str(self.aspects[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            aspect,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# PyTorch Dataset for LSTM
# ============================================================
class LSTMABSADataset(Dataset):
    """Dataset for LSTM-based ABSA with separate sentence and aspect indices."""

    def __init__(self, sentences, aspects, labels, vocab, max_len):
        self.sentences = sentences
        self.aspects = aspects
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def _text_to_indices(self, text, max_len):
        tokens = tokenize_simple(text)[:max_len]
        indices = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        # Pad
        padding = [self.vocab["<pad>"]] * (max_len - len(indices))
        indices = indices + padding
        return indices

    def __getitem__(self, idx):
        sent_indices = self._text_to_indices(self.sentences[idx], self.max_len)
        asp_indices = self._text_to_indices(self.aspects[idx], 10)
        label = self.labels[idx]

        return {
            "sentence": torch.tensor(sent_indices, dtype=torch.long),
            "aspect": torch.tensor(asp_indices, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# DataLoader Factories
# ============================================================
def create_bert_dataloaders(domain, tokenizer, batch_size, max_len):
    """Create train, val, test DataLoaders for BERT."""
    train_df, test_df = load_dataset(domain)

    # Train/Val split
    train_data, val_data = train_test_split(
        train_df, test_size=VAL_SPLIT, random_state=SEED, stratify=train_df["polarity"]
    )

    train_labels = [LABEL2ID[p] for p in train_data["polarity"]]
    val_labels = [LABEL2ID[p] for p in val_data["polarity"]]

    train_ds = BertABSADataset(
        train_data["sentence"].tolist(),
        train_data["aspect_term"].tolist(),
        train_labels,
        tokenizer,
        max_len,
    )
    val_ds = BertABSADataset(
        val_data["sentence"].tolist(),
        val_data["aspect_term"].tolist(),
        val_labels,
        tokenizer,
        max_len,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Test loader (only if we have polarity labels)
    test_loader = None
    if "polarity" in test_df.columns and test_df["polarity"].notna().all():
        test_labels = [LABEL2ID[p] for p in test_df["polarity"]]
        test_ds = BertABSADataset(
            test_df["sentence"].tolist(),
            test_df["aspect_term"].tolist(),
            test_labels,
            tokenizer,
            max_len,
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_lstm_dataloaders(domain, vocab, batch_size, max_len):
    """Create train, val DataLoaders for LSTM."""
    train_df, test_df = load_dataset(domain)

    train_data, val_data = train_test_split(
        train_df, test_size=VAL_SPLIT, random_state=SEED, stratify=train_df["polarity"]
    )

    train_labels = [LABEL2ID[p] for p in train_data["polarity"]]
    val_labels = [LABEL2ID[p] for p in val_data["polarity"]]

    train_ds = LSTMABSADataset(
        train_data["sentence"].tolist(),
        train_data["aspect_term"].tolist(),
        train_labels,
        vocab,
        max_len,
    )
    val_ds = LSTMABSADataset(
        val_data["sentence"].tolist(),
        val_data["aspect_term"].tolist(),
        val_labels,
        vocab,
        max_len,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_traditional_data(domain):
    """Return plain lists for traditional ML models."""
    train_df, test_df = load_dataset(domain)

    train_data, val_data = train_test_split(
        train_df, test_size=VAL_SPLIT, random_state=SEED, stratify=train_df["polarity"]
    )

    return {
        "train_sentences": train_data["sentence"].tolist(),
        "train_aspects": train_data["aspect_term"].tolist(),
        "train_labels": [LABEL2ID[p] for p in train_data["polarity"]],
        "val_sentences": val_data["sentence"].tolist(),
        "val_aspects": val_data["aspect_term"].tolist(),
        "val_labels": [LABEL2ID[p] for p in val_data["polarity"]],
    }
