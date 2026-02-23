"""
Bi-LSTM with Aspect-Aware Attention for ABSA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LSTM_EMBED_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    NUM_CLASSES,
    GLOVE_PATH,
)


class AspectAttention(nn.Module):
    """Attention mechanism that uses the aspect representation as query."""

    def __init__(self, hidden_dim):
        super(AspectAttention, self).__init__()
        self.W = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, hidden_states, aspect_repr):
        """
        hidden_states: (batch, seq_len, hidden*2)
        aspect_repr:   (batch, hidden*2)
        """
        seq_len = hidden_states.size(1)
        aspect_repr = aspect_repr.unsqueeze(1).expand(-1, seq_len, -1)

        combined = hidden_states + aspect_repr
        energy = torch.tanh(self.W(combined))
        attention_scores = self.v(energy).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)

        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, attention_weights


class LSTMABSAModel(nn.Module):
    """Bi-LSTM + Aspect Attention for Aspect-Based Sentiment Analysis."""

    def __init__(
        self,
        vocab_size,
        embed_dim=LSTM_EMBED_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        num_classes=NUM_CLASSES,
        pretrained_embeddings=None,
    ):
        super(LSTMABSAModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.aspect_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = AspectAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, sentence, aspect):
        # Sentence encoding
        sent_embed = self.embedding(sentence)
        sent_output, _ = self.lstm(sent_embed)  # (batch, seq_len, hidden*2)

        # Aspect encoding
        asp_embed = self.embedding(aspect)
        _, (asp_hidden, _) = self.aspect_lstm(asp_embed)
        # Concatenate forward and backward hidden states
        asp_repr = torch.cat(
            [asp_hidden[-2], asp_hidden[-1]], dim=1
        )  # (batch, hidden*2)

        # Attention
        context, attn_weights = self.attention(sent_output, asp_repr)

        # Classification
        output = self.dropout(context)
        logits = self.fc(output)
        return logits


def load_glove_embeddings(vocab, embed_dim=LSTM_EMBED_DIM, glove_path=GLOVE_PATH):
    """Load GloVe vectors for words in vocab. Returns numpy array."""
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim))
    embeddings[0] = np.zeros(embed_dim)  # <pad>

    if not os.path.exists(glove_path):
        print(f"  ⚠️  GloVe file not found at {glove_path}")
        print(
            "  ℹ️  Using random embeddings. Download glove.6B.300d.txt for better results."
        )
        return embeddings.astype(np.float32)

    print("  📥 Loading GloVe embeddings...")
    loaded = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                vector = np.array(parts[1:], dtype=np.float32)
                if len(vector) == embed_dim:
                    embeddings[vocab[word]] = vector
                    loaded += 1

    print(f"  ✅ Loaded {loaded}/{len(vocab)} GloVe vectors")
    return embeddings.astype(np.float32)
