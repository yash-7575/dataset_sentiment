"""
BERT-based Aspect-Based Sentiment Analysis model.
Uses [CLS] sentence [SEP] aspect_term [SEP] as input.
"""

import torch
import torch.nn as nn
from transformers import BertModel

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BERT_MODEL_NAME, BERT_DROPOUT, NUM_CLASSES


class BertABSA(nn.Module):
    """BERT for Aspect-Based Sentiment Analysis."""

    def __init__(
        self, model_name=BERT_MODEL_NAME, num_classes=NUM_CLASSES, dropout=BERT_DROPOUT
    ):
        super(BertABSA, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits
