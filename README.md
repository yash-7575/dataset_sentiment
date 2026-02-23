# ABSA — Aspect-Based Sentiment Analysis

Three models for SemEval-2014 Task 4 dataset (Laptops + Restaurants).

## Models
| Model | Architecture | Key Features |
|-------|-------------|-------------|
| **BERT** | `bert-base-uncased` | [CLS] sentence [SEP] aspect [SEP], AdamW + warmup |
| **LSTM** | Bi-LSTM + Aspect Attention | GloVe 300d embeddings, ReduceLROnPlateau |
| **Traditional** | SVM + Random Forest | TF-IDF (1-3 ngrams) + aspect context window |

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
# Train a specific model on a domain
python main.py --model bert --domain laptops
python main.py --model lstm --domain restaurants
python main.py --model traditional --domain laptops

# Train all models on all domains
python main.py --model all --domain all
```

## Dashboard
```bash
python dashboard.py
# Open http://localhost:5000
```

## Project Structure
```
├── config.py              # Hyperparameters & paths
├── main.py                # CLI entry point
├── dashboard.py           # Flask visualization dashboard
├── train_bert.py          # BERT training script
├── train_lstm.py          # LSTM training script
├── train_traditional.py   # Traditional ML training script
├── models/
│   ├── bert_model.py      # BERT classifier
│   ├── lstm_model.py      # Bi-LSTM + Attention
│   └── traditional_model.py  # SVM + RF
├── utils/
│   ├── data_loader.py     # XML parsing, Dataset classes
│   └── helpers.py         # Metrics, early stopping, plotting
├── results/               # Model outputs (auto-generated)
└── *.xml / *.csv          # Dataset files
```

## CUDA
CUDA is auto-detected. The device is printed at training start.
