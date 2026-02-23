"""
ABSA Project — Unified CLI Entry Point.
Usage:
    python main.py --model bert --domain laptops
    python main.py --model lstm --domain restaurants
    python main.py --model traditional --domain laptops
    python main.py --model all --domain laptops
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Aspect-Based Sentiment Analysis — Train & Evaluate Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model bert --domain laptops
  python main.py --model lstm --domain restaurants
  python main.py --model traditional --domain laptops
  python main.py --model all --domain laptops
  python main.py --model all --domain all
        """,
    )
    parser.add_argument(
        "--model",
        choices=["bert", "lstm", "traditional", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--domain",
        choices=["laptops", "restaurants", "all"],
        default="laptops",
        help="Which domain to train on (default: laptops)",
    )

    args = parser.parse_args()

    domains = ["laptops", "restaurants"] if args.domain == "all" else [args.domain]
    models = ["traditional", "lstm", "bert"] if args.model == "all" else [args.model]

    from config import DEVICE

    print(f"\n🔧 ABSA Training Pipeline")
    print(f"   Device: {DEVICE}")
    print(f"   Models: {', '.join(models)}")
    print(f"   Domains: {', '.join(domains)}\n")

    results = {}

    for domain in domains:
        for model_name in models:
            key = f"{model_name}_{domain}"
            print(f"\n{'#' * 60}")
            print(f"  Running: {model_name.upper()} on {domain.upper()}")
            print(f"{'#' * 60}")

            try:
                if model_name == "bert":
                    from train_bert import run_bert_training

                    results[key] = run_bert_training(domain)
                elif model_name == "lstm":
                    from train_lstm import run_lstm_training

                    results[key] = run_lstm_training(domain)
                elif model_name == "traditional":
                    from train_traditional import run_traditional_training

                    results[key] = run_traditional_training(domain)
            except Exception as e:
                print(f"\n❌ Error training {model_name} on {domain}: {e}")
                import traceback

                traceback.print_exc()
                results[key] = {"error": str(e)}

    # Summary
    print(f"\n\n{'=' * 60}")
    print("  📋 TRAINING SUMMARY")
    print(f"{'=' * 60}\n")

    for key, result in results.items():
        if isinstance(result, dict) and "error" in result:
            print(f"  ❌ {key}: {result['error']}")
        elif isinstance(result, dict) and "svm" in result:
            # Traditional ML has two sub-models
            svm = result["svm"]
            rf = result["rf"]
            print(
                f"  ✅ {key} (SVM):  Acc={svm['accuracy']:.4f}  F1={svm['macro_f1']:.4f}"
            )
            print(
                f"  ✅ {key} (RF):   Acc={rf['accuracy']:.4f}  F1={rf['macro_f1']:.4f}"
            )
        elif isinstance(result, dict):
            print(
                f"  ✅ {key}: Acc={result.get('accuracy', 'N/A'):.4f}  F1={result.get('macro_f1', 'N/A'):.4f}"
            )

    print()


if __name__ == "__main__":
    main()
