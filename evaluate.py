"""
Evaluation Script — NLP Chatbot
Evaluates response quality and reports system limitations
Run: python evaluate.py
"""

import os
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from utils.preprocessing import TextPreprocessor, load_conversation_data


def compute_bleu_1(reference, hypothesis):
    """Simple BLEU-1 score (unigram precision)"""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    matches = sum(1 for t in hyp_tokens if t in ref_tokens)
    return matches / len(hyp_tokens)


def evaluate():
    print("=" * 60)
    print("  NeuralBot — Evaluation Report")
    print("=" * 60)

    # ── Load artefacts ────────────────────────────────────────
    if not os.path.exists("models/best_model.keras"):
        print("\n[ERROR] No trained model found. Run  python train.py  first.\n")
        return

    model = tf.keras.models.load_model("models/best_model.keras")

    preprocessor = TextPreprocessor()
    preprocessor.load("models/preprocessor.pkl")

    with open("models/response_mapping.json") as f:
        mapping = json.load(f)

    with open("models/training_history.json") as f:
        history = json.load(f)

    inputs, responses = load_conversation_data("data/conversations.json")

    # ── 1. Model accuracy on training set ────────────────────
    X = np.array([preprocessor.text_to_sequence(inp) for inp in inputs])
    y = np.arange(len(inputs))

    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"\n{'─'*60}")
    print(f"  Model Performance")
    print(f"{'─'*60}")
    print(f"  Training Accuracy : {acc*100:.2f}%")
    print(f"  Training Loss     : {loss:.4f}")
    print(f"  Best Val Accuracy : {max(history.get('val_accuracy', [0]))*100:.2f}%")
    print(f"  Epochs Trained    : {len(history['accuracy'])}")

    # ── 2. Per-sample response quality ───────────────────────
    print(f"\n{'─'*60}")
    print(f"  Response Quality (BLEU-1 Scores)")
    print(f"{'─'*60}")

    bleu_scores = []
    correct = 0

    preds = model.predict(X, verbose=0)
    pred_indices = np.argmax(preds, axis=1)

    for i, (inp, true_resp, pred_idx) in enumerate(zip(inputs, responses, pred_indices)):
        pred_resp = mapping[str(int(pred_idx))]["response"]
        bleu = compute_bleu_1(true_resp, pred_resp)
        bleu_scores.append(bleu)
        if pred_idx == i:
            correct += 1

    print(f"  Mean BLEU-1 Score : {np.mean(bleu_scores):.4f}")
    print(f"  Min  BLEU-1 Score : {np.min(bleu_scores):.4f}")
    print(f"  Max  BLEU-1 Score : {np.max(bleu_scores):.4f}")
    print(f"  Exact Match Rate  : {correct}/{len(inputs)} ({correct/len(inputs)*100:.1f}%)")

    # ── 3. Confidence distribution ────────────────────────────
    confidences = np.max(preds, axis=1)
    print(f"\n{'─'*60}")
    print(f"  Prediction Confidence")
    print(f"{'─'*60}")
    print(f"  Mean Confidence   : {np.mean(confidences)*100:.1f}%")
    print(f"  Low Conf (<50%)   : {(confidences < 0.5).sum()} samples")
    print(f"  High Conf (>90%)  : {(confidences > 0.9).sum()} samples")

    # ── 4. Sample predictions ─────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Sample Predictions")
    print(f"{'─'*60}")
    test_cases = [
        "hello", "what is deep learning", "tell me a joke",
        "who are you", "what is python", "bye"
    ]
    for tc in test_cases:
        seq = np.array([preprocessor.text_to_sequence(tc)])
        pred = model.predict(seq, verbose=0)
        idx  = np.argmax(pred[0])
        conf = pred[0][idx]
        resp = mapping[str(int(idx))]["response"]
        print(f"\n  Input : {tc}")
        print(f"  Output: {resp}")
        print(f"  Conf  : {conf*100:.1f}%")

    # ── 5. Limitations ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  System Limitations")
    print(f"{'─'*60}")
    limitations = [
        "1. Limited to trained vocabulary — unknown words map to <UNK>.",
        "2. Closed-domain: responds best to topics seen during training.",
        "3. No dialogue state/memory across conversation turns.",
        "4. Small dataset (~65 pairs) limits generalisation.",
        "5. Retrieval-based: cannot generate novel sentences.",
        "6. No sentiment awareness or emotion detection.",
        "7. Context window fixed at 20 tokens max.",
    ]
    for lim in limitations:
        print(f"  {lim}")

    print(f"\n{'='*60}")
    print(f"  Evaluation complete!\n")


if __name__ == "__main__":
    evaluate()
