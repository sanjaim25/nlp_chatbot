"""
Training Script — NLP Chatbot
Run: python train.py
"""

import os
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils.preprocessing import TextPreprocessor
from utils.model import build_model


def load_data(data_path="data/conversations.json"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = [item for item in data if item["input"] != "default"]
    return data


def train():
    print("=" * 55)
    print("  NeuralBot — Training Pipeline")
    print("=" * 55)

    # ── 1. Load data ──────────────────────────────────────────
    data = load_data()

    # ── 2. FIXED: Sort responses for deterministic class mapping
    all_responses     = sorted(set(item["response"] for item in data))
    response_to_class = {resp: idx for idx, resp in enumerate(all_responses)}
    class_to_response = {idx: resp for idx, resp in enumerate(all_responses)}

    inputs  = []
    labels  = []

    for item in data:
        inputs.append(item["input"].lower().strip())
        labels.append(response_to_class[item["response"]])

    num_classes = len(all_responses)
    print(f"\n[Train] Total samples : {len(inputs)}")
    print(f"[Train] Unique classes: {num_classes}")

    # ── 3. Build vocabulary ───────────────────────────────────
    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(inputs)

    vocab_size  = preprocessor.vocab_size
    max_seq_len = preprocessor.max_seq_len

    print(f"[Train] Vocab size    : {vocab_size}")
    print(f"[Train] Seq length    : {max_seq_len}\n")

    # ── 4. Prepare sequences ──────────────────────────────────
    X = np.array([preprocessor.text_to_sequence(inp) for inp in inputs])
    y = np.array(labels)

    # ── 5. Build model ────────────────────────────────────────
    model = build_model(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embedding_dim=64,
        lstm_units=64,
        num_classes=num_classes
    )
    model.summary()

    # ── 6. Train ──────────────────────────────────────────────
    history = model.fit(
        X, y,
        epochs=300,
        batch_size=4,
        verbose=1
    )

    # ── 7. Save artefacts ─────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    model.save("models/best_model.keras")
    preprocessor.save("models/preprocessor.pkl")

    # FIXED: class_to_example_input uses sorted mapping
    class_to_example_input = {}
    for item in data:
        cls = response_to_class[item["response"]]
        if cls not in class_to_example_input:
            class_to_example_input[cls] = item["input"].lower().strip()

    mapping = {
        str(cls): {
            "input"   : class_to_example_input[cls],
            "response": class_to_response[cls]
        }
        for cls in range(num_classes)
    }

    with open("models/response_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    hist_data = {k: [float(v) for v in vals]
                 for k, vals in history.history.items()}
    with open("models/training_history.json", "w") as f:
        json.dump(hist_data, f, indent=2)

    loss, acc = model.evaluate(X, y, verbose=0)
    print("\n" + "=" * 55)
    print(f"  Training complete!")
    print(f"  Final accuracy : {acc*100:.2f}%")
    print(f"  Final loss     : {loss:.4f}")
    print("=" * 55)
    print("\n  Run  python app.py  to start the chatbot\n")


if __name__ == "__main__":
    train()