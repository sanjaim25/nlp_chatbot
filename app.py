"""
Flask Backend — NeuralBot Chatbot
Run: python app.py
"""

import os
import json
import time
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Globals ────────────────────────────────────────────────────
model        = None
preprocessor = None
mapping      = None
history_log  = []


def load_artefacts():
    global model, preprocessor, mapping

    import tensorflow as tf
    from utils.preprocessing import TextPreprocessor

    print("[App] Loading model…")
    model = tf.keras.models.load_model("models/best_model.keras")

    print("[App] Loading preprocessor…")
    preprocessor = TextPreprocessor()
    preprocessor.load("models/preprocessor.pkl")

    print("[App] Loading response mapping…")
    with open("models/response_mapping.json") as f:
        mapping = json.load(f)

    print("[App] All artefacts loaded ✓")


def get_response(user_input: str) -> dict:
    # FIXED: always lowercase before prediction
    clean = user_input.lower().strip()
    seq   = np.array([preprocessor.text_to_sequence(clean)])

    pred  = model.predict(seq, verbose=0)[0]
    idx   = int(np.argmax(pred))
    conf  = float(pred[idx])

    # FIXED: lowered threshold from 0.03 to 0.01
    if conf < 0.01:
        response = "I'm not sure I understand. Could you rephrase that?"
        intent   = "unknown"
    else:
        response = mapping[str(idx)]["response"]
        intent   = mapping[str(idx)]["input"]

    return {
        "response"  : response,
        "confidence": round(conf * 100, 1),
        "intent"    : intent,
        "index"     : idx
    }


# ── Routes ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    start   = time.time()
    result  = get_response(user_input)
    elapsed = round((time.time() - start) * 1000, 1)

    entry = {
        "user"      : user_input,
        "bot"       : result["response"],
        "confidence": result["confidence"],
        "intent"    : result["intent"],
        "latency_ms": elapsed
    }
    history_log.append(entry)

    return jsonify({
        "response"  : result["response"],
        "confidence": result["confidence"],
        "intent"    : result["intent"],
        "latency_ms": elapsed
    })


@app.route("/history")
def history():
    return jsonify(history_log[-20:])


@app.route("/stats")
def stats():
    if not history_log:
        return jsonify({"message": "No conversations yet"})

    avg_conf    = round(sum(h["confidence"] for h in history_log) / len(history_log), 1)
    avg_latency = round(sum(h["latency_ms"] for h in history_log) / len(history_log), 1)

    return jsonify({
        "total_messages" : len(history_log),
        "avg_confidence" : avg_conf,
        "avg_latency_ms" : avg_latency,
        "vocab_size"     : preprocessor.vocab_size,
        "model_classes"  : len(mapping)
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists("models/best_model.keras"):
        print("\n[ERROR] No trained model found!")
        print("  Run  python train.py  first, then start the app.\n")
        exit(1)

    load_artefacts()
    print("\n  NeuralBot is running at  http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)