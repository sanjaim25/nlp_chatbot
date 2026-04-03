# 🤖 NeuralBot — NLP & Deep Learning Chatbot

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A conversational AI chatbot built using **Natural Language Processing (NLP)** and **Deep Learning** techniques. Features a Bidirectional LSTM neural network trained on conversational data, with a sleek modern web UI.

![Chatbot Demo](https://via.placeholder.com/800x400/1a1a2e/eaeaea?text=NeuralBot+Chat+Interface)
> *Replace the placeholder above with an actual screenshot of your chatbot interface*

---

## 📌 Project Overview

| Item | Details |
|---|---|
| **Model** | Bidirectional LSTM with Embedding + Dense layers |
| **Framework** | TensorFlow / Keras |
| **Backend** | Flask (Python) |
| **Frontend** | HTML + CSS + Vanilla JS (no framework needed) |
| **Task** | Retrieval-based intent classification |
| **Vocab** | Built dynamically from training data |
| **Metrics** | Accuracy, Loss, BLEU-1, Confidence |

---

## 🗂️ Project Structure

```
nlp_chatbot/
├── app.py                  # Flask web server
├── train.py                # Model training script
├── evaluate.py             # Evaluation & metrics script
├── requirements.txt        # Python dependencies
│
├── data/
│   └── conversations.json  # Training conversation pairs (65+)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py    # Text cleaning, tokenization, vocab builder
│   └── model.py            # LSTM model architecture + callbacks
│
├── templates/
│   └── index.html          # Chatbot web UI
│
└── models/                 # Auto-created after training
    ├── best_model.keras
    ├── preprocessor.pkl
    ├── response_mapping.json
    └── training_history.json
```

---

## ⚙️ Setup & Run — Step by Step

### Step 1 — Prerequisites

Make sure you have **Python 3.9 – 3.11** installed.

```bash
python --version
```

### Step 2 — Open in VS Code

```bash
# Clone or unzip the project, then open folder in VS Code
code nlp_chatbot
```

### Step 3 — Create a Virtual Environment

```bash
# In VS Code terminal (Ctrl + `)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This installs TensorFlow and Flask. May take 2–5 minutes.

### Step 5 — Train the Model

```bash
python train.py
```

You will see:
- Vocabulary building
- Model summary
- Training epochs with accuracy
- Saved model files in `models/`

> ✅ Training completes in ~1–2 minutes on CPU.

### Step 6 — Evaluate the Model *(optional)*

```bash
python evaluate.py
```

This prints:
- Training accuracy & loss
- BLEU-1 score
- Confidence distribution
- Sample predictions
- System limitations

### Step 7 — Start the Chatbot

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

🎉 **The chatbot is now live!**

---

## ✨ Features

- 🧹 **NLP Preprocessing** — text cleaning, tokenization, padding, vocabulary building
- 🧠 **Deep Learning Model** — Bidirectional LSTM with dropout regularisation
- 📊 **Word Embeddings** — 64-dimensional trainable embedding layer
- 🎨 **Modern UI** — dark-themed, real-time chat, confidence scores, latency display
- 📈 **Session Stats** — tracks message count, average confidence, average latency
- 🎯 **Intent Display** — shows what intent was matched per response
- 💡 **Quick Suggestions** — clickable example prompts in the sidebar
- 📉 **Evaluation Script** — BLEU-1, accuracy, confusion report

---

## 🧠 Model Architecture

```
Input (seq_len=20)
    │
    ▼
Embedding (vocab_size × 64)
    │
    ▼
Bidirectional LSTM (128 units, return_sequences=True)
    │
    ▼
GlobalAveragePooling1D
    │
    ▼
LayerNormalization
    │
    ▼
Dense(256, ReLU) → Dropout(0.3)
    │
    ▼
Dense(128, ReLU) → Dropout(0.2)
    │
    ▼
Dense(num_classes, Softmax)
```

---

## 📊 NLP Pipeline

1. **Cleaning** — lowercase, remove special characters, collapse whitespace  
2. **Tokenisation** — whitespace split  
3. **Vocabulary** — built from all training texts; maps tokens → integers  
4. **Sequence encoding** — each sentence → integer list (padded to length 20)  
5. **OOV handling** — unknown words map to `<UNK>` token  

---

## ⚠️ System Limitations

1. Limited to trained vocabulary — unknown words map to `<UNK>`
2. Closed-domain — responds best to topics seen during training
3. No dialogue memory across conversation turns
4. Small dataset (~65 pairs) limits generalisation
5. Retrieval-based — cannot generate novel sentences
6. No sentiment awareness or emotion detection
7. Context window fixed at 20 tokens

---

## 🚀 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chat UI |
| `/chat` | POST | Send message, get response |
| `/stats` | GET | Session statistics |
| `/history` | GET | Last 20 chat turns |
| `/health` | GET | Server health check |

**Example `/chat` request:**
```json
POST /chat
{ "message": "What is deep learning?" }
```

**Response:**
```json
{
  "response": "Deep learning is a subset of machine learning...",
  "confidence": 94.2,
  "intent": "what is deep learning",
  "latency_ms": 18.4
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.x / Keras |
| NLP | Custom tokenizer + vocabulary |
| Web Server | Flask 3.x |
| Frontend | HTML5 / CSS3 / JavaScript |
| Serialisation | Keras `.keras`, Python `pickle`, JSON |

---
