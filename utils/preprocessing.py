"""
NLP Preprocessing Utilities
"""

import re
import json
import numpy as np
import pickle
import os
from collections import Counter


class TextPreprocessor:

    def __init__(self):
        self.word2idx  = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word  = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.vocab_size  = 4
        self.max_seq_len = 20

    def clean_text(self, text):
        # FIXED: strip ALL punctuation so "nlp?" → "nlp", "lstm?" → "lstm"
        text = text.lower().strip()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        return self.clean_text(text).split()

    def build_vocab(self, texts, min_freq=1):
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))

        freq = Counter(all_tokens)
        for word, count in freq.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word]          = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

        print(f"[Preprocessor] Vocabulary size: {self.vocab_size}")
        return self.word2idx

    def text_to_sequence(self, text, pad=True):
        # FIXED: explicit lowercase before tokenizing at inference time
        text   = text.lower().strip()
        tokens = self.tokenize(text)
        seq    = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        if pad:
            seq = seq[:self.max_seq_len]
            seq += [self.word2idx["<PAD>"]] * (self.max_seq_len - len(seq))
        return seq

    def sequence_to_text(self, sequence):
        words = []
        for idx in sequence:
            if idx in (0, 2, 3):
                continue
            words.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(words)

    def save(self, path="models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"[Preprocessor] Saved to {path}")

    def load(self, path="models/preprocessor.pkl"):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        print(f"[Preprocessor] Loaded from {path}")
        return self


def load_conversation_data(data_path="data/conversations.json"):
    with open(data_path, "r") as f:
        data = json.load(f)
    inputs    = [item["input"] for item in data if item["input"] != "default"]
    responses = [item["response"] for item in data if item["input"] != "default"]
    print(f"[Data] Loaded {len(inputs)} conversation pairs")
    return inputs, responses


def prepare_training_data(preprocessor, inputs, responses):
    X = np.array([preprocessor.text_to_sequence(inp) for inp in inputs])
    y_indices = []
    for resp in responses:
        tokens = preprocessor.tokenize(resp)
        idx    = tokens[0] if tokens else "<UNK>"
        y_indices.append(preprocessor.word2idx.get(idx, 1))
    y = np.array(y_indices)
    print(f"[Data] Training shapes — X: {X.shape}, y: {y.shape}")
    return X, y