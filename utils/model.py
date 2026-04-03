"""
Deep Learning Chatbot Model
Simplified LSTM-based model for small dataset classification
"""

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense,
    Dropout, Bidirectional
)
from tensorflow.keras.optimizers import Adam


def build_model(vocab_size, max_seq_len, embedding_dim=64,
                lstm_units=64, num_classes=None):
    """
    Simplified BiLSTM classification model.
    Removed excess layers that caused overfitting on small datasets.
    """
    if num_classes is None:
        num_classes = vocab_size

    inp = Input(shape=(max_seq_len,), name="input_sequence")

    # Embedding layer
    x = Embedding(vocab_size, embedding_dim,
                  mask_zero=True, name="embedding")(inp)

    # FIXED: return_sequences=False — no pooling layer needed
    # FIXED: reduced lstm_units 128→64, reduced dropout 0.2→0.1
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=False, dropout=0.1),
        name="bi_lstm"
    )(x)

    # FIXED: single dense block, reduced size, reduced dropout
    x = Dense(128, activation="relu", name="dense_1")(x)
    x = Dropout(0.1, name="dropout_1")(x)

    # Output
    out = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="NeuralBot")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_callbacks(model_path="models/best_model.keras"):
    """Training callbacks — kept for compatibility"""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return [
        EarlyStopping(monitor="loss", patience=20,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", factor=0.5,
                          patience=10, verbose=1),
    ]