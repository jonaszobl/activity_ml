# src/model/classifier.py
import json
import numpy as np


# ---------- Math / small helpers ----------
def softmax(z):
    """
    Simple softmax function.
    I subtract the max value first so it is a bit more numerically stable.
    """
    z = np.asarray(z, float)
    m = np.max(z)
    e = np.exp(z - m)
    s = np.sum(e)
    # if everything is zero or something weird happens, I just return a uniform vector
    return e / s if s > 0 else np.ones_like(e) / len(e)


# ---------- Load trained model ----------
def load_model(path="artifacts/model.json"):
    """
    Tiny loader for the model JSON that I saved after training.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Features -> class probabilities ----------
def predict_features(X: np.ndarray, M: dict):
    """
    Take feature matrix X and a model dict M and return:
    - the predicted class index for each row
    - the full probability matrix (softmax output)

    I try to apply exactly the same normalization as during training.
    """
    # normalization step (same as in training: (x - mean) / scale)
    mean = np.asarray(M["scaler_mean"], float)
    scale = np.asarray(M["scaler_scale"], float)
    Xn = (X - mean) / scale

    # weights and bias from logistic regression
    W = np.asarray(M["W"], float)   # shape: [num_classes, num_features]
    b = np.asarray(M["b"], float)   # shape: [num_classes]

    # linear part: Xn * W^T + b  -> logits
    logits = Xn @ W.T + b          # shape: [num_samples, num_classes]

    # apply softmax row-wise to get probabilities per class
    probs = np.apply_along_axis(softmax, 1, logits)

    # predicted class is just the argmax over the probability vector
    cls_idx = np.argmax(probs, axis=1)

    return cls_idx, probs
