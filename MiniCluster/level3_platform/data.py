"""Synthetic 'Student Performance' dataset.

Rows block-wise deterministic generate hoti hain, is liye 1 worker ho ya 3,
poora dataset bilkul wahi rehta hai (har worker apna hissa khud generate karta hai,
data network par nahi bheja jata).
"""
import numpy as np

FEATURES = ["matric", "intermediate", "entry_test", "attendance", "study_hours", "prev_gpa"]
D = len(FEATURES)
TRUE_W = np.array([3.0, 4.5, 6.0, 2.5, 3.5, 5.0])
TRUE_B = 50.0
BLOCK = 10_000


def generate(start, end, seed=42):
    """Rows [start, end) return karta hai: X (rows x D), y (rows)."""
    if end <= start:
        return np.zeros((0, D)), np.zeros(0)
    Xs, ys = [], []
    for b in range(start // BLOCK, (end - 1) // BLOCK + 1):
        rng = np.random.default_rng(seed * 100003 + b)
        X = rng.standard_normal((BLOCK, D))
        y = X @ TRUE_W + TRUE_B + rng.standard_normal(BLOCK) * 2.0
        lo = max(start, b * BLOCK) - b * BLOCK
        hi = min(end, (b + 1) * BLOCK) - b * BLOCK
        Xs.append(X[lo:hi])
        ys.append(y[lo:hi])
    return np.vstack(Xs), np.concatenate(ys)
