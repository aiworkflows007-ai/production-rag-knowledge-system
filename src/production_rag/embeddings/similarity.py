"""Mathematical similarity functions for vector comparisons.

These functions provide the foundational math for vector distance metrics
used in subsequent retrieval stages.
"""

from collections.abc import Sequence

import numpy as np


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    cos(theta) = (A . B) / (|A| * |B|)

    Returns a float in range [-1.0, 1.0]. If vectors are already L2-normalized,
    this is equivalent to dot_product.
    """
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"Vector dimensions must match for cosine similarity: {a.shape} vs {b.shape}"
        )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = np.dot(a, b)
    similarity = dot / (norm_a * norm_b)
    # Clamp to [-1.0, 1.0] to guard against floating-point inaccuracies
    return float(np.clip(similarity, -1.0, 1.0))


def dot_product(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute inner dot product between two embedding vectors.

    For L2-normalized unit vectors, dot product equals cosine similarity.
    """
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"Vector dimensions must match for dot product: {a.shape} vs {b.shape}"
        )

    return float(np.dot(a, b))


def euclidean_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute Euclidean (L2) distance between two embedding vectors.

    dist = sqrt(sum((a_i - b_i)^2))
    """
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"Vector dimensions must match for Euclidean distance: {a.shape} vs {b.shape}"
        )

    return float(np.linalg.norm(a - b))
