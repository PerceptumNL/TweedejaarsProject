

import numpy as np

def __dotproduct(a, b):
    return sum(a.dot(b.T).data)

def __norm(a):
    return np.sqrt(__dotproduct(a,a))

def euclidean(a, b):
    return np.sqrt(sum((a - b).data ** 2))

def cosine(a, b):
    norm_a = __norm(a)
    norm_b = __norm(b)
    dot = __dotproduct(a,b)

    if norm_a * norm_b * dot == 0:
        return 1e99
    else:
        return dot / (norm_a*norm_b)
