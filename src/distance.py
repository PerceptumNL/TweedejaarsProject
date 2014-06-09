

import numpy as np

def euclidean(a, b):
    return np.sqrt(sum((a - b).data ** 2))

def cosine(a, b):

    
    vec_a = a.toarray()
    vec_b = b.toarray()
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    dot = vec_a.dot(vec_b.T)[0][0]

    if norm_a * norm_b * dot == 0:
        return 1e99
    else:
        return dot / (norm_a*norm_b)
