
"""
This file implements several distance measures that can be used to compare
vectors in for example the nearest neighbor algorithm.
"""

import numpy as np
from scipy.sparse import issparse

def __dotproduct(a, b):
    """
    Returns the doc product of a and b. Also works on sparse matrices and
    allways returns a float instead of 1x1 matrix.
    """
    if issparse(a) and issparse(b):
        return sum(a.dot(b.T).data)
    else:
        return np.inner(a, b)

def __norm(a):
    """
    Returns the norm of vector a. Also works on sparse matrices. 
    """
    return np.sqrt(__dotproduct(a,a))

def euclidean(a, b):
    """
    Returns the euclidean distance between vector a and vector b.
    """
    return np.sqrt(sum((a - b).data ** 2))

def cosine(a, b):
    """
    returns the cosine distance between vector a and vector b. Will return
    a negative value such that the maximum value has the lowest distance.
    """
    norm_a = __norm(a)
    norm_b = __norm(b)
    dot = __dotproduct(a,b)
    
    # prevent ZeroDivisionErrors
    if norm_a * norm_b * dot == 0:
        return 0
    else:
        return -np.abs(dot / (norm_a*norm_b))
