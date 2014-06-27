
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

def __sum(a):
    """
    Returns the sum of all elements in the vector
    """
    if issparse(a):
        return np.sum(a.data)
    else:
        return np.sum(a)

def euclidean(a, b):
    """
    Returns the euclidean distance between vector a and vector b.
    """
    if issparse(a) or issparse(b):
        return np.sqrt(sum((a - b).data ** 2))
    else:
        return np.sqrt(np.linalg.norm(a-b))

def cosine(a, b):
    """
    returns the cosine distance between vector a and vector b. Will return
    a negative value such that the maximum value has the lowest distance.
    """
    norm_a = __norm(a)
    norm_b = __norm(b)
    dot = __dotproduct(a,b)
    
    # prevent ZeroDivisionErrors
    if norm_a * norm_b == 0:
        return 1
    else:
        return np.abs((dot / float((norm_a*norm_b)))-1)

def chi_square(a, b):
    if issparse(a) or issparse(b):
        a = a.toarray()
        b = b.toarray()
    a_total = np.sum(a)
    b_total = np.sum(b)
    a_norm = a / a_total
    b_norm = b / b_total

    
    return sum([((a_norm[i] - b_norm[i])**2) / a_norm[i] for i in range(0,len(a_norm))])

def correlation(a,b):
    if issparse(a):
        a = a.toarray()
    if issparse(b):
        b = b.toarray()
    r = np.abs(np.corrcoef(a,b)[0][1]-1)/2
    if np.isnan(r):
        r = cosine(a-np.mean(a), b-np.mean(b))
    return r


def intersection(a,b):
    if issparse(a):
        a = a.toarray()
    if issparse(b):
        b = b.toarray()
    sum_a = np.sum(a)
    sum_b = np.sum(b)

    if (sum_a * sum_b) == 0:
        return 0

    a_norm = a / sum_a
    b_norm = b / sum_b

    return -sum([np.min([a_norm[i], b_norm[i]]) for i in range(0, len(a))])

def bhattacharyya(a, b):
    if issparse(a):
        a = a.toarray()
    if issparse(b):
        b = b.toarray()

    N = len(a)
    a_bar = np.sum(a) / float(N)
    b_bar = np.sum(b) / float(N)

    if a_bar * b_bar == 0:
        return 1

    return np.sqrt(1-(1/np.sqrt(a_bar*b_bar*N**2))*np.sum(np.sqrt(a*b)))

    

    




