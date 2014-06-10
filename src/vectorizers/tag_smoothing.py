from itertools import combinations
import numpy as np
from numpy.linalg import cholesky
from math import sqrt, acos


def vectorize(data, new_doc):
    item_tags = [data.item(x)['tags'] for x in data.items()]
    
    flat_tags = set([item for sublist in item_tags for item in sublist])
    (C, sorted_tags) = occurrence_frequency_matrix(item_tags)
    
    smooth_matrix = decompose(C)
    
    create_desc = lambda x: tag_set_to_vector(x, C, smooth_matrix, sorted_tags)
    
    # Create descriptors for all known documents and new document
    item_descriptors = [create_desc(tags) for tags in item_tags]
    new_doc_descriptor = create_desc(new_doc['tags'])
    
    # Asssociate document ids with descriptors and return.
    return(zip(data.items(), item_descriptors), new_doc_descriptor)
    
"""
Methods for creating a descriptor based on smoothed tags
"""
def most_similar_tag_index(tag, similarity_matrix, unique_tags):
    i = unique_tags.index(tag)
    v = similarity_matrix[i,:]
    v[i] = 0
    return v.argmax()

def tag_set_to_vector(tag_set, normal_similarity, smooth_similarity, unique_tags):
    v = np.zeros([1, smooth_similarity.shape[1]], dtype=np.matrix)
    
    for tag in tag_set:
        try:
            i = unique_tags.index(tag)
            new_v = smooth_similarity[i,:]
            if np.count_nonzero(new_v) == 0:
                i = most_similar_tag_index(tag, normal_similarity, unique_tags)
                new_v = smooth_similarity[i,:]
            v += new_v
        except Exception, e:
            print("Tag {0} not previously seen, ignoring...".format(tag))
    return np.array(v, np.float)

"""
Methods for creating the occurrence frequency matrix
"""
def all_tags(tags):
    return list(set(([item for sublist in tags for item in sublist])))

def occurence_count(item, lists):
    return len([x for x in lists if item in x])

def occurence_count_double(item1, item2, lists):
    return len([x for x in lists if (item1 in x and item2 in x)])

def occurrence_frequency(item1, item2, lists):
    f_1 = occurence_count(item1, lists)
    f_2 = occurence_count(item2, lists)
    f_12 = occurence_count_double(item1, item2, lists)
    return f_12/float(f_1+f_2 - f_12)

def occurrence_frequency_matrix(tags):
    unique_tags = all_tags(tags)
    C = np.identity(len(unique_tags))
    combi = [x for x in combinations(unique_tags, 2)]
    for (item1, item2) in combi:
        index1 = unique_tags.index(item1)
        index2 = unique_tags.index(item2)
        C[index1, index2] = occurrence_frequency(item1, item2, tags)
        C[index2, index1] = occurrence_frequency(item1, item2, tags)
    return (C, unique_tags)
    
"""
Methods for decomposing the occurence frequency matrix
"""
def modified_cholesky(C):
    L = np.zeros(C.shape)
    n = C.shape[0]
    print(n)
    for i in xrange(len(C)):
        for j in xrange(i+1):
            q = [L[i, k] * L[j, k] for k in xrange(j)]
            s = sum(q)
            if (i == j):
                square_this = C[i, i] - s
                if square_this <= 0:
                    L[i,:] = 0
                    L[:,i] = 0
                else:
                    insert = sqrt(C[i, i] - s)  
                    L[i, j] = insert
            else:
                if L[j, j] != 0:
                    L[i, j] = (1.0 / L[j, j] * (C[i, j] - s))
    return L
        
def decompose(C):
    try:
        return cholesky(C)
    except Exception, e:
        return modified_cholesky(C)