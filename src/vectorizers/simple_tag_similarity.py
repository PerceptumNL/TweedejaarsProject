from itertools import combinations
import numpy as np
from numpy.linalg import cholesky
from math import sqrt, acos

"""
Simple tag vectorizer
=====================

First every tag is coupled with a index into the tag vectors for each document
then for each document a vector is created where a 1 is added to the indexes
of the tags associated with the document.

This vector can for example be used with KNN 
"""

def vectorize(data, new_doc):
    # Find all tags
    tagDict = {}
    descriptors = []
    i = 0
    new_doc_descriptor = {}
    for tag in new_doc['tags']:
        if not tagDict.has_key(tag):
            tagDict[tag] = len(tagDict)
        new_doc_descriptor[tagDict[tag]] = 1
    
    # Create vector for each document
    for item in data.items():
        vector = {}
        for tag in data.item(item)['tags']:
            if not tagDict.has_key(tag):
                tagDict[tag] = len(tagDict)
            vector[tagDict[tag]] = 1
        descriptors.append(vector)
    
    ndescriptors = []

    for descriptor in descriptors:
        nvector = []
        for tag in tagDict:
            if descriptor.has_key(tagDict[tag]):
                nvector.append(1)
            else:
                nvector.append(0)
        ndescriptors.append(np.matrix(nvector))
    nnew_doc_descriptor = []

    for tag in tagDict:
        if new_doc_descriptor.has_key(tagDict[tag]):
            nnew_doc_descriptor.append(1)
        else:
            nnew_doc_descriptor.append(0)
    nnew_doc_descriptor = np.matrix(nnew_doc_descriptor)
    # Asssociate document ids with descriptors and return.
    return(zip(data.items(), ndescriptors), nnew_doc_descriptor)
