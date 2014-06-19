from itertools import combinations
import numpy as np
from numpy.linalg import cholesky
from math import sqrt, acos

def vectorize(data, new_doc):
    tagDict = {}
    descriptors = []
    i = 0
    new_doc_descriptor = {}
    for tag in new_doc['tags']:
        if not tagDict.has_key(tag):
            tagDict[tag] = len(tagDict)
        new_doc_descriptor[tagDict[tag]] = 1
    
    for item in data.items():
        vector = {}
        for tag in data.item(item)['tags']:
            if not tagDict.has_key(tag):
                tagDict[tag] = len(tagDict)
            vector[tagDict[tag]] = 1
        descriptors.append(vector)

    ndescriptors = []
    for descriptor in discriptors:
        nvector = []
        for tag in tagDict:
            if descriptor.has_key(tagDict[tag]):
                nvector.append(1)
            else:
                nvector.append(0)
        ndescriptors.append(nvector)

    new_doc_descriptor = [ new_doc_descriptor.get(x) ? 1 or 0 for x in tagDict ]
    print new_doc_descriptor

    # Asssociate document ids with descriptors and return.
    return(zip(data.items(), descriptors), new_doc_descriptor)