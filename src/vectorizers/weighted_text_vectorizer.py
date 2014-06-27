
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing
import textvectorizer
from scipy import sparse

"""
Weigthed tag vectorizer
=======================

Create a TF-IDF vectorizer of each document. Also for each document run the
weighted tag vectorizer with depth - 1. This way a descriptor for a
part of the network is created assuming that documents in the neighborghood
say something about the document in question.
"""

def vectorize(data, new_doc, local = False):
    data_bows, new_doc_bow, vectorizer = textvectorizer.vectorize(data, new_doc, True)
    descriptors = dict(data_bows)

    # create a zero vector
    zero_vector = sparse.csc_matrix((1, len(vectorizer.get_feature_names())))
    depth = 2

    # Create descriptors for documents in network
    for i in range(0, depth):
        tmp_descriptors = {}

        for key in data.items():
            # Take vectors of all links and add with averaged 0.5 weight
            try:
                links = map(lambda x: descriptors[x], data.data['items'][key]['links'])
            except KeyError as e:
                print('Unexpected error')
                continue
            descriptor = descriptors[key] + zero_vector +  0.5*sum(links)/(len(links) + 1) 
            tmp_descriptors[key] = descriptor
        descriptors = tmp_descriptors

    if (local):
        return zip(descriptors.keys(), descriptors.values()), new_doc_bow, vectorizer
        
    return(zip(descriptors.keys(), descriptors.values()), new_doc_bow)
