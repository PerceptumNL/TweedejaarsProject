
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing
import textvectorizer
from scipy import sparse

def vectorize(data, new_doc):
    data_bows, new_doc_bow, vectorizer = textvectorizer.vectorize(data, new_doc, True)
    descriptors = dict(data_bows)

    zero_vector = sparse.csc_matrix((1, len(vectorizer.get_feature_names())))
    depth = 2

    for i in range(0, depth):
        tmp_descriptors = {}

        for key in data.items():
            # Take vectors of all links and add with averaged 0.5 weight
            links = map(lambda x: descriptors[x], data.data['items'][key]['links'])
            descriptor = descriptors[key] + zero_vector +  0.5*sum(links)/(len(links) + 1) 
            tmp_descriptors[key] = descriptor
        descriptors = tmp_descriptors

    return(zip(descriptors.keys(), descriptors.values()), new_doc_bow)