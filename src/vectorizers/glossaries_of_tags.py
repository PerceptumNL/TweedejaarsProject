
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing
from scipy import sparse

"""
Glossaries of tags vectorizer
=============================

This vectorizer created document descriptors based on the tags of each
document. The descriptor is created using the following steps.

1. For each glossary of a tag a TF-IDF bag of word vector w_i  is created
2. For each documents a descriptor is made by summing the w_i vectors for
   each tag associated with the document
"""

def vectorize(data, new_doc, local=False):
    """
    Converts data and new doc to vectors that can be used in KNN
    """

    vectorizer = TfidfVectorizer(use_idf=True)
    glossaries = dict(map(lambda x: (x, data.tag_glossary(x)), data.tags()))
    vectorizer.fit(glossaries.values())

    # Get all glossaries for all tags
    glossary_bows = vectorizer.transform(glossaries.values())
    glossary_bows = dict(zip(glossaries.keys(), glossary_bows))

    zero_vector = sparse.csc_matrix((1, len(vectorizer.get_feature_names())))
    descriptors = []

    doc_tags = map(lambda x: (x[0], x[1]['tags']), data.data['items'].items())

    for key, tags in doc_tags:
        bows = map(lambda x: glossary_bows[x], tags)
        descriptor = (sum(bows) + zero_vector) #/ float(len(tags) + 1)
        descriptors += [(key, descriptor)]

    # Get all tags for the new document
    new_doc_descriptor = sum(map(lambda x: glossary_bows[x], new_doc['tags'])) + zero_vector

    if(local):
        return(descriptors, new_doc_descriptor, vectorizer)

    return(descriptors, new_doc_descriptor)
