
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing
from scipy import sparse

def vectorize(data, new_doc):

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

    return(descriptors, new_doc_descriptor)
