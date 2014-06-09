
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing
from scipy import sparse

def vectorize(data, new_doc):

    vectorizer = TfidfVectorizer(use_idf=True)

    vectorizer.fit(data.preprocessed_texts())

    print data.tag_glossary('MathematicsTestTool')

    glossaries = dict(map(lambda x: (x, data.tag_glossary(x)), data.tags))


    # Get all glossaries for all tags
    glossary_bows = vectorizer.transform(glossaries)

    new_doc_pre = data.preprocessed_content(new_doc)
    new_doc_bow = vectorizer.transform(new_doc_pre)

    glossary_bows = dict(zip(glossaries.keys(), glossary_bows))
    zero_vector = sparse.csc_matrix((1, len(vectorizer.get_feature_names())))
    descriptors = []

    doc_tags = map(lambda x: (x[0], x[1]['tags']), data.data['items'].items())
    for key, tags in doc_tags:
        bows = map(lambda x: glossary_bows[x], tags)
        descriptor = (sum(bows) + zero_vector) #/ float(len(tags) + 1)
        descriptors += [(key, descriptor)]

    # Get all tags for the new document
    new_doc_descriptor = sum(map(lambda x: glossary_bows[x], data.item(new_doc)['tags'])) + zero_vector

    return(zip(data.items, descriptors), new_doc_descriptor)