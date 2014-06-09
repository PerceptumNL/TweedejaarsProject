
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing

def vectorize(data, new_doc):

    vectorizer = TfidfVectorizer(use_idf=True, tokenizer=tokenize)

    vectorizer.fit(data.preprocessed_texts())

    # Get all glossaries for all tags
    glossary_bows = vectorizer.transform(data.tag_glossary())
    new_doc_pre = preprocessing.preprocess_content(new_doc)
    new_doc_bow = vectorizer.transform(new_doc_pre)

    glossary_bows = dict(zip(glossaries.keys(), glossary_bows))
    zero_vector = sparse.csc_matrix((1, len(vectorizer.get_feature_names())))
    descriptors = []

    for key, tags in doc_tags:
        bows = map(lambda x: glossary_bows[x], tags)
        descriptor = (sum(bows) + zero_vector) #/ float(len(tags) + 1)
        descriptors += [(key, descriptor)]

    # Get all tags for the new document
    new_doc_descriptor = sum(map(lambda x: glossary_bows[x], new_doc['tags'])) + zero_vector

    return(zip(data.items, descriptors), new_doc_descriptor)