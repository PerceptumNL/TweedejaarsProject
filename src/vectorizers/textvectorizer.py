
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing

def vectorize(data, new_doc):
    vectorizer = TfidfVectorizer(use_idf=True)

    vectorizer.fit(data.preprocessed_texts())

    data_bows = vectorizer.transform(data.preprocessed_texts())
    new_doc_pre = preprocessing.preprocess_text(new_doc)
    new_doc_bow = vectorizer.transform(new_doc_pre)

    return(zip(data.items, data_bows), new_doc_bow)

