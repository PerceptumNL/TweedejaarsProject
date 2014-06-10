from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing

def vectorize(data, new_doc):
    vectorizer = TfidfVectorizer(use_idf=True)

    vectorizer.fit(data.preprocessed_contents())

    data_bows = vectorizer.transform(data.preprocessed_contents())
    new_doc_text = ' '.join([new_doc.get(k, '') for k in 
        ['about', 'headline', 'title', 'text']])
    new_doc_pre = preprocessing.preprocess_text(new_doc_text)
    new_doc_bow = vectorizer.transform([new_doc_pre])

    return(zip(data.items, data_bows), new_doc_bow)
