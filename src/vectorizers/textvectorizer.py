from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing

def vectorize(data, new_doc, local=False):
    vectorizer = TfidfVectorizer(use_idf=True)
    content = ('headline', 'about', 'title', 'text')    
    vectorizer.fit(data.value_for_keys(None, *content))
    data_bows = vectorizer.transform(data.value_for_keys(None, *content))

    new_doc_pre = data.value_for_keys_with_item(new_doc, *content)
    new_doc_bow = vectorizer.transform([new_doc_pre])

    if(local):
        return(zip(data.items(), data_bows), new_doc_bow, vectorizer)

    return(zip(data.items(), data_bows), new_doc_bow)