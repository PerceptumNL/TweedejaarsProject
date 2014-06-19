from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing

"""
This vectorizer only takes fields from every item to create a 
descriptor. No other data is used. Currently a TF-IDF bag of words
is created from the concatenated and preprocessed values for the
about, headline, title and text field on every item.
"""

def vectorize(data, new_doc, local=False):
    content = ('headline', 'about', 'title', 'text')
    new_doc_pre = data.value_for_keys_with_item(new_doc, *content)

    vectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
    vectorizer.fit(data.value_for_keys(None, *content))

    data_bows = vectorizer.transform(data.value_for_keys(None, *content))
    new_doc_bow = vectorizer.transform([new_doc_pre])

    if(local):
        return(zip(data.items(), data_bows), new_doc_bow, vectorizer)

    return(zip(data.items(), data_bows), new_doc_bow)
