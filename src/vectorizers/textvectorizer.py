from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing

"""
This vectorizer only takes fields from every item to create a 
descriptor. No other data is used. Currently a TF-IDF bag of words
is created from the concatenated and preprocessed values for the
about, headline, title and text field on every item.
"""

def vectorize(data, new_doc):
    keys = ('about', 'headline', 'title', 'text')
    new_doc_text = data.value_for_keys_with_item(new_doc, *keys)
    new_doc_pre = preprocessing.preprocess_text(new_doc_text)

    vectorizer = TfidfVectorizer(use_idf=True)
    vectorizer.fit(data.value_for_keys(None, *keys))

    data_bows = vectorizer.transform(data.value_for_keys(None, *keys))
    new_doc_bow = vectorizer.transform([new_doc_pre])

    return(zip(data.items(), data_bows), new_doc_bow)
