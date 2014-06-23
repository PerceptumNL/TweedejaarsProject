from sklearn.feature_extraction.text import TfidfVectorizer
from decimal import Decimal
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

    probabilities = calcProbabilities(data)
    combined = zip(data.items(), data_bows)

    for vector in combined:
        if not probabilities.has_key(new_doc['type']) or \
        not probabilities[new_doc['type']][1].has_key(data.item(vector[0])['type']):
            weight = 0
        else:
            weight = probabilities[new_doc['type']][1][data.item(vector[0])['type']]
        combined[combined.index(vector)] = (vector[0], weight*vector[1])

    if(local):
        return(combined, new_doc_bow, vectorizer)

    return(combined, new_doc_bow)

def calcProbabilities(data):
    probabilities = {}
    for item in data.items():
        item_type = data.item(item)['type']
        if not probabilities.has_key(item_type):
            probabilities[item_type] = [0, {}]
        probabilities[item_type][0] += len(data.item(item)['links'])
        for link in data.item(item)['links']:
            try:
                link_type = data.item(link)['type']
            except KeyError:
                print('Could not find link {0} for {1}'.format(link, item))
                continue
            if not probabilities[item_type][1].has_key(link_type):
                probabilities[item_type][1][link_type] = 1
            else:
                probabilities[item_type][1][link_type] += 1
    for fromtype in probabilities:
        for totype in probabilities[fromtype][1]:
            probabilities[fromtype][1][totype] /= float(probabilities[fromtype][0])
    return probabilities
