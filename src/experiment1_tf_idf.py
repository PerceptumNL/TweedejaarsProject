from __future__ import with_statement
import cPickle as pickle
import nltk
from BeautifulSoup import BeautifulSoup
import re
from unidecode import unidecode
import numpy as np
from scipy import sparse
import copy

def preprocess(text):
    """
    Preproccess text. This is described in the 'preprocessing' notebook.
    """
    html_stripped = nltk.clean_html(text)
    # regex to also decode hex entities
    hexentityMassage = [(re.compile('&#x([^;]+);'), lambda m: '&#%d;' % int(m.group(1), 16))]
    glossary_tag = BeautifulSoup(html_stripped, convertEntities=BeautifulSoup.HTML_ENTITIES, markupMassage=hexentityMassage)
    glossary_unicode = glossary_tag.text.encode('utf-8')
    return unidecode(glossary_unicode)
    
def glos(tag):
    """
    Given a tag return the preprocessed glossary
    """
    try:
        if tag[1]['glossary'] is not None:
            return preprocess(data['items'][tag[1]['glossary']]['text'])
        else:
            return ''
    except:
        return ''

def find_links(data_, new_doc_idx):
    # Remove a document from the dataset to use as 'new document'
    data = copy.deepcopy(data_)
    new_doc = data['items'][new_doc_idx]

    del data['items'][new_doc_idx]

    doc_tags = map(lambda x: (x[0], x[1]['tags']), data['items'].items())

    # Get all glossaries for all tags
    glossaries = dict(map(lambda x: (x[0], glos(x)), data['tags'].items()))

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(use_idf=True)
    vectorizer.fit(glossaries.values())
    glossary_bows = vectorizer.transform(glossaries.values())
    glossary_bows = dict(zip(glossaries.keys(), glossary_bows))

    zero_vector = sparse.csc_matrix((1, len(vectorizer.get_feature_names())))
    descriptors = []
    for key, tags in doc_tags:
        bows = map(lambda x: glossary_bows[x], tags)
        descriptor = (sum(bows) + zero_vector) #/ float(len(tags) + 1)
        descriptors += [(key, descriptor)]

    # Get all tags for the new document
    new_doc_descriptor = sum(map(lambda x: glossary_bows[x], new_doc['tags'])) + zero_vector

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors()
    nn.fit(sparse.vstack(dict(descriptors).values()))
    dist, idx = nn.kneighbors(new_doc_descriptor, 40)

    proposed_links = map(lambda x: descriptors[x][0], idx[0])

    links = []
    for link in new_doc['links']:
        if link in proposed_links:
            links += [(link, proposed_links.index(link))]
        else:
            links += [(link, None)]
    return links

def run(data):
    for k in data['items'].keys():
        print('key: {0}'.format(k))
        print(find_links(data, k))
        print('--')

if __name__ == '__main__':
    with open('../data/export_starfish_tjp.pickle') as f:
            data = pickle.load(f)
    run(data)
