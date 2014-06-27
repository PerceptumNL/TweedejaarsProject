
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing
import textvectorizer
import simple_tag_similarity
from scipy import sparse

"""
Hybrid vectorizer
================================

This vectorizer is a hybrid form of the simple_tag_similarity and 
textvectorizer. If the document has zero tags, textvectorizer is
used. Otherwise, simple_tag_similarity is used. 

"""

def vectorize(data, new_doc, local = False):
    if len(new_doc['tags']) == 0:
        return textvectorizer.vectorize(data, new_doc)
    else:
        return simple_tag_similarity.vectorize(data, new_doc)