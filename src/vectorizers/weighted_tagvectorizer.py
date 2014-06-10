
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from collections import Counter
import preprocessing

"""
Weigthed tag glossary vectorizer
================================

This vectorizer created document descriptors based on the tags of each
document. The descriptor is created using the following steps.

1. For each glossary of a tag a TF-IDF bag of word vector w_i  is created
2. A weight w(T_i) for each tag T_i is calculated by taking the inverse of the 
   number of documents that are tagged T_i.
3. A descriptor for each document d_i is created by taking the sum of the
   w_i vectors of the tags associated with the document multiplied by the
   weight associated with the tag. desc(d) = w_1T_1 + w_2T_2 + ... + w_nT_2
   for all tags associated with d.
"""

def vectorize(data, new_doc):
    """
    Vectorize the data as described in file docstring.
    """
    # Generator for all glossaries
    glossaries = lambda: (data.tag_glossary(t) for t in data.tags())

    # Create the bag of words descriptors for each glossary
    vectorizer = TfidfVectorizer(use_idf=True)
    vectorizer.fit(glossaries())
    tag_bows = dict(zip(data.tags(), vectorizer.transform(glossaries())))

    # Count the number of occurences for each tag
    tag_counter = Counter()
    for i in data.items(): tag_counter.update(data.item(i)['tags'])
        
    # Generator for lists of tags for each item
    item_tags = (data.item(i)['tags'] for i in data.items())

    # The number of dimensions in the bow vector
    v_dim = len(vectorizer.get_feature_names())
    # lambda function to create descriptors
    create_desc = lambda x: create_descriptor(x, tag_bows, tag_counter, 
                                              v_dim, len(data.data['items']))

    # Create descriptors for all known documents and new document
    item_descriptors = [create_desc(tags) for tags in  item_tags]
    new_doc_descriptor = create_desc(new_doc['tags'])
    
    # Asssociate document ids with descriptors and return.
    return(zip(data.items(), item_descriptors), new_doc_descriptor)

def create_descriptor(tags, tag_bows, tag_counter, dimension, doc_count):
    s_idf = lambda x: 1 - (x/doc_count) if x != 0 else 0 # safe idf
    zero = sparse.csc_matrix((1, dimension))
    weighted_bows = [tag_bows[t] * s_idf(tag_counter[t]) for t in tags]
    zero = sparse.csc_matrix((1, dimension))
    return sum(weighted_bows + [zero])

