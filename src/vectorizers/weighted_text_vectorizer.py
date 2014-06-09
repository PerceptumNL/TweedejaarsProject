
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing
from scipy import sparse

def vectorize(data, new_doc):
    vectorizer = TfidfVectorizer(use_idf=True)

    vectorizer.fit(data.preprocessed_contents())

    data_bows = vectorizer.transform(data.preprocessed_contents())
    new_doc_pre = data.preprocessed_content(new_doc)
    new_doc_bow = vectorizer.transform(new_doc_pre)
    
    zero_vector = sparse.csc_matrix((1, len(vectorizer.get_feature_names())))
    depth = 2
    descriptors = dict(zip(data.data['items'].keys(), data_bows))

    for i in range(0, depth):
        tmp_descriptors = {}

        for key in data.items:
            # Take vectors of all links and add with averaged 0.5 weight
            links = map(lambda x: descriptors[x], data.data['items'][key]['links'])
            descriptor = descriptors[key] + zero_vector +  0.5*sum(links)/(len(links) + 1) 
            tmp_descriptors[key] = descriptor
        descriptors = tmp_descriptors
    
    return(descriptors, new_doc_bow)