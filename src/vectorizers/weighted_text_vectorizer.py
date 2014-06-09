
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing

def vectorize(data, new_doc):
    vectorizer = TfidfVectorizer(use_idf=True)

    vectorizer.fit(data.preprocessed_texts())

    data_bows = vectorizer.transform(data.preprocessed_contents())
    new_doc_pre = preprocessing.preprocess_content(new_doc)
    new_doc_bow = vectorizer.transform(new_doc_pre)

    zero_vector = sparse.csc_matrix((1, len(vectorizer.get_feature_names()))) 

 	depth = 2
 	descriptors = data_bows
 	data_bows[new_doc] = new_doc_bow

 	for i in range(0, depth):
 		tmp_descriptor = []
    	for key in data.items:
    		
    		# Take vectors of all links and add with averaged 0.5 weight
    		links = map(lambda x: data_bows[x], data['items'][key]['links'])
    		descriptor = 0.5*sum(links)/(len(links)) + 1 + descriptors[key]
    		tmp_descriptor += [(key, descriptor)]
    	descriptors = tmp_descriptor

    new_doc_bow = descriptors[new_doc]
    del descriptors[new_doc]
 	
    return(zip(data.items, descriptors), new_doc_bow)