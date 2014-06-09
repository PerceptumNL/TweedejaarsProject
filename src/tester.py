from sklearn.feature_extraction.text import TfidfVectorizer
from vectorizers import weighted_text_vectorizer 
from vectorizers import textvectorizer 
from vectorizers import glossaries_of_tags 

import copy
import datawrapper 

data = datawrapper.DataWrapper('../data/export_starfish_tjp.pickle')
temp = copy.deepcopy(data.data)
single = temp['items'][1]
del temp['items'][1]
print(glossaries_of_tags.vectorize(data, 1))