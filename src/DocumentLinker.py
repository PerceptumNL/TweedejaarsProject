
from datawrapper import DataWrapper
import vectorizers
import distance
import sys
import itertools
import json 

class DocumentLinker(object):

    def __init__(self, datawrapper, k=10):
        """
        Initializes the document linker. Sets up a datafile and sets the
        value for k that is used by the nearest neighbor algorithm.
        """
        self.data = datawrapper
        self.k = k
        self.links = None
        self.document = None

    def get_links(self, document, vtype='textvectorizer', dtype='euclidean'):
        """
        Returns the top k proposed links of document based on the data
        that was given during initialization. Both a vectorizer type and
        a distance measure can be given.
        """
        try:
            vectorizer = getattr(__import__('vectorizers.' + vtype), vtype)
        except ImportError:
            print('Unable to load vectorizers.{0}'.format(vtype))
            sys.exit(1)

        self.document = document
        data_bows, new_doc_bow = vectorizer.vectorize(self.data, document)
        self.links = self.nearest_neighbor(data_bows, new_doc_bow, self.k, dtype)

        return self.nearest_neighbor(data_bows, new_doc_bow, self.k, dtype)

    def nearest_neighbor(self, data_vec, new_vec, k, dtype):
        """
        Naive implementation of the nearest neighbor algorihtm. Just compute
        all the distances between vectors in data_vec and new vec. Then return
        the top-k document ids and their distance.
        """
        try:
            dmeasure= getattr(distance, dtype)
        except AttributeError:
            print('Distance measure {0} does not exist'.format(dtype))

        distances = [(v[0], dmeasure(new_vec, v[1])) for v in data_vec]
        return sorted(distances, key=lambda x:x[1])[0:k]

    def formatted_links(self, filename):
        if(not self.links):
            print('First create links')
            return False
        nlinks = {}

        for link in self.links:
            title = self.data.value_for_keys(link[0], 'title', 'name')
            linktype = self.data.value_for_keys(link[0], 'type')
            content = self.data.value_for_keys(link[0], 'headline', 'about', 'title', 'text')
            correct = link in self.document['links'] 
            nlinks[link[0]] = ({'type': linktype, 'title': title, 'content': content, 'correct': correct})

        # Bring current doc in proper format
        title = self.data.value_for_keys_with_item(self.document, 'title', 'name')
        content = self.data.value_for_keys_with_item(self.document, 'headline', 'about', 'title', 'text')
        doc = {'type': self.document['type'], 'links': nlinks, 'title': title, 'content': content}
        return doc 

def run():
    data = DataWrapper('../data/export_starfish_tjp.pickle')
    filename = "../src/content.json"

    c = 0
    docs = {}
    for new_doc, datawrapper in  itertools.islice(data.test_data(), 0, 5):
        linker = DocumentLinker(datawrapper)
        linker.get_links(new_doc, vtype='textvectorizer', dtype='cosine')
        links = linker.formatted_links(filename)
        docs[c] = links
        c += 1
        print('Saved')

    file = open(filename, "w")
    file.write(json.dumps(docs))
    file.close()

if __name__ == '__main__':
    run()
