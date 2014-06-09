
from datawrapper import DataWrapper
import vectorizers
import distance
import sys

class DocumentLinker(object):

    def __init__(self, datafile, k=10):
        """
        Initializes the document linker. Sets up a datafile and sets the
        value for k that is used by the nearest neighbor algorithm.
        """
        self.data = DataWrapper(datafile)
        self.k = k

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

        data_bows, new_doc_bow = vectorizer.vectorize(self.data, document)
        print(self.nearest_neighbor(data_bows, new_doc_bow, self.k, dtype))

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

if __name__ == '__main__':
    new_doc = 'dit is een nieuw document Learning Analytics'
    linker = DocumentLinker('../data/export_starfish_tjp.pickle')
    linker.get_links(new_doc, dtype='euclidean')
