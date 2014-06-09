
from datawrapper import DataWrapper
import vectorizers
import sys

class DocumentLinker(object):

    def __init__(self, datafile, k=10):
        self.data = DataWrapper(datafile)
        self.k = k

    def get_links(self, document, vtype='textvectorizer', dtype='cosine'):
        try:
            vectorizer = getattr(__import__('vectorizers.' + vtype), vtype)
        except ImportError:
            print('Unable to load vectorizers.{0}'.format(vtype))
            sys.exit(1)

        data_bows, new_doc_bow = vectorizer.vectorize(self.data, document)

    def nearest_neighbor(k, dtype):
        pass

if __name__ == '__main__':
    new_doc = 'dit is een nieuw document Learning Analytics'
    linker = DocumentLinker('../data/export_starfish_tjp.pickle')
    linker.get_links(new_doc)
