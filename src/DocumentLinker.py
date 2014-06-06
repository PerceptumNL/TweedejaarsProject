
class DocumentLinker(object):

    def __init__(self, datafile):
        pass

    def get_links(self, document, algo='default'):
        algos = {'default': self.dd_doc_linker}

        known_docs, new_doc = algos[algo](self.documents, document)
        self.nearestNeighrbor(new_doc, known_docs, k=100)

    def dd_doc_linker(self, raw_network, new_document):
        vectorizer = DDVectorizer()
        vectorizer.fit(documents)
        return (vectorizer.transform(documents), 
                vectorizer.transform(new_documnet))
