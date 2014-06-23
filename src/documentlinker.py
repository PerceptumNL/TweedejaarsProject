
from datawrapper import DataWrapper
import vectorizers
import distance
import sys
import itertools
import json 
from decimal import *
import prob

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

    def get_links(self, document, vtype='textvectorizer', dtype='euclidean',
                  p_deval=True):
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

        if p_deval:
            self.links = self.prob_devaluation(self.data,document, self.links)

        return self.links

    def prob_devaluation(self, data, new_doc, deltas):
        # laplace_k = 1
        # link_prob = prob.compute_link_probs(data, laplace_k)
        # tag_prob = prob.compute_tag_probs(data, laplace_k)
        # link_tag_prob = prob.compute_tag_link_prob(data, laplace_k)

        # nd_type = new_doc['type'] # new doc type
        # new_doc_tags = set(new_doc['tags'])
        # dtype = lambda x: data.item(x)['type']
        # dtags = lambda x: set(data.item(x)['tags']).intersection(new_doc_tags)

        # lp = lambda x, y: link_prob[x][y]

        # res = [(doc,d*lp(nd_type,dtype(doc))**-1) for doc, d in deltas]
        # return sorted(res, key=lambda x: x[1])
        return deltas


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

    def __find_author_name(self, author_id):
        """
        Retrieves the name of an author based on author id (should be in 
        datawrapper maybe???)
        """
        if(author_id == self.document['id']):
            author = self.document['name']
        else:
            try:
                author = self.data.item(author_id)['name']
            except KeyError:
                author = 'unknown'
        return author

    def __format_item(self, item):
        """
        Formats one single item in a dictionary
        """
        title = self.data.value_for_keys(item, 'title', 'name')
        content = self.data.value_for_keys(item, 'headline', 'about', 'title', 'text')
        item_dict = self.data.item(item)
        author_id = self.data.item(item).get('author') or -1
        author = self.__find_author_name(author_id)

        tags = item_dict['tags']
        linktype = item_dict['type']

        return {'type': linktype, 'title': title, 'content': content, 'tags': tags, 'author': author}

    def formatted_links(self):
        """
        Formats a set of proposed links into a dictionary format usable for the viewer
        """
        if(not self.links):
            print('First create links before formatting them')
            return False
        nlinks = {}
        links = [i[0] for i in self.links]

        i = 1
        for link in links:
            nlink = self.__format_item(link)
            nlink['correct'] = link in self.document['links']
            nlink['rank'] = i
            nlinks[link] = nlink
            i += 1

        for link in self.document['links']:
            if not(link in links):
                try:
                    nlink = self.__format_item(link)
                    nlink['not_recalled'] = True
                    nlinks[link] = nlink
                except Exception as e:
                    print('Couldn\'t find link with id {0}'.format(link))
                    continue;

        # Bring current doc in proper format
        title = self.data.value_for_keys_with_item(self.document, 'title', 'name')
        content = self.data.value_for_keys_with_item(self.document, 'headline', 'about', 'title', 'text')
        author = self.__find_author_name(self.document.get('author') or -1)
        doc = {'type': self.document['type'], 'id': self.document['id'], 'links': nlinks, 'title': title, 'content': content, 'author': author, 'tags': self.document['tags']}
        return doc 

def run(vectorizer, distancetype):
    data = DataWrapper('../data/export_starfish_tjp_12jun.pickle')
    data.remove_aliased_tags()
    filename = "../data/data_12jun/{0}_{1}.json".format(vectorizer, distancetype)

    c = 0
    docs = {}
    percentage = 0
    for new_doc, datawrapper in data.test_data():
        linker = DocumentLinker(datawrapper)
        linker.get_links(new_doc, vtype=vectorizer, dtype=distancetype)
        links = linker.formatted_links()
        docs[c] = links
        c += 1
        correct = 0
        for link in linker.links:
            if(link[0] in new_doc['links']):
                correct += 1
        if(len(new_doc['links']) > 0):
            percentage_correct = Decimal(correct)/len(new_doc['links'])
        else:
            percentage_correct = 0
            print('No links')
            c -= 1
        percentage += percentage_correct
        print('Percentage correct: {0}'.format(percentage_correct))

    print('Average percentage correct: {0}'.format(Decimal(percentage)/c))
    file = open(filename, "w")
    file.write(json.dumps(docs))
    file.close()

if __name__ == '__main__':
    vectorizer = ''
    metric = ''
    invalid = False
    for i in range(0, len(sys.argv)):
        try:
            if sys.argv[i] == "-vectorizer":
                vectorizer = sys.argv[i + 1]
            if sys.argv[i] == "-distance":
                metric = sys.argv[i + 1]
        except:
            invalid = True
    if(vectorizer == '' or metric == '' or invalid == True):
        print('Usage: -vectorizer <algorithm> -distance <cosine/eucledian>')
        exit(0)

    run(vectorizer, metric)
