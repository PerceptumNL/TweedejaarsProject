
from datawrapper import DataWrapper
import vectorizers
import distance
import sys
import itertools
import json 
import argparse
import pkgutil
from decimal import *
import prob
import os
import numpy as np
from collections import Counter

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
                  l_deval=True, t_deval=True, threshold = 0.3, k_link = False):
        """
        Returns the top k proposed links of document based on the data
        that was given during initialization. Both a vectorizer type and
        a distance measure can be given. If threshold = false, auto threshold
        is used. If K_link is true then the number of links that are in the
        network are used as threshold.

        l_deval and t_deval are flags for the probabilistic link devaluation
        and the probabilistic tag devaluation.
        """
        try:
            vectorizer = getattr(__import__('vectorizers.' + vtype), vtype)
        except ImportError:
            print('Unable to load vectorizers.{0}'.format(vtype))
            sys.exit(1)

        if(k_link):
            self.k = len(self.data.data['items'])

        self.document = document
        data_bows, new_doc_bow = vectorizer.vectorize(self.data, document)
        self.links = self.nearest_neighbor(data_bows, new_doc_bow, self.k, dtype)

        # Remove invalid links
        self.links = self.data.remove_invalid_links_for_item(document, self.links)

        if l_deval or t_deval:
            maxo = max(self.links,key=lambda item:item[1])[1]

        # Devaluate using the probabilistic devluations schemes if flags
        # are set.
        if l_deval:
            self.links = self.link_devaluation(self.data, document, self.links)
        if t_deval:
            self.links = self.tag_devaluation(self.data, document, self.links)
        self.links = sorted(self.links, key=lambda x: x[1])

        # Set the distances into the same range as before applying probabilities
        if l_deval or t_deval:
            maxn = max(self.links,key=lambda item:item[1])[1]
            if maxn != 0:
                factor = float(maxo)/maxn
            else:
                factor = 1
            self.links = [(x[0], factor*x[1]) for x in self.links] 

        # If threshold is set use the threshold method to return a number
        # of documents
        if threshold != False:            
            (self.links, x) = self.apply_threshold(self.links, threshold)

        # If k-link is set return the number of links that are in the
        # current network
        if (k_link):
            k = len(document['links'])
            self.links = self.links[0:k]

        return self.links
        
    def apply_threshold(self, links, threshold):
        """
        Given a list of links with a distance and a treshold return a number
        T of first documents in the list as selected documents and return all
        other documents as unselected documents. This number T is determined
        based on the distance between documents in the links list.
        """
        closest_distance = next((x[1] for x in links if x[1] > 0), None)

        if(closest_distance == None):
            print('Ignoring this distance')
            return([],[])

        l1 = []
        l2 = []
        for i, (link, x) in enumerate(links):
            max_diff = ((len(links) - i)/float(len(links)) * threshold * (1 - closest_distance))

            try:
                x = x[0,0]
            except Exception, c:
                x = x
            if x <= closest_distance + max_diff and x < 1.0:
                l1.append((link, x))
            else:
                l2.append((link, x))
        return (l1, l2)

    def link_devaluation(self, data, new_doc, deltas):
        """
        Given a list with documents, distance tuples and a new document.
        Compute the probability that the document will link to this document
        and then multiply the distance with the inverse of this probability.
        This way links with a low link probability are devaluated or in other
        words get a higher distance.
        """
        laplace_k = 1
        link_prob = prob.compute_link_probs(data, laplace_k)

        nd_type = new_doc['type'] # new doc type
        new_doc_tags = set(new_doc['tags'])
        dtype = lambda x: data.item(x)['type']
        dtags = lambda x: set(data.item(x)['tags']).intersection(new_doc_tags)

        lp = lambda x, y: link_prob[x][y]**-1

        return  [(doc,d*lp(nd_type,dtype(doc))) for doc, d in deltas]

    def tag_devaluation(self, data, new_doc, deltas):
        """
        Given a list with documents, distance tuples and a new document.
        Compute the probability that the document will link to this document
        given the number of tags the two documents have in common and then
        multiply the distance with the inverse of this probability.  This way
        links with a low link probability are devaluated or in other words get
        a higher distance.
        """
        tags = lambda x: data.item(x)['tags']
        tag_int = lambda x, y: set(tags(x)).intersection(set(tags(y)))
        tl = lambda x: len(set(tags(x)).intersection(set(new_doc['tags'])))

        # Compute item cardinality and compute the number of times a number
        # of tag insersection set cardinality occurs
        item_card = len(list(data.items()))
        l_counter = Counter()
        for item_a in data.items():
            for item_b in data.items():
                if item_a == item_b: continue
                l_counter.update([len(tag_int(item_a, item_b))])

        links = lambda: ((item, l) for item in data.items() for l in data.item(item)['links']) 
        link_card = Counter([len(tag_int(*l)) for l in links()])
        total_link = sum(link_card.values())

        # Compute the actual probabilities and finally used bayes rule
        # to combine to a single probability. More on this in the report.
        l_counts = [l_counter[x] + 1 for x in range(0,11)]
        l_cards = [link_card[x] + 1 for x in range(0,11)]
        p_sigma = map(lambda x: x/float(item_card*(item_card-1)), l_counts)
        p_sigma_doc = map(lambda x: x/float(total_link), l_cards)
        p_x = sum(link_card.values())/float(item_card*(item_card-1))
        p = lambda x: ((p_sigma[tl(x)] * p_sigma_doc[tl(x)] * p_x))**-1

        r = []
        for doc, delta in deltas:
            try:
                r.append((doc, p(doc)*delta))
            except IndexError:
                print(new_doc['id'], doc)
                print(tags(doc))
                print(new_doc['tags'])
        return r

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
        Retrieves the name of an author based on author id 
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
            # First create links before formatting them
            self.links = [] 
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

def run(vectorizer, distancetype, thresh, l_deval, t_deval, k_link, directory):
    """
    Run the documentlinker with a given distancetype, threshold, link deval
    option, tag devlauation optian, k_link options and a directory to write
    output to. 
    
    This function is repsonsible for generating the reports.
    """
    data = DataWrapper('../data/expert_maybe_false.pickle')
    data.remove_aliased_tags()
    data.remove_invalid_links()

    nlinks = [len(data.item(x)['links']) \
        for x in data.items() if len(data.item(x)['links']) > 0]
    print('\nAverage links: {0}\nNumber of documents with links: {1}'
        .format(np.mean(nlinks), len(nlinks)))

    filename = "{0}/{1}_{2}_{3}_{4}_{5}_{6}.json".format(directory, vectorizer, \
        distancetype, thresh, l_deval, t_deval, k_link)

    c = 0
    docs = {}
    total_recall = 0
    total_precision = 0
    total_f1 = 0

    precision_per_type = {}
    recall_per_type = {}
    f1_per_type = {}
    types = {}

    types = set(data.item(x)['type'] for x in data.items())
    table_format = {}
    for ntype in types:
        table_format[ntype] = {}
        for itype in types:
            table_format[ntype][itype] = 0
    types = table_format

    for new_doc, datawrapper in data.test_data():
        if new_doc['type'] == 'Glossary':
            continue

        # First retrieve proposed links
        linker = DocumentLinker(datawrapper)

        linker.get_links(new_doc, vtype=vectorizer, dtype=distancetype, threshold=thresh, \
            l_deval = l_deval, t_deval = t_deval, k_link = k_link)
        links = linker.formatted_links()
        docs[c] = links
        c += 1
        correct = 0
        for link in linker.links:
            if(link[0] in new_doc['links']):
                correct += 1

        if(len(linker.links) > 0):
            precision = Decimal(correct)/len(linker.links)
        else:
            precision = 0

        if(len(new_doc['links']) > 0):
            recall = Decimal(correct)/len(new_doc['links'])
        else:
            recall = 0
            print('Document {0}\nThere are no valid links for this document \n'\
                .format(new_doc['id']))
            c -= 1
            continue

        if recall == 0 and precision == 0:
            f1 = 0
        else: 
            f1 = (recall*precision*2)/(recall + precision)

        # Save precision and recall
        total_recall += recall
        total_precision += precision
        total_f1 += f1
        if not precision_per_type.has_key(new_doc['type']):
            precision_per_type[new_doc['type']] = [precision]
        else:
            precision_per_type[new_doc['type']].append(precision)
        if not recall_per_type.has_key(new_doc['type']):
            recall_per_type[new_doc['type']] = [recall]
        else:
            recall_per_type[new_doc['type']].append(recall)
        if not f1_per_type.has_key(new_doc['type']):
            f1_per_type[new_doc['type']] = [f1]
        else:
            f1_per_type[new_doc['type']].append(f1)

        # Save linktypes
        for link in linker.links:
            linktype = data.item(link[0])['type']            
            types[new_doc['type']][linktype] += 1

        print('Document {0}, proposed links: {1}, real links: {2}'.format(new_doc['id'], \
            len(linker.links), len(new_doc['links'])))
        print('Recall: {0}'.format(recall))
        print('Precision: {0}'.format(precision))
        print('F1-measure: {0} \n'.format(f1))

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Performance report of ' + vectorizer)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')

    print('Average recall: {0}'.format(Decimal(total_recall)/c))
    print('Average precision: {0}'.format(Decimal(total_precision)/c))
    print('Average f1-measure: {0}'.format(Decimal(total_f1)/c))

    print('\nAverage recall per type')
    for key in recall_per_type:
        print('{0}: \t {1}'.format(key, np.mean(recall_per_type[key])))
    print('\nAverage precision per type')
    for key in precision_per_type:
        print('{0}: \t {1}'.format(key, np.mean(precision_per_type[key])))
    print('\nAverage f1 per type')
    for key in f1_per_type:
        print('{0}: \t {1}'.format(key, np.mean(f1_per_type[key])))

    print('\nLink distribution per type in percentages')
    for ntype in types:
        print('{0}:'.format(ntype))
        nsum = sum(types[ntype].values())
        for i in types[ntype]:
            if types[ntype][i] == 0:
                mean = 0
            else:
                mean = types[ntype][i]/float(nsum)*100
            print('{0}: \t{1}'.format(i, mean))
        print('')

    print('\nTotal number of documents: {0}'.format(c))

    file = open(filename, "w")
    file.write(json.dumps(docs))
    file.close()

class valid_dir(argparse.Action):
    """
    Check if directory is valid
    """
    def __call__(self,parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("Invalid directory: {0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentTypeError("Invalid directory: {0} is not a readable dir".format(prospective_dir))

if __name__ == '__main__':
    # Setup an argument parser
    parser = argparse.ArgumentParser(description="The documentlinker calculates the average performance for all documents in the Starfish knowledge base using the take one out principle.",\
        usage='documentlinker.py [options]')
    vectorizer_names = [name for _, name, _ in pkgutil.iter_modules(['vectorizers'])]
    distances = ['euclidean', 'cosine', 'correlation', 'intersection', 'bhattacharyya']
    parser.add_argument('-vectorizer', default='hybrid', type=str, choices=vectorizer_names,
            help="The vectorizer used to transform the documents. This argument is required.", required=True)
    parser.add_argument('-directory', default='../data', action=valid_dir,
            help="The name of the directory the JSON output files should be saved to, default is '../data'",\
            required=False)
    parser.add_argument('-metric', default='cosine', choices=distances, type=str,
            help="Distance metric for nearest neighbor, the default metric used is 'cosine'", required=True)
    parser.add_argument('-link_devaluation', action='store_true', 
            help="If this agrument is given, the link probabilitie is applied to the distances", required=False)
    parser.add_argument('-tag_devaluation', action='store_true',
            help="If this agrument is given, the tag probabilitie is applied to the distances", required=False)

    group_ex = parser.add_mutually_exclusive_group()

    group_ex.add_argument('-k_link', action='store_true',
            help="Induces k-link measuring, i.e. the same number of links as in the original document were are proposed. Is mutually exlusive with -threshold", required=False)
    group_ex.add_argument('-threshold', default=-1, type=float,
            help="Induces the threshold meganism. Give the parameter for alpha, a value between [0...1]. If both k_link and threshold are not used, the linker proposes a fixed number of 10 links.", required=False)

    # run parser 
    args = parser.parse_args()
    threshold = False if args.threshold == -1 else args.threshold

    # run documentlinker via run function
    run(args.vectorizer, args.metric, threshold, args.link_devaluation, \
        args.tag_devaluation, args.k_link, args.directory)
