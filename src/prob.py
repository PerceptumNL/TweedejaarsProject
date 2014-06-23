from datawrapper import DataWrapper
from collections import Counter
import numpy as np
from scipy.misc import comb

def laplace(total, types, from_t, to_t, k):
    """
    Compute laplace smoothing for link probabilities in network.
    """
    smoothed_count = total[from_t].get(to_t, 0) + k
    total_count = sum([sum(total[from_t].values()) for from_t in types])
    smoothed_total = total_count + len(types)**2*k
    return smoothed_count/float(smoothed_total)

def compute_link_probs(data, laplace_k=1):
    """
    Compute the probability that two documents are linked given their
    type. Smooth with laplace smoothing.
    """
    # Generator for all items
    items = lambda: ((item, data.item(item)) for item in data.items()) 
    # Function tom determine type of item
    itype = lambda item: data.item(item)['type']

    # get all unique types
    types = {item[1]['type'] for item in items()} 

    # Count all out links
    out_links = {t:Counter() for t in types}
    for item_id, item in items(): 
        out_links[itype(item_id)].update([itype(i) for i in item['links']])

    # compute the actual probabilities and store in dict of dicts.
    probs = {from_t:{to_t:laplace(out_links, types, from_t, to_t, laplace_k)
                for to_t in types}
                for from_t in types}

    return probs

def compute_tag_probs(data, lp_k=1):
    """
    Compute the probability that two document selected from the total
    set of documents have tag t.
    """
    # Generator for all tags
    tags = lambda: ((item, data.item(item)['tags']) for item in data.items()) 
    tag_count = Counter()
    for item_id, tag_list in tags(): 
        tag_count.update(tag_list)
    d = len(list(data.items())) # doc count

    # tag count
    tc = len(list(data.tags()))

    probs = {}
    for tag in data.tags():
        c = tag_count[tag]
        probs[tag] = (c*(c-1)+lp_k) / float(d*(d-1)+tc*lp_k)
    return probs

def compute_tag_link_prob(data, lp_k=1):
    """
    Compute the probability that two documents are linked given a tag.
    """
    # Generator for all items
    links = lambda: ((item, data.item(item)['links']) for item in data.items()) 
    # Generator for all tags
    tags = lambda: ((item, data.item(item)['tags']) for item in data.items()) 
    itags = lambda x: data.item(x)['tags']
    shared_tags = lambda x,y: set(itags(x)).intersection(set(itags(y)))

    # find all links in network
    all_links = [(i,l) for i, lnks in links() for l in lnks]

    
    # Compute link tag count
    ltc = {tag:len(filter(lambda x: tag in shared_tags(*x), all_links)) 
            for tag in data.tags()}
    # Compute total link count
    tlc = sum(ltc.values())

    return {k:(v+lp_k)/float(tlc+len(ltc)*lp_k) for k,v in ltc.items()}
