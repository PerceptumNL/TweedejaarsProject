
from documentlinker import DocumentLinker
from datawrapper import DataWrapper
import random
import numpy as np

def create_folds(data, k):
    """
    Create k folds from dataset and return a generator the returns all items
    from a fold and a datawrapper with the items from all other folds. This can
    be used to perform cross validation.
    """
    all_items  = filter(lambda x: data.item(x)['type'] != 'Glossary', data.items())
    all_items = filter(lambda x: len(data.item(x)['links']) > 0, all_items)
    random.shuffle(all_items)
    slice_size = len(all_items) / k
    folds = [all_items[s*slice_size:(s+1)*slice_size] for s in range(0,k)]
    for fold in folds:
        yield data.remove_item(*fold)

def evaluate(new_docs, data, vec='weighted_tagvectorizer', dis='cosine'):
    """
    v = Type of vectorizer to evaluate
    d = Type of distance to evaluate
    """
    linker = DocumentLinker(data, k=1000)
    rank_count = [0] * len(list(data.items()))
    for new_doc in new_docs:
        links = map(lambda x: x[0], linker.get_links(new_doc, vec, dis))
        for k,v in enumerate(links):
            rank_count[k] += 1 if v in new_doc['links'] else 0
    return rank_count


if __name__ == "__main__":
    data = DataWrapper('../data/expert_maybe_true.pickle')
    results = []
    for new_docs, data in create_folds(data, 5):
        results += [evaluate(new_docs, data)]
    results_arr = np.array(results)
    print(np.mean(results_arr, axis=0))
    print(np.std(results_arr, axis=0))
