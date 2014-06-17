
from documentlinker import DocumentLinker
from datawrapper import DataWrapper
import random

def create_folds(data, k):
    """
    Create k folds from dataset and return a generator the returns all items
    from a fold and a datawrapper with the items from all other folds. This can
    be used to perform cross validation.
    """
    all_items  = filter(lambda x: data.item(x)['type'] != 'Glossary', data.items())
    random.shuffle(all_items)
    slice_size = len(all_items) / k
    folds = [all_items[s*slice_size:(s+1)*slice_size] for s in range(0,k)]
    for fold in folds:
        yield data.remove_item(*fold)


if __name__ == "__main__":
    data = DataWrapper('../data/export_starfish_tjp_12jun.pickle')
    print(create_folds(data, 5).next())
