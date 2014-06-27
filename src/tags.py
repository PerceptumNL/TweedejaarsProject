import os, sys
from decimal import Decimal
sys.path.append(os.path.join(os.path.abspath('..'), 'src'))
from datawrapper import DataWrapper
import json
import matplotlib.pyplot as plt

data = DataWrapper('../data/expert_maybe_true.pickle')
data.remove_aliased_tags()
data.remove_invalid_links()
data.remove_glossaries()

tagFreq = {}
for doc in data.items():
    for tag in data.item(doc)['tags']:
        if tagFreq.has_key(tag):
            tagFreq[tag] += 1
        else:
            tagFreq[tag] = 1

plt.hist(tagFreq.values(), 70, color='g')
plt.xlabel('Number of documents a tag appeared in')
plt.ylabel('Percentage of tags')
plt.show()