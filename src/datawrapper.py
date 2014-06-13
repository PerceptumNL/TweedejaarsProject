
from __future__ import with_statement
import cPickle as pickle
import preprocessing
from copy import deepcopy

class DataWrapper(object):
    """
    This class wraps the data object exported from starfish.
    """

    def __init__(self, data):
        """
        Initialize class and read date file
        """
        if type(data) == str:
            self.datafile = data
            self.read_datafile()
        else:
            self.datefile = None
            self.data = data

    def read_datafile(self):
        """
        Reads the data file and stores in self.data
        """
        with open(self.datafile) as f:
            self.data = pickle.load(f)

    def remove_aliased_tags(self):
        for k,v in self.data['items'].items():
            tags = set()
            for tag in v['tags']:
                alias = self.get_alias_of_tag(tag)
                tags.add(tag if alias is None else alias)
            v['tags'] = list(tags)
        del_tags = []
        for tag in self.tags():
            tag_dic = self.tag(tag)
            if tag_dic['alias_of'] is not None and tag != tag_dic['alias_of']:
                del_tags.append(tag)
        for tag in del_tags: del self.data['tags'][tag]

    def get_alias_of_tag(self, tag):
        tag_dict = self.tag(tag)
        alias = tag_dict['alias_of']
        if alias is None or alias == tag:
            return tag
        else:
            return self.get_alias_of_tag(alias)
            


    def tags(self):
        """
        Returns a generator for all tags in the datafile.
        """
        for tag in self.data['tags']:
            yield tag

    def tag(self, tag):
        """
        Returns the dictionary of single tag.
        """
        return self.data['tags'][tag]

    def items(self):
        """
        A generator for all items in the datefile.
        """
        for item in self.data['items']:
            yield item

    def item(self, item):
        """
        Returns the dictionary of single item.
        """
        return self.data['items'][item]

    def value_for_keys(self, item_id=None, *keys):
        """
        Returns the preprocessed concatenated values for keys. If no
        item_id is given a generator that will return the values for all
        items in the data storage.
        """
        if item_id is None:
            return self.__value_for_keys(*keys)
        else:
            item = self.item(item_id)
            return self.value_for_keys_with_item(item, *keys)

    def __value_for_keys(self, *keys):
        """
        Returns a generator that will yield the concatenated values of
        keys of all items in the data storage.
        """
        for item_id in self.items():
            item = self.item(item_id)
            yield self.value_for_keys_with_item(item, *keys)

    def value_for_keys_with_item(self, item, *keys):
        """
        Returns the concatenated preprocessed values for keys in the
        item dictionary.
        """
        val = ' '.join([item.get(k, '') for k in keys])
        return preprocessing.preprocess_text(val)

    def tag_glossary(self, tag):
        """
        Returns the glossary text of a tag if set. Otherwise returns an
        empty string.
        """
        glossary_id = self.tag(tag)['glossary']
        if(glossary_id):
            return preprocessing.preprocess_text(self.item(glossary_id)['text'])
        return ''
    
    def test_data(self):
        """
        Generate test data sets that consists of a new document not currently
        part of the dataset and a dataset of known documents. These are
        generated by taking and removing a single document from the dataset.
        For this glossaries are ignored as these are part of other structures
        in the dataset.
        """
        for item in self.data['items']:
            data = deepcopy(self.data)
            if self.item(item)['type'] == 'Glossary':
                continue
            del data['items'][item]
            for k,v in data['items'].items():
                try:
                    v['links'].remove(item)
                except ValueError:
                    pass
            item_dict = self.item(item)
            item_dict['id'] = item
            yield (item_dict, DataWrapper(data))
