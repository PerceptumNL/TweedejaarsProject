
from __future__ import with_statement
import cPickle as pickle
import preprocessing
from copy import deepcopy

class DataWrapper(object):
    """
    This class wraps the data object exported from starfish. This class is used
    troughout the codebase to interact with the exported datafiles from
    startifsh.
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
        """
        Remove tags that are an alias of another tag and replace with
        alias.
        """
        # Walk trough items and remove aliased tags
        for k,v in self.data['items'].items():
            tags = set()
            for tag in v['tags']:
                alias = self.get_alias_of_tag(tag)
                tags.add(tag if alias is None else alias)
            v['tags'] = list(tags)
        del_tags = []
        # remove aliased tags
        for tag in self.tags():
            tag_dic = self.tag(tag)
            if tag_dic['alias_of'] is not None and tag != tag_dic['alias_of']:
                del_tags.append(tag)
        for tag in del_tags: del self.data['tags'][tag]

    def remove_glossaries(self):
        """
        Remove the glossaries from the data set. Use this method when the
        glossaries should not be taken into account to for example compute
        the correct proposed links.
        """
        data = deepcopy(self.data)
        for k,v in self.data['items'].items():
            if v['type'] == 'Glossary':
                del data['items'][k]
        self.data = data

    def ignore(self, item, item_id, link_id):
        """ 
        Indicates whether or not a particular link is valid within the dataset.
        Returns true if the link is invalid and should be ignored
        """
        link = self.item(link_id)

        # Ignore if link simply is the author of the doucment
        if (item.get('author') == link_id):
            return True
        
        # Ignore if link is glossary 
        if (link['type'] == 'Glossary'):
            return True
        
        # Ignore if link is simply a document written by person described by document
        if (item['type'] == 'Person' and link.get('author') == item_id):
            return True

        return False

    def remove_invalid_links(self):
        """
        Removes all links within the data that are invalid due to type or
        authorship. This links should not be proposed by the algorithm
        because the are already automatically added in starfish.
        """
        for k,v in self.data['items'].items():
            tempLinks = v['links'][:]
            for link in v['links']:
                if(self.ignore(v, k, link)):
                    tempLinks.remove(link)
            v['links'] = tempLinks[:]

    def remove_invalid_links_for_item(self, item_dict, links):
        """
        Remove links from list of tuples (link, weight) that are invalid due type
        or authorship. The document must be a dict with at least an id field. 
        """
        tlinks = links[:]
        for link in links:
            if(self.ignore(item_dict, item_dict['id'], link[0])):
                tlinks.remove(link)
        return tlinks 

    def get_alias_of_tag(self, tag):
        """
        Return the alias of a tag. If the tag has no alias the tag itself
        will be returned.
        """
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
            if self.item(item)['type'] == 'Glossary':
                continue
            yield self.remove_item(item)

    def remove_item(self, *items):
        """
        Remove a single item from the data set. Return this item and a
        datawrapper where this item has been removed. Will not alter the
        current datawrapper. This method does not protect against removing
        glossaries.
        """
        item_dicts = [] 
        data = deepcopy(self.data)
        for item in items:
            del data['items'][item]
            for k,v in data['items'].items():
                try:
                    v['links'] = [x for x in v['links'] if x != item]
                except ValueError:
                    pass
            item_dict = self.item(item)
            item_dict['id'] = item
            item_dicts += [item_dict]
        if len(items) == 1:
            return (item_dicts[0], DataWrapper(data))
        ids = map(lambda x: x['id'], item_dicts)
        for item_dict in item_dicts:
            item_dict['links'] = filter(lambda x: not x in ids, item_dict['links'])
        return (item_dicts, DataWrapper(data))
