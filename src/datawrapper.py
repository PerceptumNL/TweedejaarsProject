
from __future__ import with_statement
import cPickle as pickle
import preprocessing

class DataWrapper(object):
    """
    This class wraps the data object exported from starfish.
    """

    def __init__(self, datafile):
        """
        Initialize class and read date file
        """
        self.datafile = datafile
        self.read_datafile()

    def read_datafile(self):
        """
        Reads the data file and stores in self.data
        """
        with open(self.datafile) as f:
            self.data = pickle.load(f)

    @property
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

    @property
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

    def preprocessed_key(self, item, key):
        """
        Returns the preprocessed content of one particular key of an item if set. 
        Otherwise returns an empty string.
        """
        item = self.data[key][item]
        if key in item:
            return preprocessing.preprocess_text(item[key])
        else:
            return ''

    def preprocessed_text(self, item):
        """
        Returns the preprocessed text of item if set. Otherwise returns and
        empty string.
        """
        item = self.data['items'][item]
        if 'text' in item:
            return preprocessing.preprocess_text(item['text'])
        else:
            return ''

    def tag_glossary(self, tag):
        """
        Returns the glossary text of a tag if set. Otherwise returns an
        empty string.
        """
        return self.preprocessed_text(self.tag(tag)['glossary'])
