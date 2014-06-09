
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

 	def preprocessed_content(self, item):
 		"""
        Returns the preprocessed content (e.g. text and title) of item 
        """
 		if item['type'] == 'Person':
 	 		headline = self.preprocessed_by_key(item, 'headline')
         	about = self.preprocessed_by_key(item, 'about')
         	return headline + ' ' + about
 		else:
 			text = data_.preprocessed_by_key(x, 'text')
         	title = data_.preprocessed_by_key(x, 'title')
         	return title + ' ' + text

    def preprocessed_contents(self):
    	"""
        Returns a generator that yields preprocessed contents in all items.
        """
    	for item in self.items:
    		yield self.preprocessed_content(item)

    def preprocessed_by_key(self, item, key):
        """
        Returns the preprocessed text of item if set. Otherwise returns and
        empty string.
        """
        item = self.data['items'][item]
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

    def preprocessed_texts(self):
        """
        Returns a generator that yields preprocessed texts in all items.
        """
        for item in self.items:
            yield self.preprocessed_text(item)

    def tag_glossary(self, tag):
        """
        Returns the glossary text of a tag if set. Otherwise returns an
        empty string.
        """
        return self.preprocessed_text(self.tag(tag)['glossary'])

