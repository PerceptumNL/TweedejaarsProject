
"""
This file implements different functions used to preprocess items
in a datafile.
"""

import nltk
from BeautifulSoup import BeautifulSoup
import re
from unidecode import unidecode


def preprocess_text(text):
    """
    Strips all html from a text and makes sure all characters are ascii.
    """
    text_stripped_html = nltk.clean_html(text)
    hexentityMassage = [(re.compile('&#x([^;]+);'), 
        lambda m: '&#%d;' % int(m.group(1), 16))]
    text_tag = BeautifulSoup(text_stripped_html, 
            convertEntities=BeautifulSoup.HTML_ENTITIES, 
            markupMassage=hexentityMassage)
    text_unicode = text_tag.text.encode('utf-8')
    return unidecode(unicode(text_unicode))
