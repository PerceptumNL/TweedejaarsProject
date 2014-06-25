import os, sys
from decimal import Decimal
from documentlinker import DocumentLinker
from datawrapper import DataWrapper
import json

# Determine name of author (for visualization)
def __find_author_name(linker, author_id):
    if(author_id == linker.document['id']):
        author = linker.document['name']
    else:
        try:
            author = linker.data.item(author_id)['name']
        except KeyError:
            author = 'unknown'
    return author

# Check if a link should be added or ignored
def ignore(new_doc, link, data): 
    # Ignore if proposed link simply is the author of the doucment \
    if (new_doc.has_key('author') and new_doc['author'] == link):
        return True
    
    # Ignore if link is glossary 
    if (data.item(link)['type'] == 'Glossary'):
        return True
    
    # Ignore if proposed link is simply a document written by person described by document \
    if (new_doc['type'] == 'Person' and data.item(link).has_key('author') and data.item(link)['author'] == new_doc['id']):
        return True
    return False
    
# Same as in DocumentLinker.py, only now not k = 10 but k = known amount of links in doc   
def mrun(vectorizer, distancetype):
    data = DataWrapper('../data/expert_maybe_true.pickle')
    data.remove_aliased_tags()
    filename = "../data/final_analysis/probs/{0}_{1}.json".format(vectorizer, distancetype)

    c = 0
    docs = {}
    percentage = 0
    
    # Return ranking of all documents
    k = len(data.data['items']) - 1
    
    for new_doc, datawrapper in data.test_data():
        linker = DocumentLinker(datawrapper, k)
        try:
            linker.get_links(new_doc, vectorizer, distancetype, l_deval=True, t_deval=True, threshold = 0.3)
        except:
            print('Ignoring {0}'.format(new_doc['id']))
            continue
        
        # Remove all invalid ones from new doc
        tempLinks = new_doc['links'][:]
        for link in new_doc['links']:
            if (ignore(new_doc, link, data)):
                tempLinks.remove(link)
        new_doc['links'] = tempLinks[:]
        
        l = len(new_doc['links'])
        if(l == 0):
            print('No valid links for {0}'.format(new_doc['id']))
            continue
        
        # Remove all invalid ones from proposed and take number equal to new_doc
        i = 0
        tempLinks = linker.links[:]
        for link in linker.links:
            if(i == l):
                continue
            if (ignore(new_doc, link[0], data)):
                tempLinks.remove(link)
            else:
                i += 1
        linker.links = tempLinks[0:l]
        formatted_doc = linker.formatted_links()
            
        # The following is similar as in DocumentLinker.py 
        docs[c] = formatted_doc
        c += 1
        correct = 0
        for link in linker.links:
            if(link[0] in new_doc['links']):
                correct += 1
        if(len(new_doc['links']) > 0):
            percentage_correct = Decimal(correct)/len(new_doc['links'])
        else:
            percentage_correct = 0
            print('No links')
            c -= 1
        percentage += percentage_correct
        print('{0} Percentage correct: {1}'.format(new_doc['id'], percentage_correct))

    print('Average percentage correct: {0}'.format(Decimal(percentage)/c))
    file = open(filename, "w")
    file.write(json.dumps(docs))
    file.close()

mrun('simple_tag_similarity', 'cosine')