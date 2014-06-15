import json
from decimal import *

files = []
files.append('textvectorizer_cosine.json')
files.append('glossaries_of_tags_cosine.json') 
files.append('tag_smoothing_cosine.json')
files.append('weighted_tagvectorizer_cosine.json')
files.append('weighted_text_vectorizer_cosine.json')

documents = {}

def inList(mlist, mid):
    for i in mlist:
        if(i[0] == mid):
            return True
    return False

for nfile in files:
    json_data= open(nfile).read()
    data = json.loads(json_data)

    for document_id in data:
        if documents.has_key(document_id):
            links = data[document_id]['links']
            for link_id in links:
                link = data[document_id]['links'][link_id]
                if(link['type'] == 'Glossary'):
                    continue
                if inList(links, link_id):
                    continue
                else:
                    if not link.get('correct') and not link.get('not_recalled'):
                        documents[document_id]['links'][link_id] = link

        else:
            documents[document_id] = data[document_id]
            documents[document_id]['links'] = {}
            links = data[document_id]['links']
            for link_id in links:
                link = links[link_id]
                if(link['type'] == 'Glossary'):
                    continue
                if not link.get('correct') and not link.get('not_recalled'):
                    documents[document_id]['links'][link_id] = link

file = open('combined_results.json', "w")
file.write(json.dumps(documents))
file.close()

c = 0
t = 0
csv = ''
for i in documents:
    c+= 1
    t += len(documents[i]['links'])
    csv += '{0}, {1} \n'.format(i, documents[i]['title'])
    for link in documents[i]['links']:
        csv += ',{0},{1}\n'.format(link, documents[i]['links'][link]['title'])

print Decimal(t)/c
file = open('combined_results_csv.csv', 'w')
file.write(csv)
file.close()

