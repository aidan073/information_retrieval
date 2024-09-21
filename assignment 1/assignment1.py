import indexer
import json
import csv
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def search_topics(in_file, out_file):
    indexer1 = indexer.Indexer('Index.json')
    with open(in_file, 'r', encoding='utf-8') as f:
        topics = json.load(f)
        with open(out_file, 'a', newline='') as f2:
            writer = csv.writer(f2, delimiter='\t')
            for topic in topics:
                query = indexer.Indexer.process_text(topic['Body'])
                query = list(set(query)) # remove duplicates
                result = indexer1.search(query)
                rank = 1
                for key, val in result.items():
                    row = [topic['Id'], 'Q0', key, rank, val, 'BOW_Search']
                    writer.writerow(row)
                    rank+=1
                    if rank > 100:
                        break

query = indexer.Indexer.process_text("<p>A rectangular room has a floor tiled with tiles of two shapes: 1×4 and 2×2.The tiles completely cover the floor of the room, and no tile has been damaged, or cut in half. One day, a heavy object is dropped on the floor and one of the tiles is cracked. The handyman removes the damaged tile and goes to the storage to get a replacement. But he finds that there is only one spare tile, and it is of the other shape. Can he rearrange the remaining tiles in the room in such a way that the spare tile can be used to fill the hole?</p><p>Source of the puzzle <a href=\"https://math.stackexchange.com/questions/724075/heres-a-cool-problem-a-collection-of-short-questions-with-clever-solutions\">Math.SE</a></p>")
query = list(set(query))
indexer1 = indexer.Indexer('Index.json')
res = indexer1.search(query)

#search_topics('topics_1.json', 'result_binary_3.tsv')