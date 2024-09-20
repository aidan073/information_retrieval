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
                result = indexer1.search(query)
                rank = 1
                for key, val in result.items():
                    row = [topic['Id'], 'Q0', key, rank, val, 'BOW_Search']
                    writer.writerow(row)
                    rank+=1
                    if rank > 100:
                        break

search_topics('topics_1.json', 'result_binary_3.tsv')