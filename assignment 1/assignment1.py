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
        with open(out_file, 'w', newline='') as f2:
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

# query = indexer.Indexer.process_text("<p>The dogs of Oxford, I declare:<br />Numbered one third of a square.<br />If one quarter left to roam,<br />Just a cube would stay at home.</p><p>What is the smallest possible number of dogs in Oxford?</p><hr /><p>This is a relatively easy question so I would recommend the more experienced solvers leave this question for new puzzlers.</p><hr /><p>Edit for attribution - I found this riddle <a href=\"https://www.reddit.com/r/mathmemes/s/twI15znhIk\" rel=\"noreferrer\">here</a>. Minor wording tweaks for cadence.</p>")
# query = list(set(query))
# indexer1 = indexer.Indexer('Index.json')
# res = indexer1.search(query)
# pass

search_topics('topics_1.json', 'result_binary_2.tsv')