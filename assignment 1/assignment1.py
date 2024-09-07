import pyterrier as pt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

stop_words = set(stopwords.words('english'))

index = pt.IndexFactory.of(os.path.join(os.getcwd(), "index"))
retriever = pt.BatchRetrieve(index)

def format_query(query):
    tokens = word_tokenize(query)
    filtered_query = ''
    for token in tokens:
        if not token.lower() in stop_words:
            filtered_query += ' '+token.lower()
    
    return filtered_query

query = format_query("Hello this is the best query")
results = retriever.search(query)
print(results.head())