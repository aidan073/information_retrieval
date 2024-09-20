from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import indexer

stop_words = set(stopwords.words('english'))

def format_query(query):
    """Format query for searching

    Args:
        query (str): free-text input query

    Returns:
        filtered_query (str): query prepared for searching
    """
    tokens = word_tokenize(query)
    filtered_query = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return filtered_query

query = format_query("Hello, this is the best query!")
indexer1 = indexer.Indexer('Index.json')
res = indexer1.search(query)