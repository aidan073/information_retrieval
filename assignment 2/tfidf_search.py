import pickle
import indexer

def getDocs(filepath):
    with open(filepath, 'rb') as f:
        docs = pickle.load(f)
    return docs

query = 'gqrah'
tokens = indexer.process_text(query)