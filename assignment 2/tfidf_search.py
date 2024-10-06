import pickle
import indexer

def getObjects(filepath):
    with open(filepath, 'rb') as f:
        objects = pickle.load(f)
    return objects

def queryVecs(query):
    pass

def searchVecs():
    pass

query = 'gqrah'
objects = getObjects('index.pkl')
tokens = indexer.process_text(query)