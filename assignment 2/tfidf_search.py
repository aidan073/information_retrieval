import pickle
import indexer
import math
import numpy as np
from collections import Counter

def getObjects(filepath):
    with open(filepath, 'rb') as f:
        objects = pickle.load(f)
    return objects

def queryVecs(query, vocab, df_dict, doc_bm25):
    query_tfidf = np.zeros(len(vocab))
    query_bm25 = np.zeros(len(vocab))
    word_counts = Counter(query)
    index = 0
    if len(query) == 0:
        raise Exception("Empty query")
    for term in vocab:
        if term in query and term in df_dict:
            query_tfidf[index] = indexer.termTFIDF(term, word_counts, df_dict, len(doc_bm25), len(query))
            query_bm25[index] = indexer.termBM25(term, word_counts, df_dict, 1, 1, len(doc_bm25), len(query))
        index+=1
    return query_tfidf, query_bm25

def searchVecs():
    pass

query = 'This query should return a vector'
objects = getObjects('index.pkl')
tokens = indexer.process_text(query)
query_tfidf, query_bm25 = queryVecs(tokens, objects[0], objects[1], objects[2])
pass
#searchVecs()