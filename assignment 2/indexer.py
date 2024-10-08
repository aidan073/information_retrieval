import json
import pickle
import math
import sys
import random
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

def readJSON(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    return docs

def process_text(text): # pre-process text
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text()
    tokens = word_tokenize(text)
    tokens = tokens = [stemmer.stem(token.lower()) for token in tokens if token.isalpha() and len(token) < 14 and token.lower() not in stop_words]
    return tokens

def getTF(doc_text):
    doc_tf = {}
    for token in doc_text:
        doc_tf[token] = doc_tf.get(token, 0) + 1
    doc_tf = {term: freq for term, freq in doc_tf.items() if freq >= 8}
    return doc_tf

def getDF(doc_tf):
    df_dict = {}
    for doc_id, tf_dict in doc_tf.items():
        for term in tf_dict:
            df_dict[term] = df_dict.get(term, 0) + 1
    return df_dict

def termTFIDF(term, tf_dict, df_dict, total_docs, token_count):
    return (1 + math.log(tf_dict[term])) / token_count * (math.log(total_docs / df_dict[term]) + 1)

def getTFIDF(tf_dict, df_dict, total_docs, token_count):
    tfidf_dict = {term: (0 if token_count == 0 else termTFIDF(term, tf_dict, df_dict, total_docs, token_count)) for term in tf_dict}
    return tfidf_dict

def termBM25(term, tf_dict, df_dict, doc_length, avg_doc_length, total_docs, token_count, k1=1.5, b=0.75):
    numerator = (tf_dict[term] / token_count) * (k1 + 1)
    denominator = (tf_dict[term] / token_count) + k1 * (1 - b + b * (doc_length / avg_doc_length))
    return math.log((total_docs - df_dict[term] + 0.5) / (df_dict[term] + 0.5) + 1) * (numerator / denominator)

def getBM25(tf_dict, df_dict, doc_length, avg_doc_length, total_docs, token_count):
    bm25_dict = {term: termBM25(term, tf_dict, df_dict, doc_length, avg_doc_length, total_docs, token_count) for term in tf_dict}
    return bm25_dict

def getVectors(vocab, doc_tfidf, doc_bm25):
    doc_ids = list(doc_tfidf.keys())
    num_docs = len(doc_ids)
    vocab_size = len(vocab)

    # sparse matrices instead of vectors for memory optimization
    doc_vecs_tfidf = dok_matrix((num_docs, vocab_size), dtype=np.float32)
    doc_vecs_bm25 = dok_matrix((num_docs, vocab_size), dtype=np.float32)

    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)} # map doc_id to indices

    for term_index, term in enumerate(vocab):
        for doc_id in doc_ids:
            if term in doc_tfidf[doc_id]:
                doc_vecs_tfidf[doc_id_to_index[doc_id], term_index] = doc_tfidf[doc_id][term]
                doc_vecs_bm25[doc_id_to_index[doc_id], term_index] = doc_bm25[doc_id][term]
    return doc_vecs_tfidf, doc_vecs_bm25, doc_id_to_index
                    
def index(docs, outfile_path):
    vocab = []
    doc_lengths = []
    avg_doc_length = []
    doc_tfidf = {}
    doc_bm25 = {}
    for doc in docs:
        text = process_text(doc['Text'])
        if len(text) == 0:
            pass
            #raise Exception(f"Empty body in document id: {doc['Id']}")
        doc_lengths.append(len(text))
        doc_tfidf[doc['Id']] = getTF(text)
    df_dict = getDF(doc_tfidf)
    vocab = list(df_dict.keys())
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)
    doc_num = 0
    for doc_id, tf_dict in doc_tfidf.items():
        token_count = sum(tf_dict.values())
        doc_tfidf[doc_id] = getTFIDF(tf_dict, df_dict, len(docs), token_count)
        doc_bm25[doc_id] = getBM25(tf_dict, df_dict, doc_lengths[doc_num], avg_doc_length, len(docs), token_count)
        doc_num += 1
    doc_vecs_tfidf, doc_vecs_bm25, doc_id_to_index = getVectors(vocab, doc_tfidf, doc_bm25)
    save_objects(vocab, df_dict, doc_bm25, doc_vecs_tfidf, doc_vecs_bm25, doc_id_to_index, outfile_path)

def save_objects(vocab, df_dict, doc_bm25, doc_vecs_tfidf, doc_vecs_bm25, doc_id_to_index, outfile_path):
    objects = [vocab, df_dict, doc_bm25, doc_vecs_tfidf, doc_vecs_bm25, doc_id_to_index]
    with open(outfile_path, 'wb') as f:
        pickle.dump(objects, f)

if __name__ == "__main__":
    sys.argv = ['indexer.py', 'Answers.json', 'index.pkl']
    if len(sys.argv) <= 1:
        raise Exception(f"Missing {2-(len(sys.argv)-1)} required argument(s), refer to README")
    docs = readJSON(sys.argv[1])
    index(docs, sys.argv[2])