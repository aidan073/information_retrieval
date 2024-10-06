import json
import pickle
import math
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def readJSON(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    return docs

def process_text(text): # pre-process text
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text()
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return tokens

def collect_vocab(doc_text, vocab, df_dict):
    doc_tf = {}
    for token in doc_text:
        if token not in doc_tf.keys(): # each token only once
            df_dict[token] = df_dict.get(token, 0) + 1
            if token not in vocab:
                vocab.append(token)
        doc_tf[token] = doc_tf.get(token, 0) + 1
    return doc_tf

def getTFIDF(tf_dict, df_dict, total_docs, token_count):
    tfidf_dict = {term: (0 if token_count == 0 else (1 + math.log(tf_dict[term])) / token_count * (math.log(total_docs / df_dict[term]) + 1)) for term in tf_dict.keys()}
    return tfidf_dict

def getBM25(tf_dict, df_dict, doc_length, avg_doc_length, total_docs, token_count, k1=1.5, b=0.75):
    bm25_dict = {}
    for term in tf_dict:
        numerator = (tf_dict[term] / token_count) * (k1 + 1)
        denominator = (tf_dict[term] / token_count) + k1 * (1 - b + b * (doc_length / avg_doc_length))
        bm25_dict[term] = math.log((total_docs - df_dict[term] + 0.5) / (df_dict[term] + 0.5) + 1) * (numerator / denominator)
    return bm25_dict

def getVectors(vocab, doc_tfidf, doc_bm25):
    doc_vecs_tfidf = {}
    doc_vecs_bm25 = {}
    index = 0
    for term in vocab:
        for doc_id, tfidf_dict in doc_tfidf.items():
            doc_vecs_tfidf.setdefault(doc_id, np.zeros(len(vocab)))
            doc_vecs_bm25.setdefault(doc_id, np.zeros(len(vocab)))
            if term in tfidf_dict:
                doc_vecs_tfidf[doc_id][index] = tfidf_dict[term]
                doc_vecs_bm25[doc_id][index] = doc_bm25[doc_id][term]         
        index += 1
    return doc_vecs_tfidf, doc_vecs_bm25
                    
def index(docs, outfile_path):
    vocab = []
    doc_lengths = []
    avg_doc_length = []
    df_dict = {}
    doc_tfidf = {}
    doc_bm25 = {}
    for doc in docs:
        text = process_text(doc['Text'])
        if len(text) == 0:
            pass
            #raise Exception(f"Empty body in document id: {doc['Id']}")
        doc_lengths.append(len(text))
        doc_tfidf[doc['Id']] = collect_vocab(text, vocab, df_dict)
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)
    doc_num = 0
    for doc_id, tf_dict in doc_tfidf.items():
        token_count = sum(tf_dict.values())
        doc_tfidf[doc_id] = getTFIDF(tf_dict, df_dict, len(docs), token_count)
        doc_bm25[doc_id] = getBM25(tf_dict, df_dict, doc_lengths[doc_num], avg_doc_length, len(docs), token_count)
        doc_num += 1
    doc_vecs_tfidf, doc_vecs_bm25 = getVectors(vocab, doc_tfidf, doc_bm25)
    save_objects(vocab, df_dict, doc_bm25, doc_vecs_tfidf, doc_vecs_bm25, outfile_path)

def save_objects(vocab, df_dict, doc_bm25, doc_vecs_tfidf, doc_vecs_bm25, outfile_path):
    objects = [vocab, df_dict, doc_bm25, doc_vecs_tfidf, doc_vecs_bm25]
    with open(outfile_path, 'wb') as f:
        pickle.dump(objects, f)

if __name__ == "__main__":
    docs = readJSON('test.json')
    index(docs, 'index.pkl')