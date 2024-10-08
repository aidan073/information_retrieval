import pickle
import indexer
import os
import json
import csv
import sys
import numpy as np
from numpy.linalg import norm
from collections import Counter

def getQueries(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    return queries

def getObjects(filepath):
    with open(filepath, 'rb') as f:
        objects = pickle.load(f)
    return objects

def queryVecs(query, vocab, df_dict, doc_bm25):
    query_tfidf = np.array([])
    query_bm25 = np.array([])
    word_counts = Counter(query)
    if len(query) == 0:
        raise Exception("Empty query")
    for term in vocab:
        if term in query and term in df_dict:
            query_tfidf = np.append(query_tfidf, indexer.termTFIDF(term, word_counts, df_dict, len(doc_bm25), len(query)))
            query_bm25 = np.append(query_bm25, indexer.termBM25(term, word_counts, df_dict, 1, 1, len(doc_bm25), len(query)))
    return query_tfidf, query_bm25

def compareVecs(q_tfidf_vec, q_bm25_vec, d_tfidf_vec, d_bm25_vec):
    tfidf_cos_sim = np.dot(q_tfidf_vec, d_tfidf_vec)/(norm(q_tfidf_vec)*norm(d_tfidf_vec))
    bm25_cos_sim = np.dot(q_bm25_vec, d_bm25_vec)/(norm(q_bm25_vec)*norm(d_bm25_vec))
    return tfidf_cos_sim, bm25_cos_sim

def search(queries, vocab, df_dict, doc_bm25, d_tfidf_vecs, d_bm25_vecs, outfile_path_list):
    override_check = 1
    for query in queries:
        id = query['Id']
        text = query['Title'] + query['Body']
        text = indexer.process_text(text)
        query_tfidf, query_bm25 = queryVecs(text, vocab, df_dict, doc_bm25)
        results_tfidf = {}
        results_bm25 = {}
        for doc_id in doc_bm25:
            results_tfidf[doc_id], results_bm25[doc_id] = compareVecs(query_tfidf, query_bm25, d_tfidf_vecs[doc_id], d_bm25_vecs[doc_id])
        results_tfidf = dict(sorted(results_tfidf.items(), key=lambda item: item[1], reverse=True))
        results_bm25 = dict(sorted(results_bm25.items(), key=lambda item: item[1], reverse=True))
        writeResult(results_tfidf, id, outfile_path_list[0], override_check)
        writeResult(results_bm25, id, outfile_path_list[1], override_check)
        override_check = 0

def writeResult(result_dict, id, outfile_path, override_check):
    if override_check and os.path.isfile(outfile_path):
        os.remove(outfile_path)
    with open(outfile_path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        rank = 1
        for doc_id, score in result_dict.items():
            writer.writerow([id, 'Q0', doc_id, rank, score, 'vector_search'])
            rank += 1
            if rank >= 100:
                break
sys.argv = ['search.py', 'topics_1.json', 'index.pkl']
if len(sys.argv) <= 1:
    raise Exception(f"Missing {2-(len(sys.argv)-1)} required argument(s), refer to README")
queries = getQueries(sys.argv[1])
objects = getObjects(sys.argv[2])
search(queries, objects[0], objects[1], objects[2], objects[3], objects[4], ['result_tfidf_1.tsv', 'result_bm25_1.tsv'])