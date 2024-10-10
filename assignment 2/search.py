import pickle
import indexer
import os
import json
import csv
import sys
import heapq
import numpy as np
from collections import Counter
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def getQueries(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    return queries

def getObjects(filepath):
    with open(filepath, 'rb') as f:
        objects = pickle.load(f)
    return objects

def queryVecs(query, vocab, df_dict, doc_bm25):

    if len(query) == 0:
        raise Exception("Empty query")
    
    vocab_size = len(vocab)
    word_counts = dict(Counter(query))
    q_tfidf_vec = dok_matrix((1, vocab_size), dtype=np.float32)
    q_bm25_vec = dok_matrix((1, vocab_size), dtype=np.float32)

    for term_index, term in enumerate(vocab):
        if term in query:
            q_tfidf_vec[0, term_index] = indexer.termTFIDF(term, word_counts, df_dict, len(doc_bm25), len(query))
            q_bm25_vec[0, term_index] = indexer.termBM25(term, word_counts, df_dict, 1, 1, len(doc_bm25), len(query))
    q_tfidf_vec = q_tfidf_vec.tocsr()
    q_bm25_vec = q_bm25_vec.tocsr()
    return q_tfidf_vec, q_bm25_vec

def compute_cosine_similarity(q_tfidf_vec, q_bm25_vec, d_tfidf_vec, d_bm25_vec):
    cos_sim_t = cosine_similarity(q_tfidf_vec, d_tfidf_vec)[0][0]
    cos_sim_b = cosine_similarity(q_bm25_vec, d_bm25_vec)[0][0]
    return cos_sim_t, cos_sim_b

def search(queries, vocab, df_dict, doc_bm25, d_tfidf_vecs, d_bm25_vecs, doc_id_to_index, outfile_path_list):

    if os.path.isfile(outfile_path_list[0]):
        os.remove(outfile_path_list[0])
    if os.path.isfile(outfile_path_list[1]):
        os.remove(outfile_path_list[1])

    f = open(outfile_path_list[0], 'a', newline='')
    f2 = open(outfile_path_list[1], 'a', newline='')
    try:
        query_count = 0
        for query in queries:
            id = query['Id']
            text = query['Title'] + query['Body']
            text = indexer.process_text(text)
            query_tfidf_vec, query_bm25_vec = queryVecs(text, vocab, df_dict, doc_bm25)
            results_tfidf = {}
            results_bm25 = {}
            for doc_id in doc_bm25:
                results_tfidf[doc_id], results_bm25[doc_id] = compute_cosine_similarity(query_tfidf_vec, query_bm25_vec, d_tfidf_vecs[doc_id_to_index[doc_id]], d_bm25_vecs[doc_id_to_index[doc_id]])
            top_tfidf = heapq.nlargest(100, results_tfidf.items(), key=lambda item: item[1])
            top_bm25 = heapq.nlargest(100, results_bm25.items(), key=lambda item: item[1])
            writeResult(top_tfidf, id, f)
            writeResult(top_bm25, id, f2)
            query_count+=1
            print(f"{query_count}/{len(queries)}")
    finally:
        f.close()
        f2.close()

def writeResult(result, id, outfile):
    writer = csv.writer(outfile, delimiter='\t')
    rank = 1
    for doc_id, score in result:
        writer.writerow([id, 'Q0', doc_id, rank, score, 'vector_search'])
        rank += 1
        if rank >= 100:
            break
sys.argv = ['search.py', 'topics_1.json', 'index.pkl']
if len(sys.argv) <= 1:
    raise Exception(f"Missing {2-(len(sys.argv)-1)} required argument(s), refer to README")
queries = getQueries(sys.argv[1])
objects = getObjects(sys.argv[2])
print(len(objects[0]))
search(queries, objects[0], objects[1], objects[2], objects[3], objects[4], objects[5], ['result_tfidf_1.tsv', 'result_bm25_1.tsv'])