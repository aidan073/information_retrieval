import pickle
import indexer
import os
import json
import csv
import sys
import heapq
import numpy as np
from collections import Counter

# Read in queries from .json
def getQueries(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    return queries

# Read in indexed objects from .pkl
def getObjects(filepath):
    with open(filepath, 'rb') as f:
        objects = pickle.load(f)
    return objects

def queryVecs(query, vocab, df_dict, doc_bm25):
    """
    Convert a query to tfidf and bm25 vectors
    
    Args: 
    query (array): array of tokens representing the query
    vocab (array): array of tokens representing vector dimensions
    df_dict (dict): {term: df}
    doc_bm25 (dict): {doc: {term: bm25}}

    Returns:
    q_tfidf_vec (np_array): vector with dimensions = query tfidf scores for vocab, shape(1, len(vocab))
    q_bm25_vec: (np_array): vector with dimensions = query bm25 scores for vocab, shape(1, len(vocab))
    """
    if len(query) == 0:
        raise Exception("Empty query")
    
    vocab_size = len(vocab)
    word_counts = dict(Counter(query))

    q_tfidf_vec = np.zeros((1, vocab_size), dtype=np.float32)
    q_bm25_vec = np.zeros((1, vocab_size), dtype=np.float32)

    for term_index, term in enumerate(vocab):
        if term in query:
            q_tfidf_vec[0, term_index] = indexer.termTFIDF(term, word_counts, df_dict, len(doc_bm25), len(query))
            q_bm25_vec[0, term_index] = indexer.termBM25(term, word_counts, df_dict, 1, 1, len(doc_bm25), len(query))
    
    return q_tfidf_vec, q_bm25_vec

def batch_cosine_similarity(query_vector, doc_vectors):
    """
    Compute cossim for a query over all the docs

    Args:
    query_vector (np_array): vector with dimensions = query scores for vocab, shape(1, len(vocab))
    doc_vectors (np_array): array of vectors with dimensions = scores for vocab for each doc, shape(total_docs, len(vocab))
    

    Returns:
    cosine_similarities (np_array): array representing the cossim of a query with each doc, shape(1, total_docs)
    """
    query_norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
    doc_norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)

    query_norms[query_norms == 0] = 1
    doc_norms[doc_norms == 0] = 1

    normalized_queries = query_vector / query_norms
    normalized_docs = doc_vectors / doc_norms

    cosine_similarities = np.dot(normalized_queries, normalized_docs.T)

    return cosine_similarities

def search(queries, vocab, df_dict, doc_bm25, d_tfidf_vecs, d_bm25_vecs, doc_id_to_index, outfile_path_list):
    """
    Search a dict of queries over documents

    Args:
    queries (dict): dict containing queries to search
    vocab (array): array of tokens representing vector dimensions
    df_dict (dict): {term: df}
    doc_bm25 (dict): {doc: {term: bm25}}
    d_tfidf_vecs (np_array): array of vectors with dimensions = tfidf scores for vocab for each doc, shape(total_docs, len(vocab))
    d_bm25_vecs (np_array): array of vectors with dimensions = bm25 scores for vocab for each doc, shape(total_docs, len(vocab))
    doc_id_to_index (dict): {doc_id: location in np_arrays}
    outfile_path_list (array): index 0 = location to save tfidf results, index 1 = location to save bm25 results
    """
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

            results_tfidf = batch_cosine_similarity(query_tfidf_vec, d_tfidf_vecs)
            results_bm25 = batch_cosine_similarity(query_bm25_vec, d_bm25_vecs)

            results_tfidf_dict = {doc_id: results_tfidf[0, doc_id_to_index[doc_id]] for doc_id in doc_bm25}
            results_bm25_dict = {doc_id: results_bm25[0, doc_id_to_index[doc_id]] for doc_id in doc_bm25}

            top_tfidf = heapq.nlargest(100, results_tfidf_dict.items(), key=lambda item: item[1])
            top_bm25 = heapq.nlargest(100, results_bm25_dict.items(), key=lambda item: item[1])

            writeResult(top_tfidf, id, f)
            writeResult(top_bm25, id, f2)

            query_count += 1
            print(f"queries processed: {query_count}/{len(queries)}")
    finally:
        f.close()
        f2.close()

def writeResult(result, id, outfile):
    """
    Write a queries results to a file in TREC format

    Args:
    result (array): contains top 100 tuples (doc_id, score) for a query
    id (int): query id
    outfile (file): file to write results to
    """
    writer = csv.writer(outfile, delimiter='\t')
    rank = 1
    for doc_id, score in result:
        writer.writerow([id, 'Q0', doc_id, rank, score, 'vector_search'])
        rank += 1
        if rank >= 100:
            break

if __name__ == "__main__":
    #sys.argv = ['search.py', 'topics_1.json', 'index.pkl']
    if len(sys.argv) <= 1:
        raise Exception(f"Missing {2-(len(sys.argv)-1)} required argument(s), refer to README")
    queries = getQueries(sys.argv[1])
    objects = getObjects(sys.argv[2])
    print(len(objects[0]))
    search(queries, objects[0], objects[1], objects[2], objects[3], objects[4], objects[5], ['result_tfidf_1.tsv', 'result_bm25_1.tsv'])
