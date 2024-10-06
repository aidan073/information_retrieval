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

def collect_vocab(doc_text, vocab, df_dict, corpus_length):
    doc_tf = {}
    for token in doc_text:
        if token not in doc_tf.keys(): # each token only once
            df_dict[token] = df_dict.get(token, 0) + 1
            if token not in vocab:
                vocab.append(token)
        doc_tf[token] = doc_tf.get(token, 0) + 1
        corpus_length+=1
    return doc_tf, corpus_length

def getTFIDF(tf_dict, df_dict, total_docs, token_count):
    tfidf_dict = {term: (1 + math.log(tf_dict[term])) / token_count * (math.log(total_docs / df_dict[term]) + 1) for term in tf_dict.keys()}
    return tfidf_dict

def getBM25(term, document, corpus, avg_doc_length, k1=1.5, b=0.75):
    tf = term_frequency(term, document)
    idf = inverse_document_frequency(term, corpus)
    
    doc_length = len(document)
    
    # Apply BM25 formula
    numerator = tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
    
    return idf * (numerator / denominator)

def getVectors(vocab, doc_tfidf = None, ):
    doc_vecs = {}
    index = 0
    for term in vocab:
        for doc_id, tfidf_dict in doc_tfidf:
            doc_vecs.setdefault(doc_id, np.zeros(len(vocab)))
            if term in tfidf_dict:
                doc_vecs[doc_id][index] = tfidf_dict[term]         
        index += 1
    return doc_vecs
                    
def index(docs):
    vocab = []
    avg_doc_length = []
    df_dict = {}
    doc_tfidf = {}
    corpus_length = 0
    for doc in docs:
        text = process_text(doc['Text'])
        if len(text) == 0:
            raise Exception(f"Empty body in document id: {doc['Id']}")
        avg_doc_length.append(len(text))
        tf_dict, corpus_length = collect_vocab(text, vocab, df_dict, corpus_length)
        doc_tfidf[doc['Id']] = tf_dict
    avg_doc_length = sum(avg_doc_length) / len(avg_doc_length)
    for doc_id, tf_dict in doc_tfidf.items():
        token_count = sum(tf_dict.values())
        doc_tfidf[doc_id] = getTFIDF(tf_dict, df_dict, len(docs), token_count)

def save_index(docs, vocab, outfile_path):
    docs.append(vocab)
    with open(outfile_path, 'wb') as f:
        pickle.dump(docs, f)

if __name__ == "__main__":
    docs = readJSON('Answers.json')
    index(docs)