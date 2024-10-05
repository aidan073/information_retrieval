import json
import pickle
import math
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

def getTFIDF(tf_dict, df_dict, total_docs):
    total_terms = tf_dict.values().sum()
    tfidf_dict = {term: tf_dict[term]/ total_terms * (math.log(total_docs / df_dict[term]) + 1) for term in tf_dict.keys()}
    return tfidf_dict

def index(docs):
    vocab = []
    df_dict = {}
    doc_tfidf = {}
    for doc in docs:
        text = process_text(doc['Text'])
        tf_dict = collect_vocab(text, vocab, df_dict)
        doc_tfidf[doc['Id']] = tf_dict
    for doc_id, tf_dict in doc_tfidf.items():
        token_count = sum(tf_dict.values())
        doc_tfidf[doc_id] = getTFIDF(tf_dict, df_dict, len(docs), token_count)

def save_index(self, outfile_path):
    with open(outfile_path, 'wb') as f:
        pickle.dump(self.documents, f)

if __name__ == "__main__":
    docs = readJSON('Answers.json')
    index(docs)