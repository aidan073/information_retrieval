import json
import pickle
import os
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def readJSON(filepath):
    with open(filepath, 'r') as f:
        docs = json.load(f)
    return docs

def process_text(text): # pre-process text
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text()
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stopwords]
    return tokens

def doc_tf_vocab(doc_text, vocab):
    doc_tf = {}
    for token in doc_text:
        doc_tf[token] = doc_tf.get(doc_tf, 0) + 1
        if token not in vocab:
            vocab.append(token)
    return doc_tf

def index(docs):
    for doc in docs:
        text = process_text(doc['Title'] + " " + doc['Body'])
        doc_tf_vocab(text)

def save_index(self, outfile_path):
    with open(outfile_path, 'wb') as f:
        pickle.dump(self.documents, f)

if __name__ == "__main__":
    docs = readJSON('topics_1')
    index(docs)