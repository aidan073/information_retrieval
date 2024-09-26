import json
import pickle
import os
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

class Indexer:
    """
    A class to index topics and search over the index
    """
    def __init__(self, file_path):
        """
        Indexer constructor, loads documents into Indexer object

        Args:
        file_path (str): Index path
        """
        _, file_type = os.path.splitext(file_path)
        if file_type == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        elif file_type == '.pkl' or file_type == '.pickle':
            with open(file_path, 'rb') as f:
                self.documents = pickle.load(f)
    
    @staticmethod
    def process_text(text): # pre-process text
        soup = BeautifulSoup(text, 'lxml')
        text = soup.get_text()
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
        return tokens

    def index_docs(self): # replace each document text with a set of its tokens
        for document in self.documents:
            tokens = Indexer.process_text(document.get('Text'))
            document['Text'] = set(tokens)
    
    def get_doc(self, doc_id):
        return self.documents.get(doc_id)
    
    def compute_score(self, query_set, document_set):
        """
        Compute retrieval score for a query -> document.
        
        Args: 
        query_set (set): set of query tokens
        document_set (set): set of document tokens

        Returns:
        Retrieval score
        """
        if (query_set and document_set) != 0:
            return len(query_set & document_set) / len(query_set | document_set)
        return 0
    
    def search(self, query_set):
        """
        Search a query over the index.

        Args:
        query (list): Tokenized/pre-processed query
        
        returns:
        Sorted dict of documents based on retrieval scores
        """
        results = {}
        for document in self.documents:
            results[document['Id']] = self.compute_score(document['Text'], query_set) 
        return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    
    def save_index(self, outfile_path):
        # pickle documents
        with open(outfile_path, 'wb') as f:
            pickle.dump(self.documents, f)

if __name__ == "__main__":
    indexer = Indexer('Answers.json')
    indexer.index_docs()
    indexer.save_index('Index.pkl')