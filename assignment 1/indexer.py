import json
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

stop_words = set(stopwords.words('english'))

class Indexer:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

    def process_text(self, text):
        soup = BeautifulSoup(text, 'lxml')
        text = soup.get_text()
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
        bow = Counter(tokens)
        return bow
    
    def index_docs(self):
        for document in self.documents:
            bow = self.process_text(document.get('Text'))
            document['Text'] = bow
    
    def get_doc(self, doc_id):
        return self.documents.get(doc_id)
    
    def search_keyword(self, keyword):
        results = []
        for document in self.documents:
            if keyword.lower() in document.get('Text'):
                results.append(document.get['Id'])

        return results
    
    def save_index(self, outfile_path):
        with open(outfile_path, 'w') as f:
            json.dump(self.documents, f, indent=4)

if __name__ == "__main__":
    indexer = Indexer('Answers.json')
    indexer.index_docs()
    indexer.save_index('Index.json')