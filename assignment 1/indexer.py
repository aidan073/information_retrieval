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
    
    @staticmethod
    def process_text(text):
        soup = BeautifulSoup(text, 'lxml')
        text = soup.get_text()
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
        return tokens
    
    def custom_BOW(self, tokens, token_limit, counter_limit):
        bow = {}
        for token in tokens:
            if bow.get(token, 0) < counter_limit:
                bow[token] = bow.get(token, 0) + 1
        return bow

    def index_docs(self):
        for document in self.documents:
            tokens = Indexer.process_text(document.get('Text'))
            bow = self.custom_BOW(tokens, 300, 5)
            document['Text'] = bow
    
    def get_doc(self, doc_id):
        return self.documents.get(doc_id)
    
    def compute_score(self, relevant_terms, total_terms, max_terms, weight):
        score = weight * (relevant_terms/total_terms) + (1-weight) * (total_terms/max_terms)
        return round(score, 3)
    
    def search(self, query):
        results = {}
        for document in self.documents:
            results[document['Id']] = 0
            if len(document['Text']) > 5: # only consider documents with greater than 5 unique tokens (may need to be higher)
                for term in query:
                    if term in document.get('Text'):
                        results[document['Id']] = results[document['Id']] + document['Text'][term]
                results[document['Id']] = self.compute_score(results[document['Id']], sum(document['Text'].values()), 300, 0.9) 
        return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    
    def save_index(self, outfile_path):
        with open(outfile_path, 'w') as f:
            json.dump(self.documents, f, indent=4)

if __name__ == "__main__":
    indexer = Indexer('Answers.json')
    indexer.index_docs()
    indexer.save_index('Index.json')