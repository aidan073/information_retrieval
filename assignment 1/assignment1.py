import indexer
import json
import csv
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def search_topics(in_file, out_file):
    indexer1 = indexer.Indexer('Index.json')
    with open(in_file, 'r', encoding='utf-8') as f:
        topics = json.load(f)
        with open(out_file, 'w', newline='') as f2:
            writer = csv.writer(f2, delimiter='\t')
            for topic in topics:
                query = indexer.Indexer.process_text(topic['Body'])
                query = list(set(query)) # remove duplicates
                result = indexer1.search(query)
                rank = 1
                for key, val in result.items():
                    row = [topic['Id'], 'Q0', key, rank, val, 'BOW_Search']
                    writer.writerow(row)
                    rank+=1
                    if rank > 100:
                        break

# query = indexer.Indexer.process_text("<blockquote>  <p>Right now, Puzzling is seeing an unprecedented wave of riddles and ciphers which have completely flooded the main site. I firmly believe in “if you can’t beat them, join them”.<br>   Disloyalty!<br>  I do know that’s what you will say but let me tell you, to stay relevant, you got to adapt. (Nothing much to see here. Go on.)<br>  What we have here is that the site quality suffers and there’s not enough variety.   Absurd!   Very well, if that’s what you think about the problem, then, we can only count the good ones.   Except, post new puzzles!!</p>    <p>On my prefix, you may sit.<br>  Soccer player is military(3)<br>  My suffix is meant to be hit.<br>  odd rat’s painting(4)<br>  Element left in God(4)<br>  My infix is in chemicals’ name.<br>  Plan recreation of a resume(7)<br>  Iron has four number(7)<br>  On a whole, I am a game.  </p>    <p>Oops, the enumerations and the two parts seem to have been jumbled here, that is, I can't confirm whether the enumerations in the clues are correct or not. And there is an extra 6 which you have to use.  </p></blockquote><p>The question is, \"What is Missing here\"?</p><p>Hint:  </p><blockquote class=\"spoiler\">  <p> Synchronous has gotten one half of the steganography part. You need to solve that as a crossword clue. There's another part of the steganography as well.</p></blockquote>")
# query = list(set(query))
# indexer1 = indexer.Indexer('Index.json')
# res = indexer1.search(query)
# pass

search_topics('topics_1.json', 'result_binary_2.tsv')