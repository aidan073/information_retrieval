import pandas as pd
import pyterrier as pt
import json
import os

with open('Answers.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)
df = pd.DataFrame(json_data)

index_location = os.path.join(os.getcwd(), 'index')
os.makedirs(index_location, exist_ok=True)

indexer = pt.DFIndexer(index_location, overwrite=True)
indexer.index(df[['Id', 'Text']])