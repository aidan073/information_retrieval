from bs4 import BeautifulSoup
import pandas as pd
import pyterrier as pt
import shutil
import json
import os

def make_path(folder_to_create: str):
    """Create new folder in cwd

    Args:
        folder_to_create (str): name of folder to create in cwd

    Returns:
        index_location (str): file path of new folder
    """
    folder_path = os.path.join(os.getcwd(), folder_to_create)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(folder_path)
    return folder_path

# parse html into plain text
def parse_html(text):
    soup = BeautifulSoup(text, 'lxml')
    return soup.get_text()

def json_to_df(path):
    """Convert json file to pandas dataframe, formatted for pyterrier
    
    Args:
        path (str): path to json file

    Returns:
        df (dataframe): pandas dataframe
    """
    with open('Answers.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df = pd.DataFrame(json_data)
    df = df.rename(columns={'Id': 'docno', 'Text': 'text', 'Score': 'popularity'}) # pyterrier required format
    df['text'] = df['text'].apply(parse_html)
    return df

def df_to_indexes(df):
    """Convert df to indexes

    Args:
        df (dataframe): pandas dataframe

    Returns:
        index_ref: reference to indexes
    """
    index_location = make_path('index')
    indexer = pt.IterDictIndexer(index_location, meta={'docno': 10, 'text': 5000, 'popularity': 10})
    index_ref = indexer.index(df.to_dict(orient="records"))
    return index_ref

df = json_to_df('Answers.json')
index_ref = df_to_indexes(df)
