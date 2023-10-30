"""
Functionality to read text files that contain data encoded in the conll standard for NER:
e.g. word IOB-label
"""
import pandas as pd
from collections import defaultdict, Counter
import os



def from_file(file_path: str):
    """
    Read data from a conll encoded text file and represent it as pandas DF with two columns: (for the
    input sequences and for the labels)
    :param file_path: str: the path to the file
    :return: pd.DataFrame
    """
    dataset_predictors = defaultdict(list)
    dataset_labels = defaultdict(list)
    sentence_count = 0
    with open(file_path, 'r' ,encoding = 'utf8' ) as f:
        for line in f:
            if line=='\n':
                sentence_count+=1
                continue
            predictor, label = line.split(' ')
            dataset_predictors[sentence_count].append(predictor.strip())
            dataset_labels[sentence_count].append(label.strip())
    df = pd.DataFrame([dataset_predictors, dataset_labels]).T
    df.dropna(inplace=True)
    df.columns = ['predictors', 'labels']
    return df


def from_dir(data_path):
    """
    Wrapper around the from file function to process multiple conll files in a directory. Additional
    column is added: the file column holding the name of the file. Could be used for stratified split.
    :param data_path: str: path to the directory
    :return: pd.DataFrame
    """
    global_dataset = []
    for file in os.listdir(data_path):
        if not file.endswith('.conll'):
            continue
        file_path = os.path.join(data_path, file)
        df = from_file(file_path)
        df['file'] = file.split('.')[0]
        global_dataset.append(df)

    return pd.concat(global_dataset).dropna()