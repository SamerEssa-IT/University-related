import math

import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
import datasets
import pyarrow as pa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import pickle
import scipy
import random
from tqdm import tqdm

path2textfiles = "../DataNephroTexts/input/"
path2diagnosefiles = "../DataNephroTexts/label/"


def is_text_lst_tokenized(path2corpus):
    try:
        text_lst = pd.read_pickle(path2corpus)
        text1 = np.asarray(text_lst[0])
        return bool(text1.ndim)
    except:
        return False


def is_text_lst_tfidf_vectorized(path2corpus):
    try:
        with open(path2corpus, 'rb') as f:
            loaded_texts = pickle.load(f)
        return type(loaded_texts) == scipy.sparse.csr.csr_matrix
    except:
        return False


def text_label_2_labeled_dataset(texts, unfiltered_labels, print_infos=False):
    '''
        - sorts out outliear-documents (which belongs to cluster .1 or cluster Nonde)
        - converts the passed text-label pair to datastes.Dataset type.
        - returns dataset in format: {"text": labeled_texts, "label": labels}
    '''
    # collect all text-label pairs, skipping unvalid labels
    labeled_texts = []
    labels = []
    skipped_labels = 0

    # throw out invalid labels:
    for i, l in enumerate(unfiltered_labels):
        try:
            label = int(l)
            if label < 0:
                skipped_labels += 1
                continue
        except:
            skipped_labels += 1
            continue
        labels.append(label)
        labeled_texts.append(texts[i])

    if print_infos:
        print("skipped " + str(skipped_labels))

    labels = label_list_as_int_list(labels)

    # convert it to a hf_dataset, that we can use our tools:
    df = pd.DataFrame({"text": labeled_texts, "label": labels})
    return datasets.Dataset(pa.Table.from_pandas(df))


def text_label_files_to_labeled_dataset(label_set,
                                        path2corpus="./database/bow_prepro_desc.pkl",
                                        df_cases_path="./database/df_cases.pkl", print_infos=False):
    '''
    - sorts out outliear-documents (which belongs to cluster .1 or cluster Nonde)
    - converts the pandas dataframe to datastes.Dataset type.
    '''

    df_cases = pd.read_pickle(df_cases_path)
    texts = pd.read_pickle(path2corpus)
    unfiltered_labels = df_cases["label_" + label_set]

    return text_label_2_labeled_dataset(texts, unfiltered_labels, print_infos)


def get_all_label_set_ids():
    df = pd.read_pickle("./database/df_cases.pkl")
    return [e[6:] for e in df.columns if "label_" in e]


def get_filename_label_tuple(label_set, get_micro_txt=True, df_cases_file="./database/df_cases.pkl"):
    '''
    returns textfilename_list, label_lists as ([filenames],[labels, as int list]))
    it will contain outlier labels (they have value None or -1)
    '''
    df_cases = pd.read_pickle(df_cases_file)
    if "label_" + label_set not in df_cases.columns:
        raise ValueError("label set " + label_set + " does not exist in df_cases!")
        return None
    # convert labels to integers:
    int_labels = label_list_as_int_list(df_cases["label_" + label_set])
    if get_micro_txt:
        return df_cases["description_text_files"], int_labels
    else:
        return df_cases["diagnosis_text_files"], int_labels


def get_amount_unique_labels(label_set, df_cases_file="./database/df_cases.pkl"):
    '''
    returns amount unique labels (does not count nan or -1 classes!!!).
    If label_set does not exist, you will get
    an error. If so, run generate_save_hf_dataset(...) to generate a labeled dataset
    of type datasets.Dataset  (datasets is a library from huggingface)
    '''
    df_cases = pd.read_pickle(df_cases_file)
    if "label_" + label_set not in df_cases.columns:
        raise ValueError("label set " + label_set + " does not exist in df_cases!")
        return None
    # convert labels to integers:
    labels = label_list_as_int_list(df_cases["label_" + label_set])
    has_none_labels = False
    for label in labels:
        if label == -1 or np.isnan(label) or label == None:
            has_none_labels = True
            return len(list(set(labels))) - 1

    return len(list(set(labels)))


def get_amount_reports(label_set):
    '''
    returns amount of reports which have a valid label (excluding -1 and NaN values)
    '''
    # train_test_dataset = load_labeled_dataset(label_set)
    # return len(train_test_dataset["label"])
    text, labels = get_filename_label_tuple(label_set)
    return len([l for l in labels if l >= 0])


def generate_save_hf_dataset(label_set="LDA", overwrite=True, lower=False):
    '''
    Generate a labeled dataset of type datasets.Dataset
    (datasets is a library from huggingface)
    and saves it under "./database/labeled_dataframes/labeld_dataset_" + label_set
    '''

    dataset_path = "./database/labeled_dataframes/labeld_dataset_" + label_set

    if os.path.exists(dataset_path):
        print(dataset_path + " already exists.")
        if overwrite:
            print("generating it new and overwrite " + dataset_path)
        else:
            print("skipping generation of " + dataset_path)
            return

    df_cases = pd.read_pickle("./database/df_cases.pkl")
    # print(df_cases.columns)

    # collect all text-label pairs, skipping unvalid labels!
    diag_text_rokenized = pd.read_pickle("./database/diag_lst_tokenized.pkl")
    texts = []
    labels = []
    diagnoses = []
    skipped_labels = 0

    # throw out invalid labels:
    print("creating " + dataset_path)
    for i, l in enumerate(df_cases["label_" + label_set]):
        try:
            label = int(l)
            if label < 0:
                skipped_labels += 1
                continue
        except:
            skipped_labels += 1
            continue
        labels.append(label)

        file_id = df_cases["description_text_files"][i]
        with open(path2textfiles + file_id, 'r') as f:
            if lower:
                texts.append(f.read().lower())
            else:
                texts.append(f.read())

        file_id = df_cases["diagnosis_text_files"][i]
        with open(path2diagnosefiles + file_id, 'r') as f:
            if lower:
                diagnoses.append(f.read().lower())
            else:
                diagnoses.append(f.read())

    print("skipped " + str(skipped_labels) + " labels")

    # convert to dataframe:
    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'diagnose': diagnoses
    })

    # convert pandas dataframe to huggingface dataset:
    hf_dataset = datasets.Dataset(pa.Table.from_pandas(df))

    '''# how to create a DatasetDict:
    test_split_length = 100
    hf_data_dict = datasets.DatasetDict({"train": datasets.Dataset(pa.Table.from_pandas(df[test_split_length:])),
                                         "test": datasets.Dataset(pa.Table.from_pandas(df[:test_split_length])),
                                         "unsupervised": hf_dataset})
    hf_data_dict.save_to_disk(dataset_path)'''

    # print("shape of " + dataset_path + ":")
    # print(hf_dataset)
    hf_dataset.save_to_disk(dataset_path)


def label_list_as_int_list(labels):
    '''
    converts a label list to a list of integers,
    regardles if its a list of floats or strings
    '''
    int_labels = []
    for i, l in enumerate(labels):
        try:
            int_labels.append(int(labels[i]))
        except:
            int_labels.append(-1)
    return int_labels


def get_splits_for_cross_val(dataset, fold_amount=10, stratified=True,
                             merge_classes=None, oversample=False):
    '''
    dataset should be sth which can be accessed via dataset['text'] and dataset['label']
    returns splits for k-fold-cross-validation as datasets.Dataset type
    with sth like merge_classes=[(0,1),(2,3,4)] you can merge the indexted classes to one class
    '''

    if merge_classes is not None:
        new_labels = [i for i in dataset['label']]
        for classes_to_merge in merge_classes:
            new_class_name = classes_to_merge[0]
            for c in classes_to_merge[1:]:
                for i, label in enumerate(dataset['label']):
                    if int(label) == int(c):
                        new_labels[i] = new_class_name

        dataset = text_label_2_labeled_dataset(dataset['text'], new_labels)

    if oversample:
        dataset = simple_oversampling(dataset)

    if stratified:
        skf = StratifiedKFold(n_splits=fold_amount, random_state=None, shuffle=False)
        for train_index, test_index in skf.split(dataset['text'], dataset['label']):
            yield dataset[train_index], dataset[test_index]
    else:
        folds = KFold(n_splits=fold_amount, shuffle=False)
        for train_index, test_index in folds.split(list(range(len(dataset)))):
            yield dataset[train_index], dataset[test_index]

def simple_oversampling(dataset):
    print("oversampling (without augmentation!)...")
    unique_labels = np.unique(dataset['label'])
    label_amount = [0 for x in range(len(unique_labels))]
    texts = []
    for i, l in enumerate(tqdm(unique_labels)):
        i_th_labels = dataset['label'] == l
        label_amount[i] = int(np.sum(i_th_labels))
        texts.append([dataset['text'][i] for i,label in enumerate(dataset['label']) if label == l])
    max_index = label_amount.index(max(label_amount))
    for i, l in enumerate(tqdm(unique_labels)):
        if i == max_index:
            continue
        amount_copies = label_amount[max_index] - label_amount[i]
        for x in range(amount_copies):
            new_element = {'label': l, 'text': random.choice(texts[i])}
            dataset = dataset.add_item(new_element)
    return dataset

def main():
    # args = argsparse_preamble()
    # generate_save_hf_dataset(args.clustered_data)

    # print label sets
    label_sets = get_all_label_set_ids()
    print(label_sets)

    # dirty fix of OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # plot histograms: how much docs do have the same label=cluster-index?
    for i, label_set in enumerate(label_sets):
        text, labels = get_filename_label_tuple(label_set)
        labels = np.asarray(label_list_as_int_list(labels))

        # plt.subplot(3, 3, i + 1)
        plt.close()
        label_num = get_amount_unique_labels(label_set)
        x = np.arange(label_num)
        h, b = np.histogram(labels, bins=label_num)
        plt.bar(x, height=h)
        plt.xticks(x, x)
        plt.title(label_set)
        plt.savefig("TextClustering/plots/" + label_set + "_histogram.png")


if __name__ == '__main__':
    main()
