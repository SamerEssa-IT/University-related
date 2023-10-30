# -*- coding: iso-8859-1 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

import sys
import database_preparation.utils_labeled_datasets as dt

# for training validation:
import TextClassification.classification_metrics as cls_metrics
from sklearn.decomposition import TruncatedSVD

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import nltk
import datasets
import pyarrow as pa
import pickle

fold_amount = 10

#%%
# for tfidf vectorizer
def identity(words):
    return words

def create_pipeline(estimator, reduction=False):
    '''
    construct a pipeline with sklearn.pipeline
    pased estimator will be the last element of the pipeline
    using tfidf as vectorizer
    '''
    steps = []

    steps.append(
        ('vectorize', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False))
    )

    if reduction:
        steps.append((
            'reduction', TruncatedSVD(n_components=1000)
        ))

    # Add the estimator
    steps.append(('classifier', estimator))
    return Pipeline(steps)

def cross_validate_with_simple_SVM(label_set, path2corpus = "./database/bow_prepro_diag.pkl", path2dfcases='./database/df_cases.pkl'):
    """
    trains a simple SVM with the given data
    returns 10-fold-cross-validated accuracy value
    """

    print(f"Calculating SVM-classification performance of {label_set} cluster-setr "
          f"with text corpus {path2corpus}.")

    metrics = cls_metrics.ClassificationMetrics(label_set)

    #print("train SVM for cluster " + label_set + " with " + path2corpus + ".")

    text_lst = pd.read_pickle(path2corpus)
    text1 = np.asarray(text_lst[0])
    corpus_is_tokenized = bool(text1.ndim)
    del text1, text_lst

    if corpus_is_tokenized:
        dataset = dt.text_label_files_to_labeled_dataset(label_set,
                                                         path2corpus
                                                         , path2dfcases, False)
    else:
        dataset_raw = dt.text_label_files_to_labeled_dataset(label_set,
                                                             path2corpus
                                                             , path2dfcases, False)
        # tokenize
        tokenized_texts = []
        for t_text in dataset_raw['text']:
            tokenized_texts.append(nltk.tokenize.word_tokenize(t_text, language='german'))
        df = pd.DataFrame({'text': tokenized_texts, 'label': dataset_raw['label']})
        dataset = datasets.Dataset(pa.Table.from_pandas(df))

    # 10-fold crosss validation:
    folds = KFold(n_splits=10, shuffle=False)
    for i, (train_index, test_index) in enumerate(folds.split(list(range(len(dataset))))):
        train_dataset = dataset[train_index]
        test_dataset = dataset[test_index]
        pipe = create_pipeline(SGDClassifier())
        pipe.fit(train_dataset['text'], train_dataset['label'])
        y_pred = pipe.predict(test_dataset['text'])

        metrics.update_metrics(test_dataset['label'], y_pred, False)

    # train_save_SVM_for_clusterset_evaluation(label_set)
    # metrics.save_scores_to_disk("diagnose_texts_with_SGD")

    return metrics

def cross_validate_label_corpus_with_simple_SVM(labels, path2corpus = "./database/bow_prepro_diag.pkl", sample = True):
    """
    trains a simple SVM with the given data
    returns 10-fold-cross-validated accuracy value
    """

    texts = pd.read_pickle(path2corpus)
    from database_preparation.utils_labeled_datasets import text_label_2_labeled_dataset

    metrics = cls_metrics.ClassificationMetrics("temp")

    #print("train SVM for cluster " + label_set + " with " + path2corpus + ".")

    text_lst = pd.read_pickle(path2corpus)
    text1 = np.asarray(text_lst[0])
    corpus_is_tokenized = bool(text1.ndim)
    del text1, text_lst

    if corpus_is_tokenized:
        dataset = text_label_2_labeled_dataset(texts,labels)
    else:
        dataset_raw = text_label_2_labeled_dataset(texts,labels)

        # tokenize
        tokenized_texts = []
        for t_text in dataset_raw['text']:
            tokenized_texts.append(nltk.tokenize.word_tokenize(t_text, language='german'))
        df = pd.DataFrame({'text': tokenized_texts, 'label': dataset_raw['label']})
        dataset = datasets.Dataset(pa.Table.from_pandas(df))

    # 10-fold crosss validation:
    folds = KFold(n_splits=10, shuffle=False)
    for i, (train_index, test_index) in enumerate(folds.split(list(range(len(dataset))))):
        train_dataset = dataset[train_index]
        test_dataset = dataset[test_index]
        pipe = create_pipeline(SGDClassifier())
        pipe.fit(train_dataset['text'], train_dataset['label'])
        y_pred = pipe.predict(test_dataset['text'])

        metrics.update_metrics(test_dataset['label'], y_pred, False)
        if sample:
            return metrics.scores['accuracy']

    # train_save_SVM_for_clusterset_evaluation(label_set)
    # metrics.save_scores_to_disk("diagnose_texts_with_SGD")

    return np.mean(metrics.scores['accuracy'])

def train_SVM_with_clusterset(label_set, path2corpus = "./database/bow_prepro_diag.pkl", path2dfcases='./database/df_cases.pkl'):
    """
    trains ans saves a svm, trained with the whole data under as:
    "./ModelTestingAndExplaining/models/SVM_trained_with_" + label_set + "_clustered.pkl"
    """

    print("train SVM for cluster " + label_set + " with " + path2corpus + ".")

    text_lst = pd.read_pickle(path2corpus)
    text1 = np.asarray(text_lst[0])
    corpus_is_tokenized = bool(text1.ndim)
    del text1, text_lst

    if corpus_is_tokenized:
        dataset = dt.text_label_files_to_labeled_dataset(label_set,
                                                         path2corpus
                                                         , path2dfcases, False)
    else:

        dataset_raw = dt.text_label_files_to_labeled_dataset(label_set,
                                                             path2corpus
                                                             , path2dfcases, False)
        # tokenize
        tokenized_texts = []
        for t_text in dataset_raw['text']:
            tokenized_texts.append(nltk.tokenize.word_tokenize(t_text, language='german'))
        df = pd.DataFrame({'text': tokenized_texts, 'label': dataset_raw['label']})
        dataset = datasets.Dataset(pa.Table.from_pandas(df))

    pipe = create_pipeline(SVC(probability=True, kernel='linear'))
    '''svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    pipe = make_pipeline(make_pipeline(
        TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False),svd),
        SVC(C=150, gamma=2e-2, probability=True))'''
    pipe.fit(dataset['text'], dataset['label'])
    path = "./ModelTestingAndExplaining/models/SVM_trained_with_" + label_set + "_clustered.pkl"
    pickle.dump(pipe, open(path, 'wb'))


def update_cls_metric(label_set, cls_accuracy):
    file_name = label_set + "_Diagnosis"
    file_name = file_name.replace('KMeans', 'kmeans')
    file_name = file_name.replace('d2v', 'doc2vec')
    file_path = "TextClustering/cluster_metrics/" + file_name + ".pkl"
    try:
        scores = pd.DataFrame(pd.read_pickle(file_path))
    except:
        return
    if 'cls accuracy' in scores.index:
        scores[file_name]['cls accuracy'] = cls_accuracy
        new_scores = scores
    else:
        vals = list(scores[file_name])
        new_index = scores.index.append(pd.Index(['cls accuracy']))
        vals.append(cls_accuracy)
        new_scores = pd.DataFrame({file_name: vals}, index=new_index)

    new_scores.to_pickle(file_path)


def update_cls_metric_for_each_clusterset():
    '''
    does 10-fold-cross-validation with a svm for each cluster-set saved in './database/df_cases.pkl'
    using always the text in 'database/diag_lst_tokenized.pkl'
    '''
    label_sets = dt.get_all_label_set_ids()
    # label_sets = ["German_BERT"]
    for label_set in label_sets:
        accuracy = np.mean(cross_validate_with_simple_SVM(label_set,
                                                  'database/diag_lst_tokenized.pkl',
                                                  './database/df_cases.pkl').scores['accuracy'])
        print("svm-cls-accuracy of cluster set "+label_set+": "+str(accuracy))
        update_cls_metric(label_set, accuracy)


def main():
    #update_cls_metric_for_each_clusterset()
    cluster_set_name = "German_BERT"
    #text_data = 'database/darmischaemie_prostata_txt_lst.pkl' cluster_set_dict = './database/df_cases2.pkl'
    text_data  = 'database/diag_lst.pkl'
    #text_data = 'database/diag_lst_tokenized.pkl'
    cluster_set_dict = './database/df_cases.pkl'
    train_SVM_with_clusterset(cluster_set_name, text_data, cluster_set_dict)


if __name__ == '__main__':
    main()
