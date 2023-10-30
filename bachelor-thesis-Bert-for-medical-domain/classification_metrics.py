import time
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, cohen_kappa_score
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
import pandas as pd
import pickle

sys.path.append(os.getcwd())

class ClassificationMetrics(object):

    def __init__(self, model_name, metrics_save_name="metrics_new", **kwargs):
        self.scores = {
            'name': model_name,
            'fold_amount': 0,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'cohen_kappa': [],
            'time': [],
        }

        self.y_preds = []
        self.y_tests = []

        # create classification-metrics folder if not exist:
        pp = "./TextClassification/cls_metrics"
        if not os.path.isdir(pp):
            os.makedirs(pp)

        # create subfolder for our metrics if not exist:
        self.metrics_path = "./TextClassification/cls_metrics/"+metrics_save_name+"/"
        if not os.path.isdir(self.metrics_path):
            os.makedirs(self.metrics_path)

        # save paths:
        self.json_file_path = "none"
        self.object_dir = "none"

    def update_metrics(self, y_test, y_pred, print_cls_report=False, start_time=None):
        '''
        call this for each test run if you do k-fold-cross-validation
        '''
        if print_cls_report:
            print(classification_report(y_test, y_pred))

        self.y_preds.append(y_pred)
        self.y_tests.append(y_test)

        self.scores['fold_amount'] += 1

        if start_time != None:
            self.scores['time'].append(time.time() - start_time)
        else:
            self.scores['time'].append(-1)

        self.scores['accuracy'].append(accuracy_score(y_test, y_pred))

        # the ability of the classifier not to label as positive a sample that is negative - tp / (tp + fp)
        # -> precision = 1 -> This class was detected perfectly. There are only TPs! (true positives)
        # -> precision = 0.75 ->  There are sine false positives! -> sometimes the machine thought it was class A, but it wasn't
        self.scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))

        # the ability of the classifier to find all the positive samples - tp / (tp + fn)
        self.scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))

        self.scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))

        # cohen_kappa = fleiss kappa with 2 raters?
        # The kappa score measures the degree of agreement between
        # the two evaluators, also known as inter-rater reliability
        self.scores['cohen_kappa'].append(cohen_kappa_score(y_test, y_pred))

    def clean_class_score_table(self, df):
        df.drop(['accuracy', 'macro avg', 'weighted avg'], 1, inplace=True)
        df.drop(['precision', 'recall'], 0, inplace=True)
        df = df.T
        # round f1-values
        for i, x in enumerate(df['f1-score']):
            df['f1-score'][i] = round(x, 3)
        # edit suport entries:
        integer_support = [str(x)[:-2] for x in df['support']]
        df['support'] = integer_support
        df.sort_values(by=['f1-score'], inplace=True, ascending=False)
        return df

    def get_merged_predictions(self):
        merged_y_tests = []
        merged_y_preds = []
        for i in range(0, len(self.y_tests)):
            for x in self.y_preds[i]:
                merged_y_preds.append(x)
            for y in self.y_tests[i]:
                merged_y_tests.append(y)
        return merged_y_tests, merged_y_preds

    def classes_scores(self, prediction_set=0):
        '''
        returns some scored for each class
        '''
        if prediction_set < 0:

            merged_y_tests, merged_y_preds = self.get_merged_predictions()

            dic = classification_report(merged_y_tests, merged_y_preds,
                                        output_dict=True)
            df = pd.DataFrame(dic)
            return self.clean_class_score_table(df)

        else:
            dic = classification_report(self.y_tests[prediction_set], self.y_preds[prediction_set],
                                           output_dict=True)
            df = pd.DataFrame(dic)
            return self.clean_class_score_table(df)


    def plot_confusion_matrix(self, labels, prediction_set=0, plot=False, save=True,
                              filename='confusion_matrix', title=None,
                              normalized=True, annot = False, colormap='gray'):
        if title == None:
            title = filename
        if prediction_set < 0:
            y_test, y_pred = self.get_merged_predictions()
        else:
            y_test = self.y_tests[prediction_set]
            y_pred = self.y_preds[prediction_set]
        try:
            conf_matrix = np.asarray(confusion_matrix(y_test , y_pred,labels=labels),dtype=float)

            if normalized:
                for y, row in enumerate(conf_matrix[:]):
                    sum_apperiance = np.sum(row)
                    for x, pred_amount in enumerate(row):
                        if sum_apperiance == 0:
                            row[x] = 0
                        else:
                            row[x] = round(pred_amount / sum_apperiance,2)
                    conf_matrix[y] = row
        except ValueError:
            if labels[0] == 0:
                labels2=['class'+str(a) for a in range(len(labels))]
                try:
                    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels2)
                except ValueError:
                    print("confusion_matrix generation failed.")
                    print("labels:")
                    print(labels)
                    print("y_test:")
                    print(self.y_tests[prediction_set])
                    print("y_pred:")
                    print(self.y_preds[prediction_set])
                    return
            else:
                print("confusion_matrix generation failed.")
                print("labels:")
                print(labels)
                print("y_test:")
                print(self.y_tests[prediction_set])
                print("y_pred:")
                print(self.y_preds[prediction_set])
                return

        #print(conf_matrix)
        df_cm = pd.DataFrame(conf_matrix, labels, labels)
        sn.set(font_scale=1.4)  # for label size

        if plot or save:
            plt.close()
            if normalized:
                hm = sn.heatmap(df_cm, annot=annot, vmin=0, vmax=1, cmap=colormap) # this makes problems???
            else:
                hm = sn.heatmap(df_cm, annot=annot, annot_kws={"size": 10}, cmap=colormap)

            plt.xlabel("predicted", fontsize=14)
            plt.ylabel("true", fontsize=14)
            plt.title(title, fontsize=16)

        if plot:
            plt.show()
        if save:
            figure = hm.get_figure()
            save_path = "TextClassification/plots/"+filename+".png"
            try:
                figure.savefig(save_path, dpi=300)
                print("generated "+save_path)
            except FileNotFoundError:
                os.mkdir("TextClassification/plots")
                figure.savefig(save_path, dpi=300)
                print("generated " + save_path)

    def save_scores_to_disk(self, labelset):
        '''
        if file already exists, we append the new score as new row
        '''

        #  save scores as table, appending if modelname already exist:
        self.json_file_path = self.metrics_path+labelset+"_clustered_all_classifiers.json"
        if os.path.isfile(self.json_file_path):
            # add number to name, if model appears already in json file:
            with open(self.json_file_path, 'r') as f:
                amount_same_name = 0
                for line in f:
                    scores = json.loads(line)
                    if self.scores['name'] in scores["name"]:
                        amount_same_name += 1
                if amount_same_name > 0:
                    self.scores['name']=self.scores['name']+"_"+str(amount_same_name+1)

        with open(self.json_file_path, 'a') as f:
            f.write(json.dumps(self.scores) + "\n")

    def pickle_object(self, labelset, model_name='default'):
        # pickles whole object

        if model_name == 'default':
            model_name = self.scores['name']
        self.object_dir = self.metrics_path + labelset + "_clustered_" + model_name + "_classified.pickle"
        with open(self.object_dir, 'wb') as f:
            pickle.dump(self, f)


def print_results_as_latextable(jsonfile, print_only_f1_kappa=True):
    '''
    returns the results as latex table.
    expecting a jsonfile (path to json file) as saved my the metrics object
     you can optain the jsonfile of a metrics object via metrics.json_file_path
    '''

    if print_only_f1_kappa:
        print("================== " + jsonfile + " ==================")
        fields = [key for key in ClassificationMetrics(None).scores.keys()]
        to_remove = ["fold_amount","accuracy","precision","recall"]
        for remove in to_remove:
            fields.remove(remove)
        table = []
        with open(jsonfile, 'r') as f:
            for idx, line in enumerate(f):
                scores = json.loads(line)
                row = [scores['name']]

                for field in fields[1:]:
                    row.append("{:0.3f}".format(np.mean(scores[field])))

                table.append(row)

        # sort over f1 score:
        table.sort(key=lambda r: r[1], reverse=True)
        # print(tabulate.tabulate(table, headers=fields))
    else:
        print("================== " + jsonfile + " ==================")
        fields = [key for key in ClassificationMetrics(None).scores.keys()]
        table = []
        with open(jsonfile, 'r') as f:
            for idx, line in enumerate(f):
                scores = json.loads(line)
                row = [scores['name'], scores['fold_amount']]

                for field in fields[2:]:
                    row.append("{:0.3f}".format(np.mean(scores[field])))

                table.append(row)

        # sort over f1 score:
        table.sort(key=lambda r: r[5], reverse=True)
        # print(tabulate.tabulate(table, headers=fields))


    # export it to df and than to latex table:
    df = pd.DataFrame(columns=fields)
    for i, field in enumerate(fields):
        # df.append()
        df[field] = [e[i] for e in table]

    df.drop(columns=['time'], axis=1, inplace=True)
    as_latex = df.to_latex(index=False)
    print(as_latex)

    return as_latex


def main():
    y_true = [0,1,0,1,0,2]
    y_pred = [1,1,0,1,0,2]

    metrics = ClassificationMetrics("metrics_test")
    metrics.update_metrics(y_true,y_pred)
    metrics.save_scores_to_disk("testitest")
    metrics.pickle_object("testitest")
    metrics.plot_confusion_matrix([i for i in range(3)],0,True,True, colormap="gist_heat")

if __name__ == "__main__":
    main()


