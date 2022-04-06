import csv
import pickle
import pandas as pd
import numpy as np


# takes prediction, headline and body id columns, and filename to produce answer.csv format
def predictions_to_csv(predictions, headline, body_id, filename):
    pred = pd.DataFrame()
    pred['Headline'] = headline
    pred['Body ID'] = body_id
    pred_label = []
    for i in range(len(predictions)):
        if predictions[i] == 0:
            pred_label.append('agree')
        elif predictions[i] == 1:
            pred_label.append('disagree')
        elif predictions[i] == 2:
            pred_label.append('discuss')
        else:
            pred_label.append('unrelated')
    pred['Stance'] = pred_label
    pred.to_csv(filename, encoding='utf-8', index=False)


def predictions_to_csv2(predictions, headline, body_id, filename):
    classes = np.argmax(predictions, axis=1)
    pred = pd.DataFrame()
    pred['Headline'] = headline
    pred['Body ID'] = body_id
    pred_label = []
    for i in range(len(predictions)):
        if classes[i] == 0:
            pred_label.append('agree')
        elif classes[i] == 1:
            pred_label.append('disagree')
        elif classes[i] == 2:
            pred_label.append('discuss')
        else:
            pred_label.append('unrelated')
    pred['Stance'] = pred_label
    pred.to_csv(filename, encoding='utf-8', index=False)


# 2d list to csv
def list_to_csv_2d(tokens_list, filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(tokens_list)
    print(str(len(tokens_list)) + ' ' + filename)


def list_to_csv(tokens_list, filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(tokens_list)
    print(str(len(tokens_list)) + ' ' + filename)


def csv_to_list(filename):
    return list(csv.reader(open(filename)))


def list_to_sentences(data):
    new_list = []
    for sentences in data:
        new_list.append(' '.join(sentences))
    return new_list


def pickle_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def unpickle_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        return model


def pickle_cv_model(cv, model, filename):
    with open(filename, 'wb') as file:
        pickle.dump((cv, model), file)


def unpickle_cv_model(filename):
    with open(filename, 'rb') as file:
        cv, model = pickle.load(file)
        return cv, model
