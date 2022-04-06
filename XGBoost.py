import sys
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import Util

# XGBoost Params
LEARNING_RATE = 0.4
MIN_CHILD_WEIGHT = 4
MAX_DEPTH = 3
GAMMA = 0
N_ESTIMATORS = 100


def run(path, ngram, is_inference):
    x_train = pd.read_csv(path + '/features/train_features.csv')
    x_val = pd.read_csv(path + '/features/comp_features.csv')

    if ngram != '':
        if is_inference:
            train_tfidf_head = Util.unpickle_model(path + '/features/test_headline_tfidf_' + ngram + '.pkl')[:49972]
            train_tfidf_body = Util.unpickle_model(path + '/features/test_body_tfidf_' + ngram + '.pkl')[:49972]
        else:
            train_tfidf_head = Util.unpickle_model(path + '/features/val_headline_tfidf_' + ngram + '.pkl')[:49972]
            train_tfidf_body = Util.unpickle_model(path + '/features/val_body_tfidf_' + ngram + '.pkl')[:49972]

        val_tfidf_head = Util.unpickle_model(path + '/features/val_headline_tfidf_' + ngram + '.pkl')[49972:]
        val_tfidf_body = Util.unpickle_model(path + '/features/val_body_tfidf_' + ngram + '.pkl')[49972:]

        csr_train = csr_matrix(x_train)
        csr_val = csr_matrix(x_val)

        # Use these features for model with tfidf vectorizer
        x_train = hstack([train_tfidf_head, train_tfidf_body, csr_train])
        x_val = hstack([val_tfidf_head, val_tfidf_body, csr_val])

    y_train = pd.read_csv(path + '/train_cleaned.csv')['Stance'].replace(
        {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3})
    y_val = pd.read_csv(path + '/comp_cleaned.csv')['Stance'].replace(
        {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3})

    model = XGBClassifier(random_state=100, eta=LEARNING_RATE, min_child_weight=MIN_CHILD_WEIGHT, max_depth=MAX_DEPTH,
                          gamma=GAMMA, n_estimators=N_ESTIMATORS)
    model.fit(x_train, y_train)

    # pickle the model
    if ngram != '' and not inf:
        Util.pickle_model(model, path + '/models/xgb_tfidf_' + ngram + '.pkl')
    elif not inf:
        Util.pickle_model(model, path + '/models/xgb_base.pkl')

    if is_inference:
        x_test = pd.read_csv(path + '/features/test_features.csv')
        if ngram != '':
            test_tfidf_head = Util.unpickle_model(path + '/features/val_headline_tfidf_' + ngram + '.pkl')[49972:]
            test_tfidf_body = Util.unpickle_model(path + '/features/val_body_tfidf_' + ngram + '.pkl')[49972:]
            csr_test = csr_matrix(x_test)
            # Use these features for model with tfidf vectorizer
            x_test = hstack([test_tfidf_head, test_tfidf_body, csr_test])

        pred_xg = model.predict(x_test)
        test_stances_unlabeled = pd.read_csv(path + "/test_stances_unlabeled.csv")
        Util.predictions_to_csv(pred_xg, test_stances_unlabeled['Headline'], test_stances_unlabeled['Body ID'],
                                path + '/answer.csv')
    else:
        # for model tuning will pring out accuracy
        pred_xg = model.predict(x_val)
        print(model.score(x_val, y_val))
        print(accuracy_score(y_val, pred_xg))


if __name__ == '__main__':
    # Arguments passed
    ngram = ''
    path = 'data'
    inf = False
    if len(sys.argv) > 1:
        print("Path to folder with data, test, train, validation: ", sys.argv[1])
        path = sys.argv[1]
        if len(sys.argv) > 2:
            ngram = sys.argv[2]
            if len(sys.argv) > 3 and sys.argv[3] == 'inf':
                inf = True
    run(path, ngram, inf)
