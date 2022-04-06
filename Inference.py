import sys
import Util
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

MAX_SENT_LEN = 300 # 150 for base, 300 for base2
MAX_VOCAB_SIZE = 40000


def run(path, model_type, version):
    if model_type == 'xgb':
        if version == 'base':
            model = Util.unpickle_model(path + '/models/' + model_type + '_' + version + '.pkl')
        else:
            model = Util.unpickle_model(path + '/models/' + model_type + '_tfidf_' + version + '.pkl')

        x_test = pd.read_csv(path + '/features/test_features.csv')
        if version != 'base':
            test_tfidf_head = Util.unpickle_model(path + '/features/val_headline_tfidf_' + version + '.pkl')[49972:]
            test_tfidf_body = Util.unpickle_model(path + '/features/val_body_tfidf_' + version + '.pkl')[49972:]
            csr_test = csr_matrix(x_test)
            # Use these features for model with tfidf vectorizer
            x_test = hstack([test_tfidf_head, test_tfidf_body, csr_test])

        predictions = model.predict(x_test)
        test_stances_unlabeled = pd.read_csv(path + "/test_stances_unlabeled.csv")
        Util.predictions_to_csv(predictions, test_stances_unlabeled['Headline'], test_stances_unlabeled['Body ID'],
                        path + '/answers/' + model_type + '_' + version + '_answer.csv')

    elif model_type == 'lstm':
        x_test = pd.read_csv(path + "/test_cleaned.csv")
        x_test['full'] = x_test['Headline_full'].astype(str) + x_test['articleBody_full'].astype(str)

        tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(x_test['full'].astype(str))

        test_seq = tokenizer.texts_to_sequences(x_test['full'])

        x_test = pad_sequences(test_seq, maxlen=MAX_SENT_LEN, padding='post', truncating='post')

        model = keras.models.load_model(path + '/models/' + model_type + '_' + version )
        predictions = model.predict(x_test)

        test_stances_unlabeled = pd.read_csv(path + "/test_stances_unlabeled.csv")
        Util.predictions_to_csv2(predictions, test_stances_unlabeled['Headline'], test_stances_unlabeled['Body ID'],
                                path + '/answers/' + model_type + '_' + version + '_answer.csv')


if __name__ == '__main__':
    path = 'data'
    version = 'base'
    if len(sys.argv) > 2:
        print("Path to folder with data, test, train, validation: ", sys.argv[1])
        path = sys.argv[1]
        model_type = sys.argv[2]
        version = sys.argv[3]
    run(path, model_type, version)
