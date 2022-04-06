import sys

import gensim
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

MAX_SENT_LEN = 200
MAX_VOCAB_SIZE = 25000
LSTM_DIM = 100
EMBEDDING_DIM = 50
BATCH_SIZE = 200
N_EPOCHS = 10


def run(path):
    train_bodies = pd.read_csv(path + "/train_bodies.csv")
    train_stances = pd.read_csv(path + "/train_stances.csv")
    comp_bodies = pd.read_csv(path + "/competition_test_bodies.csv")
    comp_stances = pd.read_csv(path + "/competition_test_stances.csv")
    
    x_train = pd.merge(train_stances, train_bodies, how='left', left_on=['Body ID'], right_on=['Body ID'])
    x_val = pd.merge(comp_stances, comp_bodies, how='left', left_on=['Body ID'], right_on=['Body ID'])

    y_train = x_train['Stance'].replace({'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3})
    y_val = x_val['Stance'].replace({'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3})

    x_train['full'] = x_train['Headline'].astype(str) + x_train['articleBody'].astype(str)
    x_val['full'] = x_val['Headline'].astype(str) + x_val['articleBody'].astype(str)

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(x_val['full'].astype(str).append(x_train['full']))

    train_seq = tokenizer.texts_to_sequences(x_train['full'])
    test_seq = tokenizer.texts_to_sequences(x_val['full'])

    x_train = pad_sequences(train_seq, maxlen=MAX_SENT_LEN, padding='post', truncating='post')
    x_val = pad_sequences(test_seq, maxlen=MAX_SENT_LEN, padding='post', truncating='post')

    y_train = tf.keras.utils.to_categorical(np.asarray(y_train).astype('float32'))
    y_val = tf.keras.utils.to_categorical(np.asarray(y_val).astype('float32'))

    embeddings_matrix = get_embeddings_matrix(path, tokenizer)

    base_model = baseLSTM(x_train, y_train, x_val, y_val, embeddings_matrix, tokenizer)
    score, acc = base_model.evaluate(x_val, y_val, batch_size=BATCH_SIZE)
    print("Base bidirectional LSTM: Accuracy = {0:4.3f}".format(
        acc) + " Score = {0:4.3f}".format(score))
    base_model.save(path + '/lstm_base3')


def baseLSTM(x_train, y_train, x_val, y_val, embeddings_matrix, tokenizer):
    # Build a sequential model by stacking neural net units
    model = Sequential()
    # input layer word2vec
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                        output_dim=EMBEDDING_DIM,
                        weights=[embeddings_matrix], trainable=False,mask_zero=True, name='word_embedding_layer'))
    model.add(Bidirectional(LSTM(LSTM_DIM, return_sequences=False, name='bidirectional_lstm')))
    # Add activation layer
    model.add(Activation(activation='relu', name='activation_1'))
    model.add(Dropout(rate=0.2, name='dropout_1'))

    model.add(Dropout(rate=0.2, name='dropout_2'))
    model.add(Activation(activation='relu', name='activation_2'))

    model.add(Dense(4, activation='softmax', name='output_layer'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=N_EPOCHS,
              validation_data=(x_val, y_val))
    return model


def get_embeddings_matrix(path, tokenizer):
    glove2word2vec('glove.twitter.27B.50d.txt', 'glove.word2vec')
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('glove.word2vec', binary=False)

    vocab_size = len(tokenizer.word_index) + 1

    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(vocab_size, EMBEDDING_DIM))

    for word, i in tokenizer.word_index.items():
        try:
            embeddings_vector = w2v_model[word]
        except KeyError:
            embeddings_vector = None
    if embeddings_vector is not None:
        embeddings_matrix[i] = embeddings_vector
    return embeddings_matrix


if __name__ == '__main__':
    # Arguments passed
    path = 'data'
    if len(sys.argv) > 1:
        print("Path to folder with data, test, train, validation: ", sys.argv[1])
        path = sys.argv[1]
    run(path)
