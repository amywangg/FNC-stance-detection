import nltk
import ssl
import sys
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')


def preprocess(path):
    if path == '':
        path = 'data'

    train_bodies = pd.read_csv(path + "/train_bodies.csv")
    train_stances = pd.read_csv(path + "/train_stances.csv")
    comp_bodies = pd.read_csv(path + "/competition_test_bodies.csv")
    comp_stances = pd.read_csv(path + "/competition_test_stances.csv")
    test_bodies = pd.read_csv(path + "/competition_test_bodies.csv")
    test_stances_unlabeled = pd.read_csv(path + "/test_stances_unlabeled.csv")

    train = pd.merge(train_stances, train_bodies, how='left', left_on=['Body ID'], right_on=['Body ID'])
    comp = pd.merge(comp_stances, comp_bodies, how='left', left_on=['Body ID'], right_on=['Body ID'])
    test = pd.merge(test_stances_unlabeled, test_bodies, how='left', left_on=['Body ID'], right_on=['Body ID'])

    tokenize(train)
    remove_stopwords(train)
    stem_lem(train)

    tokenize(comp)
    remove_stopwords(comp)
    stem_lem(comp)

    tokenize(test)
    remove_stopwords(test)
    stem_lem(test)

    train['Headline_full'] = train['Headline'].apply(lambda x: ' '.join(map(str, x)))
    train['articleBody_full'] = train['articleBody'].apply(lambda x: ' '.join(map(str, x)))

    comp['Headline_full'] = comp['Headline'].apply(lambda x: ' '.join(map(str, x)))
    comp['articleBody_full'] = comp['articleBody'].apply(lambda x: ' '.join(map(str, x)))

    test['Headline_full'] = test['Headline'].apply(lambda x: ' '.join(map(str, x)))
    test['articleBody_full'] = test['articleBody'].apply(lambda x: ' '.join(map(str, x)))

    train.to_csv(path + '/train_cleaned.csv')
    comp.to_csv(path + '/comp_cleaned.csv')
    test.to_csv(path + '/test_cleaned.csv')


# tokenizer
def tokenize(df):
    for column in df:
        if column != 'Stance' and column != "Body ID":
            df[column] = df[column].apply(word_tokenize)


# stopword removal
def remove_stopwords(df):
    stop = stopwords.words('english')
    for column in df:
        if column != 'Stance' and column != "Body ID":
            df[column] = df[column].apply(lambda x: [item for item in x if item not in stop])
            df[column] = df[column].apply(lambda x: [word for word in x if word.isalnum()])


# stemming and lemmatization
def stem_lem(df):
    for column in df:
        if column != 'Stance' and column != "Body ID":
            df[column] = df[column].apply(stem)
            df[column] = df[column].apply(lemmatize)


def stem(text):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(w) for w in text]


def lemmatize(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]


if __name__ == '__main__':
    # Arguments passed
    path = 'data'
    if len(sys.argv) > 1:
        print("Path to folder with test, train, competition data: ", sys.argv[1])
        path = sys.argv[1]
    preprocess(path)
