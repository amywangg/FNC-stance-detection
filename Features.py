import math
import re
import sys
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import Util

# Params for TFIDF
MAX_FEATURES = 40000


def run(path, ngram):
    train = getFeatures(path, 'train', ngram)
    test = getFeatures(path, 'test', ngram)
    val = getFeatures(path, 'comp', ngram)
    if ngram != '':
        getTfidf(path, train, test, val, ngram)


def getFeatures(path, dataset, ngram):
    # read in the preprocessed data and join the tokens
    cleaned = pd.read_csv(path + '/' + dataset + '_cleaned.csv')

    # run the functions for feature collection and store in dataframe
    if ngram == '':
        features = pd.DataFrame()
        features['cosine'] = cosine(cleaned['Headline_full'].astype(str), cleaned['articleBody_full'].astype(str))
        features['euclidean'] = euclidean(cleaned['Headline_full'].astype(str), cleaned['articleBody_full'].astype(str))
        features['jaccard'] = jaccard(cleaned['Headline'], cleaned['articleBody'])

        # store the updated data and features
        features.to_csv(path + '/features/' + dataset + '_features.csv', index=False)

    return cleaned


def text_to_vector(text):
    WORD = re.compile(r"\w+")
    words = WORD.findall(text)
    return Counter(words)


def cosine(headline, body):
    cos = []
    for i in range(len(headline)):
        vector1 = text_to_vector(headline[i])
        vector2 = text_to_vector(body[i])
        intersection = set(vector1.keys()) & set(vector2.keys())
        numerator = sum([vector1[x] * vector2[x] for x in intersection])

        sum1 = sum([vector1[x] ** 2 for x in list(vector1.keys())])
        sum2 = sum([vector2[x] ** 2 for x in list(vector2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            cos.append(0.0)
        else:
            cos.append(float(numerator) / denominator)
    return cos


def euclidean(headline, body):
    euc = []
    for i in range(len(headline)):
        vector1 = text_to_vector(headline[i])
        vector2 = text_to_vector(body[i])
        euc.append(math.sqrt(
            sum((vector1.get(k, 0) - vector2.get(k, 0)) ** 2 for k in set(vector1.keys()).union(set(vector2.keys())))))
    return euc


# Jaccard similarity
def jaccard(heading, body):
    jacc = []
    for i in range(len(heading)):
        intersection = set(heading[i]).intersection(set(body[i]))
        union = set(heading[i]).union(set(body[i]))
        jacc.append(len(intersection) / len(union))
    return jacc


def getTfidf(path, train, test, val, ngram):
    ngram_class = {
        'uni': (1, 1),
        'unibi': (1, 2),
        'bi': (2, 2)
    }
    cv = CountVectorizer(ngram_range=ngram_class[ngram], tokenizer=lambda doc: doc, max_features=MAX_FEATURES, lowercase=True)

    vectors = cv.fit_transform(train['Headline_full'].astype(str).append(val['Headline_full']).astype(str))
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(vectors)
    Util.pickle_model(tfidf, path + "/features/val_headline_tfidf_" + ngram + ".pkl")

    vectors = cv.fit_transform(train['articleBody_full'].astype(str).append(val['articleBody_full']).astype(str))
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(vectors)
    Util.pickle_model(tfidf, path + "/features/val_body_tfidf_" + ngram + ".pkl")

    vectors = cv.fit_transform(
        train['Headline_full'].astype(str).append(test['Headline_full']).astype(str))
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(vectors)
    Util.pickle_model(tfidf, path + "/features/test_headline_tfidf_" + ngram + ".pkl")

    vectors = cv.fit_transform(
        train['articleBody_full'].astype(str).append(test['articleBody_full']).astype(str))
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(vectors)
    Util.pickle_model(tfidf, path + "/features/test_body_tfidf_" + ngram + ".pkl")


if __name__ == '__main__':
    # Arguments passed
    path = 'data'
    ngram = ''
    if len(sys.argv) > 1:
        print("Path to folder with data, test, train, validation: ", sys.argv[1])
        path = sys.argv[1]
        if len(sys.argv) > 2:
            ngram = sys.argv[2]
    run(path, ngram)
