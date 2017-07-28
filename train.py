# coding: utf-8
import re

import json
import pickle

import numpy as np
import pandas as pd

from nltk.stem.porter import PorterStemmer

from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR

from tqdm import tqdm_notebook as tqdm


stemmer = PorterStemmer()

stop = {'the', 'was', 'were', 'did', 'had', 'have', 'been', 'will', 'and', 
        'that', 'who', 'are', 'for', 'has'}

tags = re.compile(r'<.+?>')
non_letter = re.compile(r'\W+')
number_pattern = re.compile(r'\d+')
whitespace = re.compile(r'\s+')

def clean_text(text):
    text = text.lower()
    text = text.replace('–', '-')
    text = text.replace("'", ' ')
    text = tags.sub(' ', text)
    text = non_letter.sub(' ', text)

    tokens = []

    for t in text.split():
        if len(t) <= 2 and not t.isdigit():
            continue

        if t in stop:
            continue
        t = stemmer.stem(t)
        tokens.append(t)

    text = ' '.join(tokens)

    text = number_pattern.sub(' [N]', text)
    text = whitespace.sub(' ', text)
    text = text.strip()
    return text

def dedup_text(texts):
    return sorted(set(texts))

def clean_texts(texts):
    return [clean_text(t) for t in texts]


def read_data(folder):
    instances = []
    labels = {}

    with open(folder + '/truth.jsonl', 'r') as f:
        for line in f:
            line = json.loads(line)
            id = line['id']
            mean = line['truthMean']
            std = np.std(line['truthJudgments'])
            labels[id] = (mean, std)

    texts = ['targetTitle', 'targetDescription']
    multi = ['postText', 'targetCaptions', 'targetParagraphs']

    with open(folder + '/instances.jsonl', 'r') as f:
        for line in tqdm(f):
            line = json.loads(line)
            mean, std = labels[line['id']]

            del line['postTimestamp']
            line['truthMean'] = mean
            line['truthStd'] = std

            for m in texts:
                line[m] = clean_text(line[m])

            for m in multi:
                line[m] = dedup_text(line[m])
                line[m] = clean_texts(line[m])
                line['%s_concat' % m] = ' '.join(line[m])

            instances.append(line)

    df_data = pd.DataFrame(instances)
    return df_data



df_data1 = read_data('data/clickbait17-train-170331')
df_data2 = read_data('data/clickbait17-validation-170630')
df_data = pd.concat([df_data1, df_data2]).reset_index(drop=1)


cv_idx = KFold(n=len(df_data), n_folds=5, shuffle=True, random_state=1)


def process_keywords(text):
    text = text.lower()
    text = non_letter.sub(' ', text)
    text = whitespace.sub(' ', text).strip()
    return text


df_data.targetKeywords = df_data.targetKeywords.apply(process_keywords)

y_full_mean = df_data.truthMean.values
y_full_std = df_data.truthStd.values

cv_kw = CountVectorizer(token_pattern='\\S+', ngram_range=(1, 3), min_df=10, 
                     binary=True, dtype=np.uint8)
X_kw = cv_kw.fit_transform(df_data.targetKeywords)


def try_Cs(X, y, cv, Cs):
    results = []

    for C in Cs:
        t0 = time()
        scores = []

        for train_idx, val_idx in cv:
            svm = LinearSVR(C=C, loss='squared_epsilon_insensitive', dual=False, random_state=1)
            svm.fit(X[train_idx], y[train_idx])

            y_pred = svm.predict(X[val_idx])
            y_pred[y_pred < 0] = 0.0
            y_pred[y_pred > 1] = 1.0

            rmse = mean_squared_error(y[val_idx], y_pred)
            scores.append(rmse)

        m = np.mean(scores)
        s = np.std(scores)

        print('C=%s, took %.3fs, mse=%.3f+-%.3f' % (C, time() - t0, m, s))
        
        results.append((m.round(3), s, C))
    
    _, _, best_C = min(results)
    return best_C

def fit_final(X, y, cv, C):
    y_pred_train = np.zeros(len(y))

    for train_idx, val_idx in cv:
        svm = LinearSVR(C=C, loss='squared_epsilon_insensitive', dual=False, random_state=1)
        svm.fit(X[train_idx], y[train_idx])

        y_pred_train[val_idx] = svm.predict(X[val_idx])

    svm = LinearSVR(C=C, loss='squared_epsilon_insensitive', dual=False, random_state=1)
    svm.fit(X, y)

    return y_pred_train, svm



Cs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 1]

best_C = try_Cs(X_kw, y_full_mean, cv_idx, Cs=Cs)
print('best C is %s' % best_C)
y_kw_pred_mean, svm_kw_mean = fit_final(X_kw, y_full_mean, cv_idx, best_C)

best_C = try_Cs(X_kw, y_full_std, cv_idx, Cs=Cs)
print('best C is %s' % best_C)
y_kw_pred_std, svm_kw_std = fit_final(X_kw, y_full_std, cv_idx, best_C)



df_data['all_text'] = df_data.postText_concat + ' ' + \
                      df_data.targetCaptions_concat + ' ' + \
                      df_data.targetDescription + ' ' + \
                      df_data.targetParagraphs_concat + ' ' + \
                      df_data.targetTitle

cv = CountVectorizer(token_pattern='\\S+', ngram_range=(1, 3), min_df=10, 
                     binary=True, dtype=np.uint8)
X_full = cv.fit_transform(df_data.all_text)


Cs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]

best_C = try_Cs(X_full, y_full_mean, cv_idx, Cs=Cs)
print('best C is %s' % best_C)
y_all_pred_mean, svm_all_mean = fit_final(X_full, y_full_mean, cv_idx, best_C)

best_C = try_Cs(X_full, y_full_std, cv_idx, Cs=Cs)
print('best C is %s' % best_C)
y_all_pred_std, svm_all_std = fit_final(X_full, y_full_std, cv_idx, best_C)



X = cv.transform(df_data.postText_concat)

Cs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 1]

best_C = try_Cs(X, y_full_mean, cv_idx, Cs=Cs)
print('best C is %s' % best_C)
y_post_pred_mean, svm_post_mean = fit_final(X, y_full_mean, cv_idx, best_C)

best_C = try_Cs(X, y_full_std, cv_idx, Cs=Cs)
print('best C is %s' % best_C)
y_post_pred_std, svm_post_std = fit_final(X, y_full_std, cv_idx, best_C)



X = cv.transform(df_data.targetTitle)

Cs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 1]

best_C = try_Cs(X, y_full_mean, cv_idx, Cs=Cs)
print('best C is %s' % best_C)
y_tt_pred_mean, svm_tc_mean = fit_final(X, y_full_mean, cv_idx, best_C)

best_C = try_Cs(X, y_full_std, cv_idx, Cs=Cs)
print('best C is %s' % best_C)
y_tt_pred_std, svm_tc_std = fit_final(X, y_full_std, cv_idx, best_C)




df_second = pd.DataFrame()

df_second['all_pred_mean'] = y_all_pred_mean
df_second['all_pred_std'] = y_all_pred_std
df_second['post_pred_mean'] = y_post_pred_mean
df_second['post_pred_std'] = y_post_pred_std
df_second['tt_pred_mean'] = y_tt_pred_mean
df_second['tt_pred_std'] = y_tt_pred_std
df_second['kw_pred_mean'] = y_kw_pred_mean
df_second['kw_pred_std'] = y_kw_pred_std

X_second = df_second.values



from sklearn.ensemble import ExtraTreesRegressor

et_params = dict(
    n_estimators=50,
    criterion='mse',
    max_depth=15,
    min_samples_split=6,
    min_samples_leaf=6,
    max_features=4,
    bootstrap=False, 
    n_jobs=-1,
    random_state=1
)

et = ExtraTreesRegressor(**et_params)
et.fit(X_second, y_full_mean)


models = {}
models['cv'] = cv
models['cv_kw'] = cv_kw
models['svm_kw_mean'] = svm_kw_mean
models['svm_kw_std'] = svm_kw_std
models['svm_all_mean'] = svm_all_mean
models['svm_all_std'] = svm_all_std
models['svm_post_mean'] = svm_post_mean
models['svm_post_std'] = svm_post_std
models['svm_tc_mean'] = svm_tc_mean
models['svm_tc_std'] = svm_tc_std
models['et'] = et

with open('models.bin', 'wb') as f:
    pickle.dump(models, f)
