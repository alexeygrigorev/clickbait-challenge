
import re
import json
import pickle
import os
import sys

import itertools

import numpy as np
import pandas as pd

from nltk.stem.porter import PorterStemmer

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

texts = ['targetTitle', 'targetDescription']
multi = ['postText', 'targetCaptions', 'targetParagraphs']

def read_chunk(lines):
    instances = []

    for line in lines:
        line = json.loads(line)
        del line['postTimestamp']

        for m in texts:
            line[m] = clean_text(line[m])

        for m in multi:
            line[m] = dedup_text(line[m])
            line[m] = clean_texts(line[m])
            line['%s_concat' % m] = ' '.join(line[m])

        instances.append(line)

    return pd.DataFrame(instances)


with open('models.bin', 'rb') as f:
    print('reading the models...')
    models = pickle.load(f)

    cv = models['cv']
    cv_kw = models['cv_kw']
    svm_kw_mean = models['svm_kw_mean']
    svm_kw_std = models['svm_kw_std']
    svm_all_mean = models['svm_all_mean']
    svm_all_std = models['svm_all_std']
    svm_post_mean = models['svm_post_mean']
    svm_post_std = models['svm_post_std']
    svm_tc_mean = models['svm_tc_mean']
    svm_tc_std = models['svm_tc_std']
    et = models['et']

    print('done')

def process_keywords(text):
    text = text.lower()
    text = non_letter.sub(' ', text)
    text = whitespace.sub(' ', text).strip()
    return text

def predict(df_data):
    targetKeywords = df_data.targetKeywords.apply(process_keywords)
    all_text = df_data.postText_concat + ' ' + \
               df_data.targetCaptions_concat + ' ' + \
               df_data.targetDescription + ' ' + \
               df_data.targetParagraphs_concat + ' ' + \
               df_data.targetTitle

    X_kw = cv_kw.transform(targetKeywords)
    y_kw_pred_mean = svm_kw_mean.predict(X_kw)
    y_kw_pred_std = svm_kw_std.predict(X_kw)

    X_full = cv.transform(all_text)
    y_all_pred_mean = svm_all_mean.predict(X_full)
    y_all_pred_std = svm_all_std.predict(X_full)

    X = cv.transform(df_data.postText_concat)
    y_post_pred_mean = svm_post_mean.predict(X)
    y_post_pred_std = svm_post_std.predict(X)

    X = cv.transform(df_data.targetTitle)
    y_tt_pred_mean = svm_tc_mean.predict(X)
    y_tt_pred_std = svm_tc_std.predict(X)

    df_features = pd.DataFrame()
    df_features['all_pred_mean'] = y_all_pred_mean
    df_features['all_pred_std'] = y_all_pred_std
    df_features['post_pred_mean'] = y_post_pred_mean
    df_features['post_pred_std'] = y_post_pred_std
    df_features['tt_pred_mean'] = y_tt_pred_mean
    df_features['tt_pred_std'] = y_tt_pred_std
    df_features['kw_pred_mean'] = y_kw_pred_mean
    df_features['kw_pred_std'] = y_kw_pred_std

    X = df_features.values
    pred = et.predict(X)
    pred[pred < 0] = 0
    pred[pred > 1] = 1

    df_res = pd.DataFrame()
    df_res['id'] = df_data.id
    df_res['clickbaitScore'] = pred

    return df_res.to_dict(orient='records')

def iter_slice(it, n):
    while True:
        s = itertools.islice(it, n)
        s = list(s)
        if len(s) == 0:
            return
        yield s


args = sys.argv
input_folder = args[1] #'data/clickbait17-validation-170630/'
output_folder = args[2] #'output/'

try:
    os.makedirs(output_folder)
except:
    pass

input_f = open(os.path.join(input_folder, 'instances.jsonl'))
output_f = open(os.path.join(output_folder, 'results.jsonl'), 'w')


for i, chunk in enumerate(iter_slice(input_f, 100)):
    print('processing chunk #%d... ' % i, end='')
    df = read_chunk(chunk)
    preds = predict(df)
    for pred in preds:
        pred = json.dumps(pred)
        output_f.write(pred)
        output_f.write('\n')
    output_f.flush()
    print('done')

input_f.close()
output_f.close()