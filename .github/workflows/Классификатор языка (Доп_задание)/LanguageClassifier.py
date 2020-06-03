import numpy as np
import pandas as pd
import Tokenize



def get_ngram_frequency(ngrams):
    series = pd.Series()
    for ngram in ngrams:
        if ngram not in series:
            series[ngram] = 1
        else:
            series[ngram] += 1
    return series.sort_values(ascending=False)


def get_distance(pretrained_series, current_series):
    distance = 0
    for word in current_series.index:
        if word not in pretrained_series.index:
            distance += abs(current_series.size - current_series[word])
        else:
            distance += abs(pretrained_series[word] - current_series[word])
    return distance


def train_models(languages, pretrainded_head=5000):
    pretrainded_models = {}
    for language in languages:
        corpus = ''
        for i in range(1, 4):
            with open('Texts/{}/{}{}.txt'.format(language, language, i), encoding='utf-8') as f:
                corpus += f.read().rstrip()
        ngrams = Tokenize.get_ngrams(corpus, ngram_size=3)
        pretrained = get_ngram_frequency(ngrams).head(pretrainded_head)
        pretrained.to_csv('{}_{}.csv'.format(language,pretrainded_head), header=False)
        pretrainded_models[language] = pretrained
    return pretrainded_models


def read_models(languages, trained_set):
    pretrainded_models = {}
    for language in languages:
        pretrained = pd.read_csv('{}_{}.csv'.format(language, trained_set), index_col=0, header=None).iloc[:, 0]
        pretrainded_models[language] = pretrained
    return pretrainded_models


def get_score(pretrainded_models, current_model, n=3):
    result = {}
    total_distance = 0
    for language, model in pretrainded_models.items():
        distance = get_distance(model, current_model)
        result[language] = distance
        total_distance += distance
    for item in result.keys():
        result[item] = abs(result[item] - int(total_distance / n))
    return result


languages = ['blr', 'rus', 'ukr']
pretrainded_models = train_models(languages)
pretrainded_models = read_models(languages, 5000)
df = pd.DataFrame(np.zeros([len(languages), len(languages)]), index=languages, columns=languages)
labels = pd.read_csv('Texts/labels.csv').set_index('name')
print('Count text dataframe: \n{}'.format(labels.loc[:, 'language'].value_counts()))
for i in range(1, 11):
    with open('Texts/text_{}.txt'.format(i), encoding='utf-8') as f:
        text = f.read().rstrip()
    current_model = Tokenize.get_ngrams(text)
    current_model_frequency = get_ngram_frequency(current_model)
    score = get_score(pretrainded_models, current_model_frequency)
    predicted_language = [k for k in score if score[k] == max(score.values())][0]
    actual_language = labels.loc['text_{}.txt'.format(i), 'language']
    print('Predicted: text{} in {} language'.format(i, predicted_language))
    print('Predicted score: {}'.format(score))
    print('Actual: text{} in {} language'.format(i, actual_language))
    if predicted_language is actual_language:
        df.loc[predicted_language][predicted_language] += 1
    else:
        df.loc[predicted_language][actual_language] += 1
print('Resulted dataframe: \n{}'.format(df))
with np.errstate(divide='ignore'):
    for language in languages:
        recall = df.loc[language][language] / sum(df[language])
        precision = df.loc[language][language] / sum(df.loc[language])
        print('Recall {}-class: {}'.format(language, recall))
        print('Precision {}-class: {}'.format(language, precision))
    true_positive = sum(np.diag(df))
    total_without_true_positive = df.mask(np.eye(3, dtype=bool)).fillna(0.0).values.sum()
    accuracy = true_positive / (total_without_true_positive + true_positive)
    print('Accuracy: {}'.format(accuracy))
