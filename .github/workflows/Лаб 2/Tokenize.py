
import Algorithm
import nltk
#import string
#from nltk.corpus import stopwords
import numpy as np
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

import re

nltk.download('stopwords')
nltk.download('punkt')

def tag_map(_tag):
    if _tag is 'J':
        return wn.ADJ
    elif _tag is 'V':
        return wn.VERB
    elif _tag is 'R':
        return wn.ADV
    else:
        return wn.NOUN

def tokenize_word(file_text):
    # words = nltk.word_tokenize(file_text)
    tokens = re.split(r'[\.;:—,\?\-!(...)(\s\d+)]*\s', file_text.lower())

    # удаление пунктуации
    tokens = [re.sub(r'[-(\.)+;:!\?"\(\)«»]+','', i) for i in tokens ]
    tokens = [value for value in tokens if value]
 #   tokens = [i for i in tokens if i not in string.punctuation]

    # удаление стоп-слов
    # stop_words = stopwords.words('russian')
    # stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на', 'т.к', 'т.п', 'др'])
    # print(stop_words)
    # tokens = [i for i in tokens if i not in stop_words]

    # чистка от кавычек
   # tokens = [i.replace("«", "").replace("»", "") for i in tokens]
   # tokens = [i.replace("(", "").replace(")", "") for i in tokens]
    #words = [i for i in set(tokens)]
    return tokens

def tokenize_sentence(file_text):
    #sentences = nltk.sent_tokenize(file_text)
    sentences = re.split(r'[.\?!(...)]\s', file_text)

    return sentences

text1 = 'CHEESE CHORES GEESE GLOVES'
text2 = 'Бараны бодаются, бьют барабаны, истек источник хранения хруста, ищем деятелей искусства для пробужедния чувства, что же мы хотим ответить на зов китов, мурчания котов, просто помычим'

words = tokenize_word(text2)

def d(coord, words):
    i, j = coord
    return Algorithm.Levinstein(words[i], words[j])

indeces = np.triu_indices(len(words), 1)
weights = np.apply_along_axis(d, 0, indeces, words)
#
# print(tokenize_word(text2))
# print(tokenize_sentence(text2))
# print(indeces)
# print(weights)
#
# Z = linkage(weights, 'ward')
# dn = dendrogram((Z),labels=np.array(words), orientation="right", leaf_font_size=7)
# fig = plt.figure(figsize=(100, 50))
#
# plt.show()