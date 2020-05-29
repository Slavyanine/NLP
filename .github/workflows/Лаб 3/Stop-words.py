import Tokenize
import pymorphy2
import math
import copy
import pandas as pd
from nltk.corpus import stopwords

morph = pymorphy2.MorphAnalyzer()

word_set_1 = []
word_dict_tf_1 = {}
idf_1 = {}
tf_idf_1 = {}

word_set_2 = []
word_dict_tf_2 = {}
idf_2 = {}
tf_idf_2 = {}

stop_words = set()
specific_words_1 = set()
specific_words_2 = set()
stop_words_default = stopwords.words('russian')
n = 10

#обработка первой выборки
for i in range(1, n+1):
    sum_w = 0
    print("текст 1_{id}.txt".format(id=i))
    with open("текст 1_{id}.txt".format(id=i), encoding='utf-8') as f:
        for line in f:
            tokens_from_line = Tokenize.tokenize_word(line.strip())
            sum_w += len(tokens_from_line)
            words = [morph.parse(word)[0].normal_form for word in tokens_from_line]
            word_set_1 = set(word_set_1).union(set(words))
            for word in word_set_1:
                if word not in word_dict_tf_1:
                    word_dict_tf_1[word] = [0*i for i in range(n)]
            for word in words:
                word_dict_tf_1[word][i-1] += 1
        for word in word_set_1:
            word_dict_tf_1[word][i - 1] /= sum_w



idf_1 = dict.fromkeys(list(word_dict_tf_1.keys()), 0)
for i in range(n):
    for word in word_set_1:
        if word_dict_tf_1[word][i]>0:
            idf_1[word] += 1
for word, v in idf_1.items():
    idf_1[word] = math.log(n / float(v))

tf_idf_1 = copy.deepcopy(word_dict_tf_1)
for key in tf_idf_1:
    for i in range(n):
        tf_idf_1[key][i] *= idf_1[key]


#обработка второй выборки
for i in range(1, n+1):
    sum_w = 0
    print("текст 2_{id}.txt".format(id=i))
    with open("текст 2_{id}.txt".format(id=i), encoding='utf-8') as f:
        for line in f:
            tokens_from_line = Tokenize.tokenize_word(line.strip())
            sum_w += len(tokens_from_line)
            words = [morph.parse(word)[0].normal_form for word in tokens_from_line]
            word_set_2 = set(word_set_2).union(set(words))
            for word in word_set_2:
                if word not in word_dict_tf_2:
                    word_dict_tf_2[word] = [0*i for i in range(n)]
            for word in words:
                word_dict_tf_2[word][i-1] += 1
        for word in word_set_2:
            word_dict_tf_2[word][i - 1] /= sum_w



idf_2 = dict.fromkeys(list(word_dict_tf_2.keys()), 0)
for i in range(n):
    for word in word_set_2:
        if word_dict_tf_2[word][i]>0:
            idf_2[word] += 1
for word, v in idf_2.items():
    idf_2[word] = math.log(n / float(v))

tf_idf_2 = copy.deepcopy(word_dict_tf_2)
for key in tf_idf_2:
    for i in range(n):
        tf_idf_2[key][i] *= idf_2[key]

print('Первая выборка:')
print('TF_1: ', word_dict_tf_1)
print('IDF_1: ', idf_1)
print('TF_IDF_1: ', tf_idf_1)
data1 = pd.DataFrame(list(tf_idf_1.items()), columns=['word', 'tf*idf'])
print(data1)

print()
print('Вторая выборка:')
print('TF_2: ', word_dict_tf_2)
print('IDF_2: ', idf_2)
print('TF_IDF_2: ', tf_idf_2)
data2 = pd.DataFrame(list(tf_idf_2.items()), columns=['word', 'tf*idf'])
print(data2)

eps = 0.0001
for key in tf_idf_1:
    if sum(tf_idf_1[key])< eps:
        stop_words.add(key)

for key in tf_idf_2:
    if sum(tf_idf_2[key])< eps:
        stop_words.add(key)

print("отобранные стоп-слова: ", stop_words)
print("стоп-слова(для сравнения): ", stop_words_default)

eps2 = 0.02
for key in tf_idf_1:
    if sum(tf_idf_1[key])> eps2:
        specific_words_1.add(key)

specific_words_1.difference_update(set(list(tf_idf_2.keys())))

for key in tf_idf_2:
    if sum(tf_idf_2[key])> 0.011:
        specific_words_2.add(key)

specific_words_2.difference_update(set(list(tf_idf_1.keys())))

print("специфичные слова выборки №1:", specific_words_1)
print("специфичные слова выборки №2:", specific_words_2)
