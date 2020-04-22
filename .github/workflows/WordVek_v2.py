import Tokenize #подключение ранее реальзованого токенизатора
import pymorphy2
import math
import copy
import pandas as pd
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import numpy as np
import pandas
from nltk.corpus import stopwords
from scipy.spatial.distance import squareform

#выделение вектора-слова
def get_vector(words, word_matrix, word):
    for y, x in enumerate(words):
        if x == word:
            return word_matrix[y]
    return 0


# функции-метрики
def div_kl(p, q):
    ans = 0
    for i in range(len(p)):
        if p[i] != 0 and q[i] != 0:
            ans += p[i] * math.log(p[i] / q[i])
    return ans


def div_gs(p, q):
    c = []
    for i in range(len(p)):
        c.append((p[i] + q[i]) / 2)
    return div_kl(p, c) + div_kl(q, c)


def jaccard(v1, v2):
    x, y = 0, 0
    for i in range(len(v1)):
        x += min(v1[i], v2[i])
        y += max(v1[i], v2[i])
    return x / y


def cosine_similarity(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


morph = pymorphy2.MorphAnalyzer()

stop_words = stopwords.words('english')
n = 10
ws = 5  # ширина контекста

word_set = []
D = []
dict_freq_word = {}

# построение word-contex матрицы
for index in range(1, n + 1):
    sum_w = 0
    print("текст {id}.txt".format(id=index))
    with open("текст 4_{id}.txt".format(id=index), encoding='utf-8') as f:
        for line in f:
            tokens_from_line = Tokenize.tokenize_word(line.strip())
            tokens_from_line = [i for i in tokens_from_line if i not in stop_words]
            sum_w += len(tokens_from_line)
            words = [morph.parse(word)[0].normal_form for word in tokens_from_line]

            for token in set(words):
                if token not in word_set:
                    word_set.append(token)
                    dict_freq_word[token] = 0
                    D.append([0] * len(D))
                    for i in range(len(D)):
                        D[i].append(0)

            for word in words:
                dict_freq_word[word] += 1

            for i in range(len(word_set)):
                for token_number in [y for y, x in enumerate(words) if x == word_set[i]]:
                    for j in range(len(word_set)):
                        for k in range(token_number - ws, token_number + ws + 1):
                            if k < 0:
                                continue
                            if k > len(words) - 1:
                                continue
                            if words[k] == word_set[j]:
                                D[j][i] += 1

for word in word_set:
    dict_freq_word[word] /= sum_w

# построение  PPMI-модели
D2_ppmi = [[0] * len(word_set) for _ in range(len(word_set))]

for i in range(len(word_set)):
    for j in range(len(word_set)):
        if D[i][j] == 0:
            D2_ppmi[i][j] = 0
        else:
            D2_ppmi[i][j] = math.log(
                (D[i][j] / sum(D[i])) / (dict_freq_word[word_set[i]] * dict_freq_word[word_set[j]]))
        if D2_ppmi[i][j] < 0:
            D2_ppmi[i][j] = 0

# получение векторов-слов для тестирования
test_list = ['planet', 'sun', 'opera', 'industry', 'money', 'cash', 'bank',
             'credit', 'card', 'information', 'computer',
             'internet', 'software','professor', 'cucumber', 'doctor']

def d(coord, word):
    i, j = coord
    v1 = get_vector(word_set, D2_ppmi, word[i])
    v2 = get_vector(word_set, D2_ppmi, word[j])
    return jaccard(v1, v2)


tri = np.triu_indices(len(test_list), 1)
weights = np.apply_along_axis(d, 0, tri, test_list)
print(tri)
print(weights)


Z = linkage(weights, 'ward')
dn = dendrogram(Z, labels=np.array(test_list), leaf_rotation=90, leaf_font_size=7)
fig = plt.figure(figsize=(10, 5))

filename = "wordsim353crowd.csv"
wordsim353 = pandas.read_csv(filename)

dict_test_word = {}
for word in test_list:
    dict_test_word[word] = get_vector(word_set, D, word)

print('pair 1: planet & sun')
print('kl:', div_kl(dict_test_word['planet'], dict_test_word['sun']))
print('gs:', div_gs(dict_test_word['planet'], dict_test_word['sun']))
print('cos:', cosine_similarity(dict_test_word['planet'], dict_test_word['sun']))
print('jac:', jaccard(dict_test_word['planet'], dict_test_word['sun']))
print(wordsim353[wordsim353['Word 2'] == 'sun'], '\n')

print('pair 2: opera & industry')
print('kl:', div_kl(dict_test_word['opera'], dict_test_word['industry']))
print('gs:', div_gs(dict_test_word['opera'], dict_test_word['industry']))
print('cos:', cosine_similarity(dict_test_word['opera'], dict_test_word['industry']))
print('jac:', jaccard(dict_test_word['opera'], dict_test_word['industry']))
print(wordsim353[wordsim353['Word 2'] == 'industry'], '\n')

print('pair 3: money & cash')
print('kl:', div_kl(dict_test_word['money'], dict_test_word['cash']))
print('gs:', div_gs(dict_test_word['money'], dict_test_word['cash']))
print('cos:', cosine_similarity(dict_test_word['money'], dict_test_word['cash']))
print('jac:', jaccard(dict_test_word['money'], dict_test_word['cash']))
print(wordsim353[wordsim353['Word 2'] == 'cash'], '\n')

print('pair 4: bank & money')
print('kl:', div_kl(dict_test_word['bank'], dict_test_word['money']))
print('gs:', div_gs(dict_test_word['bank'], dict_test_word['money']))
print('cos:', cosine_similarity(dict_test_word['bank'], dict_test_word['money']))
print('jac:', jaccard(dict_test_word['bank'], dict_test_word['money']))
print(wordsim353[wordsim353['Word 1'] == 'bank'], '\n')

print('pair 5: credit & card')
print('kl:', div_kl(dict_test_word['credit'], dict_test_word['card']))
print('gs:', div_gs(dict_test_word['credit'], dict_test_word['card']))
print('cos:', cosine_similarity(dict_test_word['credit'], dict_test_word['card']))
print('jac:', jaccard(dict_test_word['credit'], dict_test_word['card']),'\n')


print('pair 6: credit & information')
print('kl:', div_kl(dict_test_word['credit'], dict_test_word['information']))
print('gs:', div_gs(dict_test_word['credit'], dict_test_word['information']))
print('cos:', cosine_similarity(dict_test_word['credit'], dict_test_word['information']))
print('jac:', jaccard(dict_test_word['credit'], dict_test_word['information']))
print(wordsim353[wordsim353['Word 1'] == 'credit'], '\n')

print('pair 7: computer & internet')
print('kl:', div_kl(dict_test_word['computer'], dict_test_word['internet']))
print('gs:', div_gs(dict_test_word['computer'], dict_test_word['internet']))
print('cos:', cosine_similarity(dict_test_word['computer'], dict_test_word['internet']))
print('jac:', jaccard(dict_test_word['computer'], dict_test_word['internet']))
print(wordsim353[wordsim353['Word 2'] == 'internet'], '\n')

print('pair 8: computer & software')
print('kl:', div_kl(dict_test_word['computer'], dict_test_word['software']))
print('gs:', div_gs(dict_test_word['computer'], dict_test_word['software']))
print('cos:', cosine_similarity(dict_test_word['computer'], dict_test_word['software']))
print('jac:', jaccard(dict_test_word['computer'], dict_test_word['software']))
print(wordsim353[wordsim353['Word 2'] == 'software'], '\n')

print('pair 9: professor & cucumber')
print('kl:', div_kl(dict_test_word['professor'], dict_test_word['cucumber']))
print('gs:', div_gs(dict_test_word['professor'], dict_test_word['cucumber']))
print('cos:', cosine_similarity(dict_test_word['professor'], dict_test_word['cucumber']))
print('jac:', jaccard(dict_test_word['professor'], dict_test_word['cucumber']))


print('pair 10: professor & doctor')
print('kl:', div_kl(dict_test_word['professor'], dict_test_word['doctor']))
print('gs:', div_gs(dict_test_word['professor'], dict_test_word['doctor']))
print('cos:', cosine_similarity(dict_test_word['professor'], dict_test_word['doctor']))
print('jac:', jaccard(dict_test_word['professor'], dict_test_word['doctor']))
print(wordsim353[wordsim353['Word 1'] == 'professor'], '\n')

plt.show()
