import gensim
import Tokenize
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.cluster.hierarchy import dendrogram, linkage


def lemmatize_sentence(sentence):
    stop_words = stopwords.words('english')
    tokens = Tokenize.tokenize_word(sentence)
    without_stopwords = [word.lower() for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for token, tag in nltk.pos_tag(without_stopwords):
        mapped_tag = Tokenize.tag_map(tag[0])
        lemma = lemmatizer.lemmatize(token, mapped_tag)
        lemmas.append(lemma)
    return lemmas


# функции-метрики
def div_kl(p, q):
    ans = 0
    for i in range(len(p)):
        if p[i] != 0 and q[i] != 0:
            ans += p[i] * math.log(p[i] / q[i])
    return ans


def div_js(p, q):
    c = []
    for i in range(len(p)):
        c.append((p[i] + q[i]) / 2)
    return div_kl(p, c) + div_kl(q, c)


def jaccard_(v1, v2):
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


def get_word_set(corpus):
    word_set = []
    for document in corpus:
        for sentence in document:
            for word in list(set(sentence)):
                if word not in word_set:
                    word_set.append(word)
    return word_set


def get_document_set(corpus):
    document_set = []
    for i in range(len(corpus)):
        document_set.append('document{}'.format(i))
    return document_set


def get_corpus(group_name, n):
    corpus = []
    for i in range(1, n + 1):
        with open('docs/{}{}.txt'.format(group_name, i), encoding='utf-8') as f:
            corpus.append(f.read().rstrip())
    return corpus

def get_word_context_matrix(corpus, word_set, ws=5):
    word_count = len(word_set)
    word_context_matrix = pd.DataFrame(np.zeros([word_count, word_count]), columns=word_set, index=word_set)
    for document in corpus:
        for sentence in document:
            for i, word in enumerate(sentence):
                for j in range(max(i - ws, 0), min(i + ws, len(sentence))):
                    word_context_matrix[word][sentence[j]] += 1
    return word_context_matrix


def get_document_context_matrix(corpus, word_set, document_set):
    word_count = len(word_set)
    document_count = len(corpus)
    document_context_matrix = pd.DataFrame(np.zeros([document_count, word_count]), columns=word_set, index=document_set)
    for i, document in enumerate(corpus):
        for sentence in document:
            for word in sentence:
                document_context_matrix.iloc[i][word] += 1
    return document_context_matrix


def get_pmi(df, ppmi=True):
    col_totals = df.sum(axis=0)
    row_totals = df.sum(axis=1)
    sum_value = col_totals.sum()
    expected = np.outer(row_totals, col_totals) / sum_value
    df = df / expected
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0
    if ppmi:
        df[df < 0] = 0.0
    return normalize_dataframe(df)


def get_linkages(indeces, words, ppmi_df):
    cosine_weights = np.apply_along_axis(cosine_similarity, 0, indeces, words, ppmi_df)
    jaccard_weights = np.apply_along_axis(jaccard_, 0, indeces, words, ppmi_df)
    kullback_leibler_weights = np.apply_along_axis(div_kl, 0, indeces, words, ppmi_df,
                                                   positive=False)
    jensen_shannon_weights = np.apply_along_axis(div_js, 0, indeces, words, ppmi_df,
                                                 positive=True)
    return [linkage(cosine_weights, 'ward'),
            linkage(jaccard_weights, 'ward'),
            linkage(kullback_leibler_weights, 'ward'),
            linkage(jensen_shannon_weights, 'ward')]


def draw_matrix_heatmap(df, word_set):
    figure(figsize=(15, 10))
    df = pd.DataFrame(df, index=word_set, columns=word_set)
    sns.heatmap(df, annot=True, xticklabels=word_set,yticklabels=word_set,linewidths=.5,linecolor='black',cmap="Blues")
    plt.yticks(rotation=0)
    plt.title('Word-Context Matrix')
    plt.show()


def plot_dendrogram(linkages, title, words):
    template_title = 'Hierarchical Clustering Dendrogram - {}'
    titles = []
    if isinstance(linkages, list):
        for i in range(len(linkages)):
            titles.append(template_title.format(title[i]))
            dendrogram(linkages[i], labels=np.array(words), orientation="right", leaf_font_size=8)
            plt.title(titles[i])
            plt.show()
    else:
        titles.append(template_title.format(title))
        dendrogram(linkages, labels=np.array(words), orientation="right", leaf_font_size=8)
        plt.title(titles)
        plt.show()


def draw(wordsim_pairs, ppmi_df, word_context_df, word_set):
    plot_dendrogram(linkage(ppmi_df, 'ward'), 'Weighted matrix', ppmi_df.columns)
    indeces = np.triu_indices(len(ppmi_df.columns), 1)
    linkages = get_linkages(indeces, wordsim_pairs, ppmi_df)
    titles = ['Cosine', 'Jaccard', 'Kullback-Leibler', 'Jensen-Shannon']
    plot_dendrogram(linkages, titles, wordsim_pairs)
    draw_matrix_heatmap(word_context_df, word_set)


def compare_word(df, w1, w2, method):
    if method == 'PLSA':
        w1_df, w2_df = df.wv[w1], df.wv[w2]
    elif method == 'WORD2VEC':
        w1_df, w2_df = df.loc[:, w1], df.loc[:, w2]
    print(method)
    wordsim353 = pandas.read_csv('docs/wordsim353.csv')
    wordsim353['Human (Mean)'] = normalize_series(wordsim353['Human (Mean)'])
    print('{} - {}'.format(w1, w2))
    wordsim = wordsim353[(wordsim353['Word 1'] == w1) & (wordsim353['Word 2'] == w2) |
                         (wordsim353['Word 1'] == w2) & (wordsim353['Word 2'] == w1)]
    print(wordsim)
    cosine = cosine_similarity(w1_df, w2_df)
    print('Cosine: {}'.format(cosine))
    jaccard = jaccard_(w1_df, w2_df)
    print('Jaccard: {}'.format(jaccard))
    kullback_leibler = div_kl(w1_df, w2_df)
    print('Kullback-Leibler: {}'.format(kullback_leibler))
    jensen_shannon = div_js(w1_df, w2_df)
    print('Jensen-Shannon: {}\n'.format(jensen_shannon))
    result = {'wordsim353': wordsim.iloc[0]['Human (Mean)'], 'cosine': cosine, 'jaccard': jaccard,
              'kullback_leibler': kullback_leibler, 'jensen_shannon': jensen_shannon}
    return result


def get_pearson_coef(x, y):
    x_avg = sum(x) / len(x)
    y_avg = sum(y) / len(y)
    numerator, denominator_x, denominator_y = 0, 0, 0
    for i in range(len(x)):
        numerator += (x[i] - x_avg) * (y[i] - y_avg)
        denominator_x += (x[i] - x_avg)**2
        denominator_y += (y[i] - y_avg)**2
    return numerator / (denominator_x * denominator_y)**0.5


def normalize_dataframe(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def normalize_series(series, min_max=True):
    if min_max:
        return (series - series.min()) / (series.max() - series.min())
    else:
        return (series - series.mean()) / series.std()


def plsa(corpus, word_set, document_set, number_of_topics=5, max_iterations=1):
    word_count = len(word_set)
    document_count = len(document_set)

    document_context_matrix = get_document_context_matrix(corpus, word_set, document_set)
    document_topic_probability = normalize_dataframe(
        pd.DataFrame(np.random.random(size=(document_count, number_of_topics)), index=document_set))
    topic_word_probability = normalize_dataframe(pd.DataFrame(np.random.random(size=(number_of_topics, word_count)),
                                                              columns=word_set))
    topic_probability = np.zeros([document_count, word_count, number_of_topics])
    # E-Step
    for iteration in range(max_iterations):
        for i, doc in enumerate(document_set):
            for j, word in enumerate(word_set):
                probability = document_topic_probability.loc[doc, :] * topic_word_probability.loc[:, word]
                topic_probability[i][j] = normalize_series(probability)
    # M-Step
    for topic in range(number_of_topics):
        for j, word in enumerate(word_set):
            result = 0
            for i, doc in enumerate(document_set):
                result += document_context_matrix.loc[doc][word] * topic_probability[i, j, topic]
            topic_word_probability.loc[topic][word] = result
        topic_word_probability.loc[topic] = normalize_series(topic_word_probability.loc[topic])
    for _i, _doc in enumerate(document_set):
        for _topic in range(number_of_topics):
            result = 0
            for j, _word in enumerate(word_set):
                result += document_context_matrix.loc[doc][word] * topic_probability[i, j, topic]
            document_topic_probability.loc[_doc][topic] = result
        document_topic_probability.loc[doc] = normalize_series(document_topic_probability.loc[doc])
    return topic_word_probability


def word2vec(corpus):
    sentences = [[j for i in [sentence for sentence in k] for j in i] for k in corpus]
    model = gensim.models.Word2Vec(sentences, min_count=0)
    return model


def show(result, method):
    wordsim_values, cosine_values, jaccard_values, kullback_leibler_values, jensen_shannon_values = [], [], [], [], []
    for dict_item in result:
        wordsim_values.append(dict_item['wordsim353'])
        cosine_values.append(dict_item['cosine'])
        jaccard_values.append(dict_item['jaccard'])
        kullback_leibler_values.append(dict_item['kullback_leibler'])
        jensen_shannon_values.append(dict_item['jensen_shannon'])
    print(method)
    print('Cosine: {}'.format(get_pearson_coef(wordsim_values, cosine_values)))
    print('Jaccard_values: {}'.format(get_pearson_coef(wordsim_values, jaccard_values)))
    print('Kullback_leibler_values: {}'.format(get_pearson_coef(wordsim_values,                                                                         kullback_leibler_values)))
    print('Jensen_shannon_values: {}'.format(get_pearson_coef(wordsim_values,jensen_shannon_values)))



dict_files = {('planet','sun'): 9, ('opera', 'industry'): 8, ('money', 'cush'): 8, ('bank', 'money'): 8,
              ('credit', 'card'): 8, ('credit', 'information'): 8, ('computer', 'internet'): 8, ('computer', 'software'): 8,
              ('professor', 'cucumber'): 8, ('professor', 'doctor'): 11}
result_base, result_plsa, result_word2vec = [], [], []
for group_names, count in dict_files.items():
    first_word, second_word = group_names
    corpus = get_corpus(first_word, count)
    corpus = [[lemmatize_sentence(sentence) for sentence in Tokenize.tokenize_sentence(doc)] for doc in corpus]

    word_set = get_word_set(corpus)
    document_set = get_document_set(corpus)

    word_context_df = get_word_context_matrix(corpus, word_set)
    ppmi_df = get_pmi(word_context_df)
    result_plsa.append(compare_word(plsa(corpus, word_set, document_set), first_word, second_word, 'PLSA'))
    result_word2vec.append(compare_word(word2vec(corpus), first_word, second_word, 'WORD2VEC'))
    wordsim_pairs = [(first_word, second_word)]
    draw(wordsim_pairs, ppmi_df, word_context_df, word_set)

show(result_plsa, 'PLSA')
show(result_word2vec, 'WORD2VEC')
