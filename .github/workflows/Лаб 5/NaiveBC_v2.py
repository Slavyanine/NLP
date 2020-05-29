import numpy as np
import pandas as pd
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('input\spam_copy.csv', encoding='latin-1')
data.head(n=10)

f = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = f.fit_transform(data["v2"])
np.shape(X)
data["v1"]=data["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33, random_state=42)
print([np.shape(X_train), np.shape(X_test)])

# Обучаем байсовскую модель,меняя параметр регуляризции α,
# оцениваем Accuracy', Recall',  Precision .с помощью тестового набора

list_alpha = np.arange(1/100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1

# Пронаблюдаем динамику изменений метрик при различных значениях α

matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns =
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
print(models.head(n=30))

#По результатам видим, что при увеличении альфа увеличивается точность,
# но постепенно падает полнота. Баланс значений метрик достигается исходя из условий поставленных задач
# Например, считаем, что хуже классифицировать неправильно не-спам, тогда test_precision должно стремиться к 1.