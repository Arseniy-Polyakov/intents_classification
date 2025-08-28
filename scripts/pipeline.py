import re
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

def metrics(y_pred: list, y_test: list) -> dict:
    """
    Функция для определения основных метрик качества модели многоклассовой классификации:
    1. Accuracy
    2. Macro precision
    3. Macro recall
    4. Macro f1-мера
    """
    accuracy = round(accuracy_score(y_pred, y_test), 2)
    precision = round(precision_score(y_pred, y_test, average="macro", zero_division=0), 2)
    recall = round(recall_score(y_pred, y_test, average="macro", zero_division=0), 2)
    f1 = round(f1_score(y_pred, y_test, average="macro"), 2)
    scores = {
        "accuracy_score": accuracy,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1
    }
    return scores

class Vectorization:
    def __init__(self, model_name, sentences):
        self.model_name = model_name
        self.sentences = sentences

    def tfidf_vectorization(self):
        """
        Функция для векторизации текста с помощью алгоритма tf-idf
        """
        vectorizer = self.model_name
        embeddings = vectorizer.fit_transform(self.sentences)
        return embeddings

    def sentence_transformers_vectorization(self):
        """
        Фунцкия для векторизации текста с помощью трансформеров
        """
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(self.sentences)
        return embeddings

class Classification:
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def logistic_regression(self):
        """
        Функция для обучения модели логистической регрессии
        """
        model = LogisticRegression(multi_class="multinomial")
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.labels, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_metrics = metrics(y_pred, y_test)
        return all_metrics

    def k_neighbours(self):
        """
        Функция для обучения метрической модели K-ближайших соседей
        """
        model = KNeighborsClassifier(n_neighbors=3)
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.labels, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_metrics = metrics(y_pred, y_test)
        return all_metrics

    def naive_bayes(self):
        """
        Функция для обучения модели гауссовского наивного байеса
        """
        model = GaussianNB()
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.labels, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_metrics = metrics(y_pred, y_test)
        return all_metrics

    def svm(self):
        """
        Функция для обучения модели опорных векторов
        """
        model = SVC()
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.labels, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_metrics = metrics(y_pred, y_test)
        return all_metrics

    def random_forest(self):
        """
        Функция для обучения ансамблевой модели рандомного леса
        """
        model = RandomForestClassifier(n_estimators=100)
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.labels, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_metrics = metrics(y_pred, y_test)
        return all_metrics

    def gradient_boosting(self):
        """
        Функция для обучения моделей градиентного бустинга:  
        1. LightGBM
        2. XGBoost
        3. CatBoost
        """
        # model = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=6, loss_function='MultiClass')
        # model = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=6, learning_rate=0.05, n_estimators=100, verbose=-100)
        model = XGBClassifier(num_leaves=31, max_depth=6, learning_rate=0.05, n_estimators=100)
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.labels, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_metrics = metrics(y_pred, y_test)
        return all_metrics

def vectorization_all(dataset: list, vectorization_model: str) -> tuple:
    """
    Функция для нахождения эмбеддингов всего датасета и кодирования категориальных меток
    """
    texts = []
    labels = []
    for i in range(len(dataset)):
        try:
            texts.append(re.sub(r"[^\w\s]", "", dataset[i].split(":")[1]).strip())
            labels.append(re.sub(r"[^\w\s]", "", dataset[i].split(":")[0]).strip())
        except:
            i += 1

    encoder = LabelEncoder()
    labels_final = encoder.fit_transform(labels)

    vectorization = Vectorization(vectorization_model, texts)
    if vectorization_model == "sergeyzh/rubert-tiny-turbo" or vectorization_model == "cointegrated/rubert-tiny2":
        embeddings = vectorization.sentence_transformers_vectorization() # для нахождения эмбеддингов с помощью трансформеров
    else:
        embeddings = vectorization.tfidf_vectorization() # для алгоритма tf-idf векторизации 
    return embeddings, labels_final

def classification_all(embeddings: list, labels_final: list) -> tuple:
    """
    Функция для обучения всех моделей и оценки их качества по метрикам
    """
    classification = Classification(embeddings, labels_final)
    logistic_regression_results = classification.logistic_regression()
    knn_results = classification.k_neighbours()
    naive_bayes_results = classification.naive_bayes()
    svm_results = classification.svm()
    random_forest_results = classification.random_forest()
    gradient_boosting_results = classification.gradient_boosting()
    return logistic_regression_results, knn_results, naive_bayes_results, svm_results, random_forest_results, gradient_boosting_results

def experiment_all(vectorization_model: str) -> pd.DataFrame:
    """
    Функция для проведения экспериментов над n-количеством предобработанных текстов:
    1. Парсинг предобработанного файла
    2. Нахождение эмбеддингов (векторизация)
    3. Обучение моделей
    4. Оценка качества моделей по метрикам
    5. Вывод результатов эксперимента (метрик алгоритмов для определенной модели векторизации) в виде дата фрейма
    """
    intent = []
    all = []

    for i in tqdm(range(1, 77)):
        with open(f"datasets\intents_preprocessed\intents_preprocessed_{i}.json", "rt", encoding="utf-8") as file:
            result = file.read().split(", ")
        embeddings = vectorization_all(result, vectorization_model)[0]
        labels_final = vectorization_all(result, vectorization_model)[1]
        intent.append(classification_all(embeddings, labels_final)[0])
        intent.append(classification_all(embeddings, labels_final)[1])
        intent.append(classification_all(embeddings, labels_final)[2])
        intent.append(classification_all(embeddings, labels_final)[3])
        intent.append(classification_all(embeddings, labels_final)[4])
        intent.append(classification_all(embeddings, labels_final)[5])
        all.append(intent)
        intent = []

    columns = ["Logistic Regression", "KNN", "Naive Bayes", "SVM", "Random Forest", "Gradient Boosting"]
    df = pd.DataFrame(data=all, index=[f"intents_{i}" for i in range(1, 77)], columns=columns)
    print(df)

experiment_all("cointegrated/rubert-tiny2")