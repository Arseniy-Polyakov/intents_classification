import re

import pandas as pd
import numpy as np
import json_repair
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from classical_machine_learning.knn import knn
from classical_machine_learning.logistic_regression import logistic_regression
from classical_machine_learning.naive_bayes import naive_bayes_classificator
from classical_machine_learning.svm import support_vector_machine 

from ensembles.gradient_boosting import gradient_boosting
from ensembles.random_forest import random_forest_classifier

stopwords = stopwords.words("russian")

def preprocessing(text: str) -> str:
    """
    The function for russian texts preprocessing: making the lowercase, removing stopwords, 
    punctuation marks and symbols from other languages; lemmatization
    """
    lemmatizer = MorphAnalyzer()
    text_lower = text.lower().strip().split()
    text_without_stopwords = " ".join([word for word in text_lower if word not in stopwords])
    text_cleaned = re.sub(r"[^А-Яа-яёЁ\-\s]", "", text_without_stopwords)
    text_lemmatized = " ".join([lemmatizer.parse(word)[0].normal_form for word in text_cleaned.split()])
    return text_lemmatized

def parsing_train_data(file_name: str) -> tuple:
    """
    The function for parsing JSON file with train data for the model
    """
    with open(file_name, "rt", encoding="utf-8") as file:
        sample_str = file.read()
        sample_json = json_repair.loads(sample_str)
    intents = {}
    for i in tqdm(range(len(sample_json))):
        try:
            intents[sample_json[i]["path"]] = [preprocessing(phrase["text"]) for phrase in sample_json[i]["phrases"]]
        except:
            i += 1
    
    number_intents = len(list(intents.keys()))

    train_texts = []
    train_labels = []
    number_sentences_per_intent = []
    for intent in list(intents.items()):
        number_sentences_per_intent.append(len(intent[1]))
        for sentence in intent[1]:
            if len(sentence) > 3:
                train_texts.append(sentence)
                train_labels.append(intent[0])

    return number_intents, number_sentences_per_intent, train_texts, train_labels

def index_gini(dataset: list) -> float:
    """
    The metric which helps to evaluate a disbalance in a dataset
    """
    dataset = np.array(dataset)
    gini_index = 1
    dataset_unique = np.unique(dataset)
    for item in dataset_unique:
        part = np.sum(dataset == item) / len(dataset)
        gini_index -= part ** 2
    return gini_index

def classificator_to_index(classificator: str) -> int:
    classificators_indexes = {
        "knn": 0, 
        "logistic_regression": 1, 
        "naive_bayes": 2,
        "svm": 3, 
        "gradient_boosting": 4, 
        "random_forest": 5
    }
    return classificators_indexes[classificator]

def index_to_classificator(index: int) -> str:
    classificators_indexes = {
        0: "knn", 
        1: "logistic_regression",
        2: "naive_bayes",
        3: "svm",
        4: "gradient_boosting",
        5: "random_forest"
    }
    return classificators_indexes[index]

def dataset_stat(file_name: str) -> dict:
    """
    The function for finding dataset features (volume, gini index - classes' balance) 
    and the best classificator according to the F1 score
    Comparing: knn, logistic regression, naive bayes, svm, random rorest, gradient boosting
    """
    parsing_data = parsing_train_data(file_name)
    number_intents, number_sentences_per_intent, train_texts, train_labels = parsing_data[0], parsing_data[1], parsing_data[2], parsing_data[3]
    number_sentences = sum(number_sentences_per_intent)
    gini_index = round(float(index_gini(number_sentences_per_intent)), 2)

    knn_f1 = knn(train_texts, train_labels)["f1_score"]
    logistic_regression_f1 = logistic_regression(train_texts, train_labels)["f1_score"]
    naive_bayes_f1 = naive_bayes_classificator(train_texts, train_labels)["f1_score"]
    svm_f1 = support_vector_machine(train_texts, train_labels)["f1_score"]
    gradient_boosting_f1 = gradient_boosting(train_texts, train_labels)["f1_score"]
    random_forest_f1 = random_forest_classifier(train_texts, train_labels)["f1_score"]

    scores = {
        "knn": knn_f1, 
        "logistic_regression": logistic_regression_f1, 
        "naive_bayes": naive_bayes_f1,
        "svm": svm_f1, 
        "gradient_boosting": gradient_boosting_f1, 
        "random_forest": random_forest_f1
    }

    scores_list = sorted(list(scores.items()), key=lambda x:x[1], reverse=True)

    results = {
        "number_intents": number_intents, 
        "number_sentences": number_sentences,
        "gini_index": gini_index, 
        "knn": knn_f1, 
        "logistic_regression": logistic_regression_f1, 
        "naive_bayes": naive_bayes_f1, 
        "svm": svm_f1, 
        "gradient_boosting": gradient_boosting_f1, 
        "random_forest": random_forest_f1, 
        "max_f1": scores_list[0][1],
        "best_classificator": scores_list[0][0],
        "best_classificator_index": classificator_to_index(scores_list[0][0])
    }
    return results

def overall_stat(file_name: str) -> tuple:
    """
    Parsing the CSV file with experiments' data and fincind the most efficient algorithm
    """
    df = pd.read_csv(file_name, sep=";")
    number_intents = list(df.iloc[0][1:].astype("int"))
    number_sentences = list(df.iloc[1][1:].astype("int"))
    gini_index = list(df.iloc[2][1:].astype("float"))
    intents_sentences_gini = [[number_intents[i], number_sentences[i], gini_index[i]] for i in range(len(number_sentences))]

    best_classificator_index = list(df.iloc[11][1:].astype("int"))
    frequency_classificator = {}
    for index in best_classificator_index:
        if index_to_classificator(index) not in frequency_classificator:
            frequency_classificator[index_to_classificator(index)] = 1
        else:
            frequency_classificator[index_to_classificator(index)] += 1
    methods = ["knn", "logistic_regression", "naive_bayes", "svm", "gradient_boosting", "random_forest"]
    for method in methods:
        if method not in frequency_classificator:
            frequency_classificator[method] = 0
    frequency_classificator = dict(sorted(frequency_classificator.items(), key = lambda x:x[1], reverse=True))
    return intents_sentences_gini, best_classificator_index, frequency_classificator

def graphics(data: list):
    """
    Making box-plot graphics for the number of intents, sentences and gini range in the dataset
    """
    intents = np.array([item[0] for item in data])
    sentences = [item[1] for item in data]
    gini = [item[2] for item in data]
    sns.boxplot(intents)
    plt.title("Распределение по количеству интентов")
    plt.show()
    sns.boxplot(sentences)
    plt.title("Распределение по количеству предложений")
    plt.show()
    sns.boxplot(gini)
    plt.title("Распределение по gini индексу")
    plt.show()

graphics(overall_stat("tests.csv")[0])