import re
import numpy as np
import json_repair
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
    The function for russian texts preprocessing: making the lowercase; removing stopwords, 
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
    intents = {}
    with open(file_name, "rt", encoding="utf-8") as file:
        sample_str = file.read()
        sample_json = json_repair.loads(sample_str)
        for intent in sample_json:
            intents[intent["path"]] = [phrase["text"].strip() for phrase in intent["phrases"]]
    intents_texts = list(intents.items())
    train_texts = [preprocessing(sentence) for intent in intents_texts for sentence in intent[1]]
    train_labels = [intent[0] for intent in intents_texts for _ in intent[1]]
    return train_texts, train_labels

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

def classes(file_name: str) -> list:
    """
    The function for finding the number of phrases in each intent
    """
    with open(file_name, "rt", encoding="utf-8") as file:
        text = file.read()
    json_file = json_repair.loads(text) 
    classes = [len(item["phrases"]) for item in json_file]
    dataset_volume = len(classes)
    return dataset_volume, classes

def classificator_to_index(classificator: str) -> int:
    classificators_indexes = {
        # "knn": 0, 
        # "logistic_regression": 1, 
        # "naive_bayes": 2,
        "svm": 0, 
        "gradient_boosting": 1, 
        "random_forest": 2
    }
    return classificators_indexes[classificator]

def index_to_classificator(index: int) -> str:
    classificators_indexes = {
        0: "svm", 
        1: "gradient_boosting",
        2: "random_forest"
        # 3: "svm",
        # 4: "gradient_boosting",
        # 5: "random_forest"
    }
    return classificators_indexes[index]

def dataset_stat(file_name: str) -> dict:
    """
    The function for finding dataset features (volume, gini index - classes' balance) 
    and the best classificator according to the F1 score
    Comparing: knn, logistic regression, naive bayes, svm, random rorest, gradient boosting
    """
    train_texts = parsing_train_data(file_name)[0]
    train_labels = parsing_train_data(file_name)[1]

    dataset_volume = classes(file_name)[0]
    gini_index = round(float(index_gini(classes(file_name)[1])), 2)
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
        "dataset_volume": dataset_volume, 
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

# def train_datasets_whole_stat(datasets_paths: list) -> list:
#     train_data = []
#     train_labels = []
#     for path in datasets_paths:
#         scores = dataset_stat(path)
#         train_data.append([scores["dataset_volume"], scores["gini_index"]])
#         train_labels.append(scores["best_classificator_index"])
#     return train_data, train_labels

path = "intents/intents.json"