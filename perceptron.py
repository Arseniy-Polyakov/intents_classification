import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from main import index_to_classificator, overall_stat

def perceptron(dataset_features: list, dataset_labels: list, target_dataset: list) -> tuple:
    """
    The perceptron algorithm for finding the best classificator
    """
    perceptron = Perceptron(max_iter=1000, eta0=0.5)
    X_train, X_test, y_train, y_test = train_test_split(dataset_features, dataset_labels, test_size=0.2, random_state=42)
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_pred, y_test), 
        "precision": precision_score(y_pred, y_test, average="weighted", zero_division=0),
        "recall": recall_score(y_pred, y_test, average="weighted", zero_division=0), 
        "f1": f1_score(y_pred, y_test, average="weighted", zero_division=0)
    }
    target_classificator = index_to_classificator(perceptron.predict(target_dataset)[0])
    return scores, target_classificator

target_dataset = [[4, 20, 0.98]]
dataset_features, dataset_labels = overall_stat("tests.csv")[0], overall_stat("tests.csv")[1]
model_data = perceptron(dataset_features, dataset_labels, target_dataset)
model_scores = model_data[0]
target_classificator = model_data[1]
print(target_classificator)
print(model_scores)

