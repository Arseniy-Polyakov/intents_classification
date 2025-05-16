import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from general import index_to_classificator

start = time.time()
def perceptron_best_classificator(dataset_features: list, dataset_labels: list, target_dataset: list) -> tuple:
    class Perceptron:
        def __init__(self, learning_rate=0.01, n_iters=1000):
            self.lr = learning_rate
            self.n_iters = n_iters
            self.activation_func = self._unit_step_func
            self.weights = None
            self.bias = None

        def fit(self, X, y):
            n_samples, n_features = X.shape

            self.weights = np.zeros(n_features)
            self.bias = 0

            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    linear_output = np.dot(x_i, self.weights) + self.bias
                    y_predicted = self.activation_func(linear_output)

                    update = self.lr * (y[idx] - y_predicted)
                    self.weights += update * x_i
                    self.bias += update

        def predict(self, X):
            linear_output = np.dot(X, self.weights) + self.bias
            y_predicted = self.activation_func(linear_output)
            return y_predicted

        def _unit_step_func(self, x):
            return np.where(x>=0, 1, 0)

    model = Perceptron(learning_rate=0.01, n_iters=100)

    # X_train, X_test, y_train, y_test = train_test_split(dataset_features, dataset_labels, test_size=0.2, random_state=42)
    model.fit(dataset_features, dataset_labels)
    # y_pred = model.predict(X_test)
    # scores = {
    #     "accuracy": accuracy_score(y_pred, y_test), 
    #     "precision": precision_score(y_pred, y_test),
    #     "recall": recall_score(y_pred, y_test), 
    #     "f1": f1_score(y_pred, y_test)
    # }

    target_classificator = model.predict(target_dataset)
    return target_classificator

# Target intent: [25, 0.89]
# Intents with F1 score: [18, 0.93, 0.54], [42, 0.9, 0.67], [176, 0.97, 0.23], [13, 0.89, 0.4], [15, 0.93, 0.86], [25, 0.89, 0.9]]
dataset_features = np.array([[18, 0.93], [42, 0.9], [176, 0.97], [13, 0.89], [15, 0.93], [25, 0.89]])
dataset_labels = np.array([2, 2, 2, 1, 0, 2])
target_dataset = np.array([[25, 0.89]])

best_classificator = index_to_classificator(perceptron_best_classificator(dataset_features, dataset_labels, target_dataset)[0])
# scores = perceptron_best_classificator(dataset_features, dataset_labels, target_dataset)[1]
end = time.time()

print("Best classificator: " + best_classificator)
print("----------------")
print("Time: " + str(end-start))
# print("----------------")
# print("Scores: ")
# print(scores)


