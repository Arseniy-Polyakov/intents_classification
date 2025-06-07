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

            for _ in tqdm(range(self.n_iters)):
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

    X_train, X_test, y_train, y_test = train_test_split(dataset_features, dataset_labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_pred, y_test), 
        "precision": precision_score(y_pred, y_test),
        "recall": recall_score(y_pred, y_test), 
        "f1": f1_score(y_pred, y_test)
    }

    target_classificator = model.predict(target_dataset)
    return target_classificator, scores