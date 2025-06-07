from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def logistic_regression(train_texts: list, train_labels: list) -> list: 
    """
    The logistic regression function for text classification
    """
    model = LogisticRegression()
    vectorizer = TfidfVectorizer()
    vector_texts = vectorizer.fit_transform(train_texts)
    X_train, X_test, y_train, y_test = train_test_split(vector_texts, train_labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
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