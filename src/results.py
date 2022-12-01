import time

from sklearn.metrics import confusion_matrix, classification_report

from src.alias import Classifier, Split


def print_results(model: Classifier, train_time: float, dataset: Split):
    # Unpack the dataset tuple
    X_train, X_test, y_train, y_test = dataset

    # Make predictions
    start = time.time()
    y_test_predicted = model.predict(X_test)
    end = time.time()

    print("Training Accuracy: {0:.6g}".format(model.score(X_train, y_train)))
    print("Test Accuracy:     {0:.6g}".format(model.score(X_test, y_test)))
    print("Training length (in seconds):   {0:.2g}".format(train_time))
    print("Prediction length (in seconds): {0:.2g}".format(end - start))
    print("\n=====\nConfusion Matrix\n=====")
    print(confusion_matrix(y_test, y_test_predicted))
    print("\n=====\nClassification Report\n=====")
    print(classification_report(y_test, y_test_predicted))
