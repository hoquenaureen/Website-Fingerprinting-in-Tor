import time

from sklearn.neighbors import KNeighborsClassifier

from src.alias import Split


def knn_fit(dataset: Split, n: int, p: int, weights: str, quiet: bool = False):
    # Unpack the dataset tuple
    X_train, X_test, y_train, y_test = dataset

    if not quiet:
        print("Fitting K-NN classifier...")

    # Start the timer
    start = time.time()

    # Train the model
    knn = KNeighborsClassifier(n_neighbors=n, p=p, weights=weights)
    knn.fit(X_train, y_train)

    # End the timer
    end = time.time()

    return knn, end - start


def get_classifier():
    return KNeighborsClassifier()
