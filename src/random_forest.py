import time

from sklearn.ensemble import RandomForestClassifier

from src.alias import Split


def rf_fit(dataset: Split, trees: int, quiet: bool = False):
    # Unpack the dataset tuple
    X_train, X_test, y_train, y_test = dataset

    if not quiet:
        print("Fitting Random Forest classifier...")

    # Start the timer
    start = time.time()

    # Train the model
    rf = RandomForestClassifier(n_estimators=trees)
    rf.fit(X_train, y_train)

    end = time.time()

    return rf, end - start


def get_classifier():
    return RandomForestClassifier()
