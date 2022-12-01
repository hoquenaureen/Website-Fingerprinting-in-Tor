import time

from sklearn.svm import SVC

from src.alias import Split

# Chosen by fair dice roll. Guaranteed (sic) to be random.
# https://xkcd.com/221/
RANDOM_STATE = 4


def svm_fit(dataset: Split, c: int, gamma: float, kernel: str, quiet: bool = False):
    # Unpack the dataset tuple
    X_train, X_test, y_train, y_test = dataset

    if not quiet:
        print("Fitting SVM classifier...")

    # Start the timer
    start = time.time()

    # Train the model
    svm = SVC(kernel=kernel, C=c, gamma=gamma, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)

    end = time.time()

    return svm, end - start


def get_classifier():
    return SVC()
