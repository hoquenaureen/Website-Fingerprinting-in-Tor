import time
from typing import Tuple

from sklearn.neural_network import MLPClassifier

from src.alias import Split


def nn_fit(
    dataset: Split,
    hidden_layer_sizes: Tuple[int, int],
    solver: str,
    alpha: float,
    activation: str,
    tol: float,
    learning_rate: str,
    quiet: bool = False,
) -> Tuple[MLPClassifier, float]:
    # Unpack the dataset tuple
    X_train, X_test, y_train, y_test = dataset

    if not quiet:
        print("Fitting MLP classifier...")

    # Start the timer
    start = time.time()

    # Train the model
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        solver=solver,
        alpha=alpha,
        activation=activation,
        tol=tol,
        learning_rate=learning_rate,
    )
    mlp_clf.fit(X_train, y_train)

    # Stop the timer
    end = time.time()

    return mlp_clf, end - start


def get_classifier():
    return MLPClassifier()
