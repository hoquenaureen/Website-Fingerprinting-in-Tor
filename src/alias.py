from typing import Tuple, TypeVar

from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# Represents the value returned by test_train_split
Split = Tuple[ndarray, ndarray, ndarray, ndarray]

# Represents one of our classifiers
Classifier = TypeVar(
    "Classifier", KNeighborsClassifier, LinearSVC, RandomForestClassifier
)
