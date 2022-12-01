import argparse
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV

from src import cache, knn, nn, processing, random_forest, results, svm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General arguments
    parser.add_argument(
        "model",
        choices=["none", "knn", "nn", "rf", "svm"],
        help=(
            "Which model to use to train and classify the data. If 'none', the program "
            "will simply perform the data loading and preprocessing step and then "
            "exit."
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Disables TQDM progress bars and other progress-related output.",
    )

    # Grid search
    search = parser.add_argument_group("Grid Search")
    search.add_argument(
        "--grid-search",
        action="store_true",
        help=(
            "If passed, perform a grid search on the specified model. Any command-line "
            "arguments used to specify model parameters will be ignored."
        ),
    )
    search.add_argument(
        "--grid-search-path",
        help=(
            "The path to look under for grid search parameters. This folder should "
            "contain YAML files that define the parameters to be used for a grid "
            "search. Each file should be named MODEL.yml, where MODEL is the name of "
            "the model being run. For example, the k-NN model loads parameters from "
            "knn.yml."
        ),
    )
    search.add_argument(
        "--grid-search-results",
        help=(
            "If specified, this is the path to save grid search results to in CSV "
            "format. This file will be overwritten if it already exists. By default, "
            "grid search results are saved under the path specified in "
            "--grid-search-path as a file named MODEL.csv."
        ),
    )
    search.add_argument(
        "--grid-search-jobs",
        type=int,
        default=-1,
        help=(
            "The number of jobs to run in parallel when performing a grid search. To "
            "use all CPU cores, pass -1 (the default)."
        ),
    )

    # Data loading and processing
    data = parser.add_argument_group("Data loading and preprocessing")
    data.add_argument(
        "--split",
        type=float,
        default=0.3,
        help=(
            "The proportion of data to use for testing. The rest of the data will be "
            "used for training."
        ),
    )
    data.add_argument(
        "--cache-path",
        default="./cache",
        help=(
            "The folder that will be used to store picklefiles containing cached "
            "results of each of the processing stages, such as data preprocessing, "
            "model training, optimization, and results. If this directory does not "
            "exist, it will be created."
        ),
    )
    data.add_argument(
        "--data-url",
        default="http://home.cse.ust.hk/~taow/wf/data/knndata.zip",
        help=(
            "The direct download link for the dataset zipfile. This should only be "
            "modified if the dataset has been moved to a new website."
        ),
    )
    data.add_argument(
        "--num-components",
        type=int,
        default=40,
        help="The number of PCA components to use as inputs to the models.",
    )

    # k-NN parameters
    knn = parser.add_argument_group("k-NN parameters")
    knn.add_argument(
        "--n-neighbors", type=int, default=1, help="The number of neighbors to query."
    )
    knn.add_argument(
        "--power",
        type=int,
        default=2,
        help="The power parameter for the Minkowski metric.",
    )
    knn.add_argument(
        "--weights",
        default="uniform",
        choices=["uniform", "distance"],
        help="The weight function used in prediction.",
    )

    # Neural network parameters
    nn = parser.add_argument_group("Neural network parameters")
    nn.add_argument(
        "--hidden-layer-sizes",
        type=int,
        nargs=2,
        default=[150, 20],
        help="The shape of the hidden layers to use.",
    )
    nn.add_argument(
        "--solver",
        choices=["sdg", "lbfgs", "adam"],
        default="lbfgs",
        help="The solver for weight optimization",
    )
    nn.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="The regularization parameter.",
    )
    nn.add_argument(
        "--activation",
        choices=["identity", "logistic", "tanh", "relu"],
        default="relu",
        help="The activation function for the hidden layer(s).",
    )
    nn.add_argument(
        "--tol",
        type=float,
        default=0.0001,
        help="The optimization tolerance.",
    )
    nn.add_argument(
        "--learning-rate",
        choices=["adaptive", "constant"],
        default="adaptive",
        help=(
            "Whether to use a constant learning rate or to adapt the learning rate as "
            "training continues."
        ),
    )

    # Random forest parameters
    rf = parser.add_argument_group("Random forest parameters")
    rf.add_argument(
        "--trees",
        type=int,
        default=250,
        help="The number of trees in the forest.",
    )

    # SVM parameters
    svm = parser.add_argument_group("SVM parameters")
    svm.add_argument(
        "--regularization",
        type=int,
        default=300,
        help="The regularization parameter. Must be positive.",
    )
    svm.add_argument(
        "--kernel",
        default="rbf",
        choices=["linear", "poly", "rbf", "sigmoid", "precomputed"],
        help="The kernel type to use in the SVM algorithm.",
    )
    svm.add_argument(
        "--kernel-coefficient",
        type=float,
        default=0.005,
        help=(
            "The kernel coefficient. Used when the kernel is 'rbf', 'poly', or "
            "'sigmoid'."
        ),
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    cacher = cache.Cache(args.cache_path, quiet=args.quiet)

    # Load the dataset
    dataset = cacher.run(
        processing.preprocess_data, args.data_url, args.split, quiet=args.quiet
    )

    # Apply PCA
    dataset_pca = cacher.run(processing.apply_pca, dataset, args.num_components)

    if args.model == "none":
        if not args.quiet:
            print("No model was selected. Exiting...")
        sys.exit(0)

    if args.grid_search:
        # Load parameters from file
        with open(f"{args.grid_search_path}/{args.model}.yml") as param_file:
            params = yaml.load(param_file, Loader=yaml.FullLoader)

        # Initialize the grid search
        if args.model == "knn":
            classifier = knn.get_classifier()
            X_train, X_test, y_train, y_test = dataset_pca
        elif args.model == "svm":
            classifier = svm.get_classifier()
            X_train, X_test, y_train, y_test = dataset_pca
        elif args.model == "rf":
            classifier = random_forest.get_classifier()
            X_train, X_test, y_train, y_test = dataset
        elif args.model == "nn":
            classifier = nn.get_classifier()
            X_train, X_test, y_train, y_test = dataset_pca
        else:
            print("Invalid model type was specified")
            exit(1)

        search = GridSearchCV(
            classifier, params, verbose=10, n_jobs=args.grid_search_jobs
        )

        # Since the grid search performs cross-validation, we don't want to use the
        # train/test split.
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        # Run the search and save the results
        search.fit(X, y)
        search_results = pd.DataFrame(search.cv_results_)
        if args.grid_search_results:
            search_results.to_csv(args.grid_search_results, index=False)
        else:
            search_results.to_csv(f"{args.grid_search_path}/{args.model}.csv")

    else:
        # Run the model
        if args.model == "knn":
            model, train_time = cacher.run(
                knn.knn_fit,
                dataset_pca,
                args.n_neighbors,
                args.power,
                args.weights,
                quiet=args.quiet,
            )
            results.print_results(model, train_time, dataset_pca)
        elif args.model == "svm":
            model, train_time = cacher.run(
                svm.svm_fit,
                dataset_pca,
                args.regularization,
                args.kernel_coefficient,
                args.kernel,
                quiet=args.quiet,
            )
            results.print_results(model, train_time, dataset_pca)
        elif args.model == "rf":
            model, train_time = cacher.run(
                random_forest.rf_fit, dataset, args.trees, quiet=args.quiet
            )
            results.print_results(model, train_time, dataset)
        elif args.model == "nn":
            model, train_time = cacher.run(
                nn.nn_fit,
                dataset_pca,
                args.hidden_layer_sizes,
                args.solver,
                args.alpha,
                args.activation,
                args.tol,
                args.learning_rate,
                quiet=args.quiet,
            )
            results.print_results(model, train_time, dataset_pca)
        else:
            print("Invalid model type was specified")
            exit(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
