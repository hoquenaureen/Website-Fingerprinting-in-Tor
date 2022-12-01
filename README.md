Installation
------------

Package dependencies are recorded in `requirements.txt`. Run the following commands to create a new virtual environment and install the dependencies within that environment.

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Usage
-----

To train a model against the dataset and test its performance, run the following command:

```shell
python3 main.py MODEL
```

Where `MODEL` is one of the available models:

- `knn`: _k_ Nearest Neighbors
- `nn`: Multi-layer Perceptron
- `rf`: Random Forest
- `svm`: Support Vector Machine

There are a variety of command-line arguments available that can be used to configure the parameters of each of these models. However, their defaults have been set to the parameter values that were found to provide the best performance. If all you want to do is verify the claimed performance of the models, run the program without any additional parameters.

A grid search can be performed on any of the models - just add the `--grid-search` argument. The parameters used in the original grid searches are configured in the YAML files in the [`grid_search_params`](./grid_search_params/) folder. You can use these files as-is, or modify them to search over different parameters if you wish.

Caching
-------

A caching system has been implemented to reduce execution time. The first time a step has been performed, the results will be saved to disk. Every execution after that will load the results from disk instead of performing the computation of the step.

For example, the dataset will only be downloaded and processed the first time you run the code. Every time after that, the already-processed dataset will simply be loaded from disk.

This is important to note if you choose to try different parameters for a model using the command-line arguments. If you try different parameters for a model you've already run, make sure to delete its cached `.pkl` file under `--cache-path` before rerunning with the new parameters. Otherwise, the code will load the old model with the old parameters and you won't see a difference.

Note that this isn't an issue for grid searches - after loading the dataset, they ignore the cache completely.

The Dataset
-----------

The dataset is downloaded automatically from [Tao Wang's page on the HKUST Cybersecurity Lab website](http://home.cse.ust.hk/~taow/wf/data/). The script will automatically download the dataset for you - you don't have to download it yourself.

For more information about the dataset format, check out the first section on the [attacks page](http://home.cse.ust.hk/~taow/wf/attacks/), or read below for a description.

The dataset is made up of a series of plaintext files in a folder named `batch`. Each filename consists of either two numbers with a hyphen in between (e.g. `1-40`), or a single number (e.g. `4000`). Files with a hypen are "monitored" websites. Monitored websites are those to which traffic may be blocked and/or monitored by some governments. Files without a hyphen are unmonitored websites.

Filenames with a hypen are formatted as `X-Y`, where X is the number of the monitored website and Y identifies the unique datapoint of that website. Each monitored website has 90 unique datapoints.

The files are TSV (tab-separated values) files that describe a connection to the website. Each line represents one packet. The first column is the time the packet was sent, and the second column (called "packetsize") indicates if the packet was inbound or outbound. If the packet was inbound, packetsize is -1. If the packet was outbound, packetsize is +1.