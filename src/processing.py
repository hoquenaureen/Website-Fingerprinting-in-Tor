import io
import statistics as stats
from itertools import zip_longest
from typing import List, TypeVar, Tuple, Union
from zipfile import ZipFile

import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.alias import Split


def preprocess_data(url: str, split: float, quiet: bool = False) -> Split:
    if not quiet:
        print("Downloading dataset...")
    resp = requests.get(url)

    # Store the dataset temporarily in a list of lists - each sub-list will store the
    # label at index 0, the times at index 1, and the sizes at index 2.
    raw_data = []

    # Load all 90 instances for each of the 100 monitored websites, plus each instance
    # for the 9000 unmonitored websites
    if not quiet:
        print("Loading dataset...")
    with io.BytesIO(resp.content) as zipped_bytes:
        with ZipFile(zipped_bytes) as zf:
            # Load each file in the zipfile
            for name in tqdm(zf.namelist(), disable=quiet):
                # Load the times and sizes from the file
                times = []
                sizes = []
                with zf.open(name) as unzipped:
                    for line in unzipped.read().split(b"\n"):
                        try:
                            time, size = line.split(b"\t")
                        except ValueError:
                            continue

                        times.append(float(time))
                        sizes.append(int(size))

                # Skip the file if no data points were saved
                if len(times) == 0:
                    continue

                # Remove the "batch/" part of the name
                name = name[len("batch/") :]

                # Grab the label (which website it is) - we don't care about the
                # instance number
                if "-" not in name:
                    # The website is unmonitored
                    label = -1
                else:
                    # The website is monitored, and we want to know which one it is
                    label = int(name.split("-")[0])

                # Store the data
                raw_data.append([label, times, sizes])

    if not quiet:
        print("Processing dataset...")

    # Process the data into features
    data = []
    for data_point in tqdm(raw_data, disable=quiet):
        features = _generate_features(data_point[1], data_point[2])
        features.append(data_point[0])
        data.append(features)

    df = pd.DataFrame.from_records(data)

    # Drop useless data (columns that are all zero)
    # Credit to https://stackoverflow.com/a/16486950
    df = df.drop(df.columns[(df == 0).all()])

    # Separate data and labels
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    return train_test_split(X, y, test_size=split, random_state=0)


def apply_pca(dataset: Split, components: int) -> Split:
    # Unpack the dataset tuple
    X_train, X_test, y_train, y_test = dataset

    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Perform PCA
    pca = PCA(n_components=components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, y_train, y_test


def _generate_features(times: List[float], sizes: List[int]) -> List[Union[int, float]]:
    """Generates the features for a datapoint.

    The features generated by this function are those discussed by Jamie Hayes and
    George Danezis in section 5, "Feature selection" of their 2016 paper
    "K-fingerprinting: a robust scalable website fingerprinting technique."

    The features discussed by Tao Wang et. al. in section 4.2, "Feature set" of their
    2014 paper "Effective Attacks and Provable Defenses for Website Fingerprinting" are
    also generated. Note that some of the features were not implemented because they use
    the packet size, which has been removed from the dataset we are using (see `sizes`
    below for more information).

    Some additional features have been extracted from the dataset and added.

    :param times: A list of times when packets were sent.
    :type times: List[float]
    :param sizes: A list of packet "sizes" - note that all sizes have an absolute value
        of zero, so the term is pretty much just borrowed here. A positive size
        indicates the packet was outbound, while a negative size indicates the packet
        was inbound.
    :type sizes: List[int]
    :return: A list of feature values for the data point described by `times` and
        `sizes`.
    :rtype: List[Union[int, float]]
    """
    features: List[Union[int, float]] = []

    # Extract the lists of incoming and outgoing times
    incoming_times = []
    outgoing_times = []
    for time, size in zip(times, sizes):
        if size < 0:
            incoming_times.append(time)
        else:
            outgoing_times.append(time)

    # --- Number of packets statistics -------------------------------------------------

    # Total number of packets
    features.append(len(times))

    # Number of incoming packets
    features.append(len(incoming_times))

    # Number of outgoing packets (the remaining number of packets)
    features.append(len(outgoing_times))

    # --- Incoming and outgoing packets as fraction of total packets -------------------

    # Number of incoming packets as a fraction of the total
    features.append(len(incoming_times) / len(times))

    # Number of outgoing packets as a fraction of the total
    features.append(len(outgoing_times) / len(times))

    # --- Packet ordering statistics ---------------------------------------------------

    incoming_ordering = []
    outgoing_ordering = []
    for idx, val in enumerate(sizes):
        if val < 0:
            incoming_ordering.append(idx)
        else:
            outgoing_ordering.append(idx)

    # Incoming ordering mean
    features.append(stats.mean(incoming_ordering))

    # Outgoing ordering mean
    features.append(stats.mean(outgoing_ordering))

    # Incoming ordering standard deviation
    incoming_ordering = _zero_pad(incoming_ordering, 2)
    features.append(stats.stdev(incoming_ordering))

    # Outgoing ordering standard deviation
    outgoing_ordering = _zero_pad(outgoing_ordering, 2)
    features.append(stats.stdev(outgoing_ordering))

    # --- Concentration of outgoing packets --------------------------------------------

    concentrations = []
    for chunk in _chunks(sizes, 20, fillvalue=0):
        concentrations.append(len([x for x in chunk if x > 0]))

    # Concentration mean
    features.append(stats.mean(concentrations))

    # Concentration median
    features.append(stats.median(concentrations))

    # Concentration max
    features.append(max(concentrations))

    # Concentration standard deviation
    concentrations = _zero_pad(concentrations, 2)
    features.append(stats.stdev(concentrations))

    # --- Concentration of incoming and outgoing packets in first and last 30 packets --

    # Incoming packets in first 30
    features.append(len([x for x in sizes[:30] if x < 0]))

    # Outgoing packets in first 30
    features.append(len([x for x in sizes[:30] if x > 0]))

    # Incoming packets in last 30
    features.append(len([x for x in sizes[-30:] if x < 0]))

    # Outgoing packets in last 30
    features.append(len([x for x in sizes[-30:] if x > 0]))

    # --- Number of packets per second -------------------------------------------------

    # Number of packets per second
    features.append(times[-1] / len(times))

    # Mean of packet time
    features.append(stats.mean(times))

    # Standard devision of packet time
    features.append(stats.stdev(times))

    # Minimum of packet time
    features.append(min(times))

    # Maximum of packet time
    features.append(max(times))

    # Median of packet time
    features.append(stats.median(times))

    # --- Packet inter-arrival time statistics -----------------------------------------

    inter_times = _inter_arrival_times(times)
    incoming_inter_times = _inter_arrival_times(incoming_times)
    outgoing_inter_times = _inter_arrival_times(outgoing_times)

    # Max of inter-arrival times
    inter_times = _zero_pad(inter_times, 1)
    features.append(max(inter_times))
    incoming_inter_times = _zero_pad(incoming_inter_times, 1)
    features.append(max(incoming_inter_times))
    outgoing_inter_times = _zero_pad(outgoing_inter_times, 1)
    features.append(max(outgoing_inter_times))

    # Mean of inter-arrival times
    features.append(stats.mean(inter_times))
    features.append(stats.mean(incoming_inter_times))
    features.append(stats.mean(outgoing_inter_times))

    inter_times = _zero_pad(inter_times, 2)
    incoming_inter_times = _zero_pad(incoming_inter_times, 2)
    outgoing_inter_times = _zero_pad(outgoing_inter_times, 2)

    # Standard deviation of inter-arrival times
    features.append(stats.stdev(inter_times))
    features.append(stats.stdev(incoming_inter_times))
    features.append(stats.stdev(outgoing_inter_times))

    # Third quartile of inter-arrival times
    features.append(stats.quantiles(inter_times)[-1])
    features.append(stats.quantiles(incoming_inter_times)[-1])
    features.append(stats.quantiles(outgoing_inter_times)[-1])

    # --- Transmission time statistics -------------------------------------------------

    # Total transmission time
    features.append(times[-1] - times[0])
    features.append(incoming_times[-1] - incoming_times[0])
    features.append(outgoing_times[-1] - outgoing_times[0])

    # Transmission time quartiles (1st, 2nd, and 3rd quartiles)
    times = _zero_pad(times, 2)
    features.extend(stats.quantiles(times))
    incoming_times = _zero_pad(incoming_times, 2)
    features.extend(stats.quantiles(incoming_times))
    outgoing_times = _zero_pad(outgoing_times, 2)
    features.extend(stats.quantiles(outgoing_times))

    # --- Packet burst statistics ------------------------------------------------------

    for min_size in [2, 5, 10, 15]:
        incoming_bursts, outgoing_bursts = _find_bursts(sizes, min_size)

        # Number of bursts
        features.append(len(incoming_bursts))
        features.append(len(outgoing_bursts))

        incoming_bursts = _zero_pad(incoming_bursts, 1)
        outgoing_bursts = _zero_pad(outgoing_bursts, 1)

        # Mean burst length
        features.append(stats.mean(incoming_bursts))
        features.append(stats.mean(outgoing_bursts))

        # Maximum burst length
        features.append(max(incoming_bursts))
        features.append(max(outgoing_bursts))

    return features


def _inter_arrival_times(times: List[float]) -> List[float]:
    results = []

    for first, second in zip(times, times[1:]):
        results.append(second - first)

    return results


def _chunks(iterable, n, fillvalue=None):
    """Helper function to iterate over a list in chunks of a specific size.

    Implemented based on StackOverflow answer: https://stackoverflow.com/a/434411
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def _find_bursts(sizes: List[int], min_burst_size: int) -> Tuple[List[int], List[int]]:
    # bursts = []
    # burst_size = 0
    # for packet in sizes:
    # if packet > 0:
    # if burst_size >= min_burst_size:
    # bursts.append(burst_size)
    # burst_size = 0
    # else:
    # burst_size += 1

    incoming_bursts = []
    outgoing_bursts = []
    incoming_burst_size = 0
    outgoing_burst_size = 0
    for packet in sizes:
        if packet < 0:
            # Packet is incoming
            incoming_burst_size += 1

            if outgoing_burst_size >= min_burst_size:
                outgoing_bursts.append(outgoing_burst_size)

            outgoing_burst_size = 0

        else:
            # Packet is outgoing
            outgoing_burst_size += 1

            if incoming_burst_size >= min_burst_size:
                incoming_bursts.append(incoming_burst_size)

            incoming_burst_size = 0

    return incoming_bursts, outgoing_bursts


T = TypeVar("T", int, float)


def _zero_pad(to_pad: List[T], length: int) -> List[T]:
    while len(to_pad) < length:
        to_pad.append(0)

    return to_pad
