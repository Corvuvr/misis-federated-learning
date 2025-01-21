import logging

import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


def load_data(data_sampling_percentage=0.5, client_id=1, total_clients=2):
    """Load federated dataset partition based on client ID.

    Args:
        data_sampling_percentage (float): Percentage of the dataset to use for training.
        client_id (int): Unique ID for the client.
        total_clients (int): Total number of clients.

    Returns:
        Tuple of arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    # Download and partition dataset
    fds = FederatedDataset(dataset="cifar100", partitioners={"train": total_clients})
    partition = fds.load_partition(client_id - 1, "train")
    partition.set_format("numpy")

    # Divide data on each client: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train, z_train = \
        partition["train"]["img"] / 255.0, \
        partition["train"]["fine_label"], \
        partition["train"]["coarse_label"]
    x_test, y_test, z_test = \
        partition["test"]["img"] / 255.0, \
        partition["test"]["fine_label"], \
        partition["test"]["coarse_label"]
    # Apply data sampling
    num_samples = int(data_sampling_percentage * len(x_train))
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train, y_train, z_train = x_train[indices], y_train[indices], z_train[indices]

    x_train = tf.image.resize(images=x_train, size=[x_train.shape[1]*2, x_train.shape[2]*2], preserve_aspect_ratio=True)
    x_test = tf.image.resize(images=x_test, size=[x_test.shape[1]*2, x_test.shape[2]*2], preserve_aspect_ratio=True)
    
    return (x_train, y_train, z_train), (x_test, y_test, z_test)
