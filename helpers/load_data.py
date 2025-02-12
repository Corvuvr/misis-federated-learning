import io
import logging
import zipfile
import requests
import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

def shuffle(percentage, args):
    it = iter(args)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('ERROR: lists are not of the same length!')
    # Apply data sampling
    num_samples = int(percentage * len(args[0]))
    indices = np.random.choice(len(args[0]), num_samples, replace=False)
    for arg in args:
        yield arg[indices]

def load_data_local(train_split: float = 0.5, scale_factor: int = 1):
    
    # Download dataset
    fds = FederatedDataset(dataset="cifar100", partitioners={"train": 1})
    data = fds.load_partition(0, "train")
    data.set_format("numpy")

    # Divide data on each client:
    partition = data.train_test_split(test_size=1-train_split, seed=42)
    x_train, y_train, z_train = \
        partition["train"]["img"] / 255.0, \
        partition["train"]["fine_label"], \
        partition["train"]["coarse_label"]
    x_test, y_test, z_test = \
        partition["test"]["img"] / 255.0, \
        partition["test"]["fine_label"], \
        partition["test"]["coarse_label"]
    # Scale
    x_train, x_test = scale_input(scale=scale_factor, args=(x_train, x_test))
    return (x_train, y_train, z_train), (x_test, y_test, z_test)

def load_data(client_id: int, train_split: float = 0.5, scale_factor: int = 1, server_ip: str = "0.0.0.0"):
    
    # Request data
    buffer = requests.get(f'http://{server_ip}:7272/load_dataset', params={"client_id": client_id}).content
    
    print(f'{zipfile.ZipFile(io.BytesIO(buffer)).infolist()=}')
    
    # Get image metadata
    meta = np.frombuffer(
        zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{0}'),
        dtype='int16'
    )

    # Get images
    images = np.frombuffer(
        zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{1}'),
        dtype='float32'
    )
    images = np.reshape(images, newshape=meta)

    # Get labels
    fine_labels = np.frombuffer(
        zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{2}'),
        dtype='int16'
    )
    coarse_labels = np.frombuffer(
        zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{3}'),
        dtype='int16'
    )
 
    # Shuffle data
    (images, fine_labels, coarse_labels) = shuffle(args=(images, fine_labels, coarse_labels), percentage=1.0)
    # Split data: train=train_split, test=1-train_split
    split_point: int = int(len(images) * train_split)
    (x_train, y_train, z_train) = images[split_point:], fine_labels[split_point:], coarse_labels[split_point:]
    (x_test, y_test, z_test) = images[:split_point], fine_labels[:split_point], coarse_labels[:split_point]

    # Scale
    x_train, x_test = scale_input(scale=scale_factor, args=(x_train, x_test))

    return (x_train, y_train, z_train), (x_test, y_test, z_test)

def scale_input(args, scale: int = 1):
    for arg in args:
        yield np.resize(
            arg, 
            (arg.shape[0], arg.shape[1]*scale, arg.shape[2]*scale, arg.shape[3])
        )

class db:
    count = 0
    def __init__(self):
        db.count += 1
        print(f'Debug point:\t{db.count}')