import io
import logging
import zipfile
import requests
import datasets

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
    clients = 2
    fds = FederatedDataset(dataset="cifar100", partitioners={"train": clients})

    column_names = ["train", "test"]

    x_train = y_train = z_train = \
        x_test = y_test = z_test = np.array([])
    
    for i in range(clients):

            data: dict = fds.load_partition(i, "train")
            data.set_format("numpy")
            partition = data.train_test_split(test_size=1-train_split, seed=42)
            
            for name in column_names:
                x_tmp = partition[name]["img"] / 255.0
                y_tmp = partition[name]["fine_label"] 
                z_tmp = partition[name]["coarse_label"]
      
                if name == column_names[0]:
                    if x_train.any():
                        x_train = np.concatenate((x_tmp,  x_train), axis=0)
                        y_train = np.concatenate((y_tmp,  y_train), axis=0)
                        z_train = np.concatenate((z_tmp,  z_train), axis=0)
                    else:
                        x_train, y_train, z_train = x_tmp, y_tmp, z_tmp
                elif name == column_names[1]:
                    if x_test.any():
                        x_test = np.concatenate((x_tmp,  x_test), axis=0)
                        y_test = np.concatenate((y_tmp,  y_test), axis=0)
                        z_test = np.concatenate((z_tmp,  z_test), axis=0)
                    else:
                        x_test, y_test, z_test = x_tmp, y_tmp, z_tmp

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


    return (x_train, y_train, z_train), (x_test, y_test, z_test)

def scale_input(args, scale: int = 1):
    for arg in args:
        yield np.array(arg).repeat(scale, axis=1).repeat(scale, axis=2)

class db:
    count = 0
    def __init__(self):
        db.count += 1
        print(f'Debug point:\t{db.count}')