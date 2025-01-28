import io
import logging
import zipfile
import requests
import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

SERVER_LOCAL_IP = "172.21.0.5"

def load_data(data_sampling_percentage=0.5, client_id=1, total_clients=2):
    
    # Request data
    buffer = requests.get(f'http://{SERVER_LOCAL_IP}:7272/load_dataset', params={"client_id": client_id}).content
    
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
    print(f'{images.shape=}')

    # Get labels
    fine_labels = np.frombuffer(
        zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{2}'),
        dtype='int16'
    )
    coarse_labels = np.frombuffer(
        zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{3}'),
        dtype='int16'
    )

    # Split data: 80% train, 20% test
    split_factor: float = 0.8
    split_point: int = int(len(images) * split_factor)
    (x_test, y_test, z_test) = images[split_point:], fine_labels[split_point:], coarse_labels[split_point:]
    (x_train, y_train, z_train) = images[:split_point], fine_labels[:split_point], coarse_labels[:split_point]
    
    print(f'{x_test.shape=} {x_test.dtype=}\n{y_test.shape=} {z_test.shape=}')
    print(f'{x_train.shape=} {x_train.dtype=}\n{y_train.shape=} {z_train.shape=}')

    return (x_train, y_train, z_train), (x_test, y_test, z_test)