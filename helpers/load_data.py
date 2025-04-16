import io
import json
import logging
import zipfile
import requests

import numpy as np
from flwr_datasets import FederatedDataset

from dataset import *

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# ========================================== UTILS ============================================
class db:
    count = 0
    def __init__(self):
        db.count += 1
        print(f'Debug point:\t{db.count}')

def get_coverage(ev: set, tr: set) -> float:
    u = ev.union(tr)
    coverage: float = len(u) / len(ev)
    return coverage

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

# ========================================== LOADERS ==========================================
def load_data_local(train_split: float = 0.5):
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

def load_data_legacy(client_id: int, server_ip: str = "0.0.0.0"):
    # Request data
    buffer = requests.get(f'http://{server_ip}:7272/load_dataset', params={"client_id": client_id}).content
    print(f'{zipfile.ZipFile(io.BytesIO(buffer)).infolist()=}')
    # Get Train image metadata
    train_meta = np.frombuffer(zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{0}'),dtype='int16')
    train_images = np.frombuffer(zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{1}'),dtype='float32')
    train_images = np.reshape(train_images, newshape=train_meta)
    train_fine_labels = np.frombuffer(zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{2}'),dtype='int16')
    train_coarse_labels = np.frombuffer(zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{3}'),dtype='int16')
    # Get Test image metadata
    test_meta = np.frombuffer(zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{4}'),dtype='int16')
    test_images = np.frombuffer(zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{5}'),dtype='float32')
    test_images = np.reshape(test_images, newshape=test_meta)
    test_fine_labels = np.frombuffer(zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{6}'),dtype='int16')
    test_coarse_labels = np.frombuffer(zipfile.ZipFile(io.BytesIO(buffer)).read(f'~tmp-{client_id}-{7}'),dtype='int16')
    return (train_images, train_fine_labels, train_coarse_labels), (test_images, test_fine_labels, test_coarse_labels)

def load_data_fed(client_id: int, total_clients: int, train_split: float = 0.8, split_type: str = "coarse"):
    partition = classic_scenario(
        split_type=split_type, 
        total_clients=total_clients,
        client_id=client_id,
        train_split=train_split
    )
    x_train = partition["train"]["img"         ]
    y_train = partition["train"]["fine_label"  ] 
    z_train = partition["train"]["coarse_label"]
    x_test  = partition["test" ]["img"         ]
    y_test  = partition["test" ]["fine_label"  ] 
    z_test  = partition["test" ]["coarse_label"]
    return (x_train, y_train, z_train), (x_test, y_test, z_test)

def load_data(
    mode: str = "local",
    split_type: str = "coarse",
    client_id: int = 1, 
    total_clients: int = 2, 
    train_split: float = 0.8, 
    server_ip: str = "0.0.0.0"
    ):
    match mode:
        case "local":
            return load_data_local(train_split)
        case "fed":
            return load_data_fed(client_id, total_clients, train_split, split_type)
        case "legacy":
            return load_data_legacy(client_id, server_ip)
        case _:
            raise Exception(f"Mode: {mode} - no such option! Modes availible: [ local | fed | legacy ]")

# ========================================== JSON ==========================================
def make_json(path: str, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([data], f, ensure_ascii=False, indent=4)
def push_json(path: str, data):
        with open(path, 'r', encoding='utf-8') as f:
            df = json.load(f)
            df.append(data)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(df, f, ensure_ascii=False, indent=4)