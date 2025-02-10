import argparse
import logging
import time
import threading
import logging
import requests
import zipfile

import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset

import flwr as fl

from prometheus_client import Gauge, start_http_server
from strategy.strategy import FedCustom

from flask import Flask, request, send_file

RESIZE_DATA: bool = False

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a gauge to track the global model accuracy
accuracy_gauge = Gauge("model_accuracy", "Current accuracy of the global model")

# Define a gauge to track the global model loss
loss_gauge = Gauge("model_loss", "Current loss of the global model")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower Server")
parser.add_argument(
    "--number_of_rounds",
    type=int,
    default=100,
    help="Number of FL rounds (default: 100)",
)
parser.add_argument(
    "--flask_address", type=str, default="0.0.0.0", help="Address of the data server"
)
args = parser.parse_args()

# Function to Start Federated Learning Server
def start_fl_server(strategy, rounds):
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy
        )
    except Exception as e:
        logger.error(f"FL Server error: {e}", exc_info=True)

def split_list_by_step(input: list, step: int):
    if step <= 0:
        return None
    j = 0
    while j < len(input):
        partition: list = []
        for i in range(len(input)):
            try:
                partition.append(input[j+i])
            except:
                j += i + 1
                break
            if i == step - 1:
                j += i + 1
                break
        yield partition

def get_indicies_of_classes(data, classes: list[str]):
    for i in range(len(data)):
        if data[i] in classes:
            yield i

# Dataset sender
total_clients = 0

dataset_sender = Flask(__name__)

@dataset_sender.route('/establish_connection')
def establish_connection():
    #print("omg hiiii!")
    return "Hello"

@dataset_sender.route('/load_dataset')
def load_data_proxy():
    
    # Get node info
    client_id = int(request.args.get(key="client_id"))
    print(f"{request.args=}")
    
    # Get preloaded data
    global images
    global fine_labels 
    global coarse_labels

    # Get names of the classes that the specific client will learn
    global total_clients
    global data
    global all_classes
    class_partition = list(split_list_by_step(
        input=all_classes,
        step=np.ceil(len(all_classes)/total_clients)
    ))[client_id-1]
    print(f'Client {client_id} will learn these classes: {class_partition}')
    
    # Get ids of the dataset part which has the mentioned classes
    train_partition_indicies: list[int] = list(get_indicies_of_classes(
        data=fine_labels, classes=class_partition
    ))
    
    # Partition data
    partition_images = np.array(images[train_partition_indicies])
    partition_fine_labels = np.array(fine_labels[train_partition_indicies]).astype('int16')
    partition_coarse_labels = np.array(coarse_labels[train_partition_indicies]).astype('int16')
    partition_image_metadata = np.array(partition_images.shape).astype('int16')
    
    partition_images = partition_images.flatten().astype('float32')

    # Save data to zip
    data = (partition_image_metadata, partition_images, partition_fine_labels, partition_coarse_labels)
    workdir = "/workdir/"
    zip_filename = f"~data{client_id}.zip"
    zip_filepath = f"{workdir}{zip_filename}"
    zip = zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED)
    for i, el in enumerate(data):
        tmp_filename = f"~tmp-{client_id}-{i}"
        tmp_filepath = f"{workdir}{tmp_filename}"
        el.tofile(tmp_filepath)
        zip.write(tmp_filepath, arcname=tmp_filename)
    zip.close()

    return send_file(
        path_or_file=zip_filepath,
        download_name=zip_filename,
    )

def run_flask_server(host='172.19.0.5', port=7272):
    dataset_sender.run(host=args.flask_address, port=port)

# Main Function
if __name__ == "__main__":

    # Initialize Strategy Instance 
    strategy_instance = FedCustom(accuracy_gauge=accuracy_gauge, loss_gauge=loss_gauge)
    total_clients = 3

    # Download dataset
    fds = FederatedDataset(dataset="cifar100", partitioners={"train": total_clients})
    data = fds.load_partition(0, "train")
    data.set_format("numpy")
    all_classes: list[str] = [i for i, el in enumerate(data.info.features['fine_label'].names)]
    
    # Prepare data to be sent
    images = np.array(data["img"]).astype(dtype='float32') / 255.0
    fine_labels = np.array(data["fine_label"])
    coarse_labels = np.array(data["coarse_label"])

    # Start Prometheus Metrics Server
    start_http_server(8000)
    
    # Start Dataset Server 
    lock = threading.Lock()
    thread = threading.Thread(target=run_flask_server, args=('0.0.0.0', 7272))
    thread.start()

    # Start FL Server
    start_fl_server(strategy=strategy_instance, rounds=args.number_of_rounds)
    print(f'{total_clients=}')

    thread.join()