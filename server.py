# Standard imports
import zipfile
import logging
import threading
import numpy as np
from typing import Sequence, Iterable, Generator

# Federated stuff
import flwr as fl
from flask import Flask, request, send_file
from prometheus_client import Gauge, start_http_server

# Local imports
from dataset import get_full_dataset, get_split_partition, get_label_banks
from strategy.strategy  import FedCustom
from helpers.plots      import updatePlot
from helpers.load_data  import make_json
from helpers.server_args   import args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================== FLASK DATASET SENDER ===============================
dataset_sender = Flask(__name__)
@dataset_sender.route('/establish_connection')
def establish_connection():
    return "Hello"
@dataset_sender.route('/load_dataset')
def load_data_proxy():
    global args
    global dataset
    global label_banks  
    client_id = int(request.args.get(key="client_id"))  
    data: dict = get_split_partition(dataset=dataset, label_banks=label_banks, split_type=args.split_type, client_id=client_id)
    flat: tuple[np.ndarray] = (
        # Train
        np.array(data["train"]["img"].shape).astype('int16'  ),
        data["train"]["img"].flatten()      .astype('float32'),
        data["train"]["fine_label"  ]       .astype('int16'  ), 
        data["train"]["coarse_label"]       .astype('int16'  ),
        # Test
        np.array(data["test"]["img"].shape).astype('int16'  ),
        data["test"]["img"].flatten()      .astype('float32'),
        data["test"]["fine_label"  ]       .astype('int16'  ), 
        data["test"]["coarse_label"]       .astype('int16'  ),
    )
    # Save data to zip
    workdir = "/data/"
    zip_filename = f"~data{client_id}.zip"
    zip_filepath = f"{workdir}{zip_filename}"
    zip = zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED)
    for i, el in enumerate(flat):
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
    logger.info(f"Running flask at: {host}:{port}")
    dataset_sender.run(host=host, port=port)

# =============================== FEDERATED LEARNING ===============================
def start_fl_server(strategy, rounds):
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy
        )
    except Exception as e:
        logger.error(f"FL Server error: {e}", exc_info=True)

def legacy(b: bool):
    def wrapper_(function):
        def wrapper(*rgs, **kwargs):
            if b:
                # Load Dataset
                global dataset
                global label_banks
                dataset = get_full_dataset(train_split=0.8)
                label_banks = get_label_banks(dataset=dataset, split_type=args.split_type, total_clients=args.total_clients)
                # Start Flask Dataset Server
                lock = threading.Lock()
                thread = threading.Thread(target=run_flask_server, args=(args.flask_address, 7272))
                thread.start()
                # Main
                function(*rgs, **kwargs)
                # End
                thread.join()
            else:
                function(*rgs, **kwargs)
            return
        return wrapper
    return wrapper_

@legacy(args.legacy)
def main():
    global args
    # Define gauges to track the accuracy and loss of the global model
    accuracy_gauge = Gauge("model_accuracy", "Current accuracy of the global model")
    loss_gauge = Gauge("model_loss", "Current loss of the global model")
    # Initialize Strategy Instance 
    strategy_instance = FedCustom(accuracy_gauge=accuracy_gauge, loss_gauge=loss_gauge)
    # Start servers
    start_http_server(8000) # Prometheus Metrics Server
    start_fl_server(strategy=strategy_instance, rounds=args.number_of_rounds) # Flower Federated Learning Server
    # Derive metrics
    updatePlot(data_path='/results', num_clients=args.total_clients) 
        
# Main Function
if __name__ == "__main__":
    make_json('/results/server.json', args.__dict__)
    main()
    raise Exception("Finished!")