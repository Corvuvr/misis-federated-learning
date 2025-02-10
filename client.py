import os
import time
import json
import logging
import argparse
import requests

import flwr as fl
import tensorflow as tf

from model.model import Model
from helpers.plots import updatePlot
from helpers.load_data import load_data, load_data_local, shuffle

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")

# Common args
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.1, help="Learning rate for the optimizer"
)
parser.add_argument(
    "--data-percentage", type=float, default=0.5, help="Portion of client data to use"
)
parser.add_argument(
    "--epochs-per-subset", type=int, default=10, help="Epochs in each iterations"
)
parser.add_argument(
    "--total-epochs", type=int, default=500, help="Total epochs"
)
parser.add_argument(
    "--scale", type=int, default=1, help="Scale the input of the model"
)
parser.add_argument(
    "--alpha", type=float, default=1.0, help="Parameters of the model"
)
parser.add_argument(
    "--train-split", type=float, default=0.5, help="In which proportion split partition"
)

# Fed args
parser.add_argument(
    "--server_address", type=str, default="server:8080", help="Address of the fed server"
)
parser.add_argument(
    "--flask_address", type=str, default="0.0.0.0", help="Address of the data server"
)
parser.add_argument(
    "--client_id", type=int, default=1, help="Unique ID for the client"
)
parser.add_argument(
    "--total_clients", type=int, default=2, help="Total number of clients"
)

# Local args
parser.add_argument(
    "--local", default=False, action='store_true', help="Whether to launch locally or as a federated client"
)

args = parser.parse_args()
print(f'{args=}')

# Globals
LOCAL_LEARNING = args.local
COARSE_LEARNING = False
CLIENT_FOLDER = "results" if LOCAL_LEARNING else "/results"
CLIENT_PREFIX: str = str(args.client_id) if not LOCAL_LEARNING else "-solo"
JSON_PATH = f'{CLIENT_FOLDER}/client{CLIENT_PREFIX}.json'
WEIGHTS_PATH = f'{CLIENT_FOLDER}/client{CLIENT_PREFIX}.weights.h5'

if not LOCAL_LEARNING:
    wait_for_server: bool = True
    while wait_for_server:
        time.sleep(1)
        try:
            if requests.get(f'http://{args.flask_address}:7272/establish_connection'):
                wait_for_server = False
        except:
            print("Waiting for server...")
    print(f"Connection established. Initializing client{CLIENT_PREFIX}...")

# Logs
logging.basicConfig(level=logging.INFO)     # Configure logging
logger = logging.getLogger(__name__)        # Create logger for the module
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Make TensorFlow log less verbose

class Client(fl.client.NumPyClient):
    def __init__(self, args):
        super().__init__()
        self.args = args

        logger.info("Preparing data...")
        (x_train, y_train, z_train), (x_test, y_test, z_test) = load_data(
            client_id=self.args.client_id,
            train_split = args.train_split, 
            scale_factor = args.scale, 
            server_ip=args.flask_address
        ) if not LOCAL_LEARNING else load_data_local(
            train_split=self.args.data_percentage,
            scale_factor=self.args.scale
        )

        self.x_train = x_train # img
        self.y_train = y_train # fine_label - 100 classes
        self.z_train = z_train # coarse label - 20 superclasses
        self.x_test = x_test
        self.y_test = y_test
        self.z_test = z_test

        self.train_images = x_train
        self.train_labels = z_train if COARSE_LEARNING else y_train

        print(f'{self.train_images.shape=}')
        
        self.test_images = x_test
        self.test_labels = z_test if COARSE_LEARNING else y_test

    def get_parameters(self, config=None):
        # Return the parameters of the model
        return model.get_model().get_weights()

    def fit(self, parameters, config=None):
        global epochs
        global args
        global train_accuracy
        # Set the weights of the model
        model.get_model().set_weights(parameters)

        # Get training subset
        train_images, train_labels = shuffle(
            percentage=args.data_percentage,
            args=(self.train_images, self.train_labels)
        )

        # Add a callback that saves weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=WEIGHTS_PATH,
            save_weights_only=True,
            verbose=1
        )

        # Train the model
        history = model.get_model().fit(
            train_images,
            train_labels, 
            batch_size=self.args.batch_size, 
            callbacks=[cp_callback], 
            epochs=args.epochs_per_subset
        )
        epochs += args.epochs_per_subset
        
        # Calculate evaluation metric
        train_accuracy = float(history.history["accuracy"][-1])
        results = { "accuracy": train_accuracy, }

        # Get the parameters after training
        parameters_prime = model.get_model().get_weights()

        # Directly return the parameters and the number of examples trained on
        return parameters_prime, len(self.x_train), results


    def evaluate(self, parameters, config=None):
        
        global epochs
        # Set the weights of the model
        model.get_model().set_weights(parameters)
        # Evaluate the model and get the loss and accuracy
        loss, eval_accuracy = model.get_model().evaluate(
            self.test_images, self.test_labels, batch_size=self.args.batch_size
        )

        diagnostic_data = [epochs, loss, eval_accuracy, train_accuracy]
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            df = json.load(f)
            df.append(diagnostic_data)
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(df, f, ensure_ascii=False, indent=4)

        # Return the loss, the number of examples evaluated on and the accuracy
        return float(loss), len(self.x_test), {"accuracy": float(eval_accuracy)}


# Function to Start the Client
def start_fl_client():
    try:
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump([args.__dict__], f, ensure_ascii=False, indent=4)
        client = Client(args).to_client()
        fl.client.start_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}
    

if __name__ == "__main__":

    # Model
    num_classes = 20 if COARSE_LEARNING else 100
    model = Model(
        learning_rate=args.learning_rate, 
        classes_=num_classes,
        alpha_=args.alpha,
        scale_input=args.scale
    )
    model.compile()
    epochs = 0
    # Learning
    if LOCAL_LEARNING:
        with open(f'{CLIENT_FOLDER}/client{CLIENT_PREFIX}.json', 'w', encoding='utf-8') as f:
            json.dump([args.__dict__], f, ensure_ascii=False, indent=4)
        c = Client(args)
        params = c.get_parameters()
        
        while epochs <= args.total_epochs:
            print(f"Epoch {epochs}...")
            params, num_examples, results = c.fit(params)
            epochs += args.epochs_per_subset
            c.evaluate(params)
            updatePlot(mode="solo", data_path=JSON_PATH)
    else:
        start_fl_client()