import time
import json
import argparse
import logging
import os

import flwr as fl
import tensorflow as tf

from model.model import Model
from helpers.load_data import load_data

load_data()

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")

parser.add_argument(
    "--server_address", type=str, default="server:8080", help="Address of the server"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.1, help="Learning rate for the optimizer"
)
parser.add_argument("--client_id", type=int, default=1, help="Unique ID for the client")
parser.add_argument(
    "--total_clients", type=int, default=2, help="Total number of clients"
)
parser.add_argument(
    "--data_percentage", type=float, default=0.5, help="Portion of client data to use"
)

args = parser.parse_args()

checkpoint_path = "/results/client.weights.h5"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

COARSE_LEARNING = True
num_classes = 20 if COARSE_LEARNING else 100

# Create an instance of the model and pass the learning rate as an argument
model = Model(learning_rate=args.learning_rate, classes_=num_classes)

# Compile the model
model.compile()

start: float = time.time()
class Client(fl.client.NumPyClient):
    def __init__(self, args):
        super().__init__()
        self.args = args

        logger.info("Preparing data...")
        (x_train, y_train, z_train), (x_test, y_test, z_test) = load_data(
            data_sampling_percentage=self.args.data_percentage,
            client_id=self.args.client_id,
            total_clients=self.args.total_clients,
        )

        self.x_train = x_train # img
        self.y_train = y_train # fine_label - 100 classes
        self.z_train = z_train # coarse label - 20 superclasses
        self.x_test = x_test
        self.y_test = y_test
        self.z_test = z_test

        self.train_images = x_train
        self.train_labels = z_train if COARSE_LEARNING else y_train
        
        self.test_images = x_test
        self.test_labels = z_test if COARSE_LEARNING else y_test

    def get_parameters(self, config):
        # Return the parameters of the model
        return model.get_model().get_weights()

    def fit(self, parameters, config):
        # Set the weights of the model
        model.get_model().set_weights(parameters)

        global start
        # Train the model
        history = model.get_model().fit(
            self.train_images, self.train_labels, batch_size=self.args.batch_size, callbacks=[cp_callback], epochs=10
        )
        time_elapsed = time.time() - start
        # Calculate evaluation metric
        results = {
            "accuracy": float(history.history["accuracy"][-1]),
        }
        print(f"{results} | time_elapsed: {time_elapsed:.2f}s")

        # Get the parameters after training
        parameters_prime = model.get_model().get_weights()

        # Directly return the parameters and the number of examples trained on
        return parameters_prime, len(self.x_train), results

    def evaluate(self, parameters, config):
        # Set the weights of the model
        model.get_model().set_weights(parameters)

        global start
        # Evaluate the model and get the loss and accuracy
        loss, accuracy = model.get_model().evaluate(
            self.test_images, self.test_labels, batch_size=self.args.batch_size
        )
        time_elapsed = time.time() - start

        diagnostic_data = [time_elapsed, loss, accuracy]
        with open('/results/data.json', 'r', encoding='utf-8') as f:
            df = json.load(f)
            df.append(diagnostic_data)
        with open('/results/data.json', 'w', encoding='utf-8') as f:
            json.dump(df, f, ensure_ascii=False, indent=4)

        # Return the loss, the number of examples evaluated on and the accuracy
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}


# Function to Start the Client
def start_fl_client():
    try:
        with open('/results/data.json', 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        client = Client(args).to_client()
        fl.client.start_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}
    

if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()
