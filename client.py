import os
import time
import logging
import requests

import flwr as fl
import tensorflow as tf

from model.model import Model
from helpers.plots import updatePlot
from helpers.load_data import load_data, shuffle, push_json, make_json
from helpers.client_args import args

# Logs
logging.basicConfig(level=logging.INFO)     # Configure logging
logger = logging.getLogger(__name__)        # Create logger for the module
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"    # Make TensorFlow log less verbose
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger.info(f"GPUS:\t{tf.config.list_physical_devices('GPU')}")
logger.info(str(args))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], 
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*3)]
        )
        #tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        logger.error(e)

# ============================================ FEDERATED CLIENT ============================================
class Client(fl.client.NumPyClient):
    def __init__(self, args):
        super().__init__()
        self.args = args
        logger.info("Preparing data...")
        (self.train_images, y_train, z_train), (self.test_images, y_test, z_test) = load_data(
            mode="local",
            client_id=args.client_id,
            total_clients=args.total_clients,
            train_split=args.train_split,
            server_ip=args.flask_address
        )
        self.train_labels = z_train if COARSE_LEARNING else y_train
        self.test_labels = z_test if COARSE_LEARNING else y_test
        logger.info(f"Got data:\t{self.train_images.shape=}\t{self.test_images.shape=}")

    def get_parameters(self, config=None):
        # Return the parameters of the model
        return model.get_model().get_weights()

    def fit(self, parameters, config=None):
        global epochs
        global train_accuracy
        # Set the weights of the model
        model.get_model().set_weights(parameters)
        # Get training subset
        train_images, train_labels = self.train_images, self.train_labels
        global DO_SHUFFLE
        if DO_SHUFFLE:
            train_images, train_labels = shuffle(
                percentage=args.data_percentage,
                args=(self.train_images, self.train_labels)
            )
        # Early stopping after no improvement in N epochs 
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
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
            # callbacks=[cp_callback, es_callback], 
            callbacks=[es_callback], 
            epochs=args.epochs_per_subset
        )
        # The reason why client 1st epoch is 0?
        epochs += args.epochs_per_subset
        # Calculate evaluation metric
        train_accuracy = float(history.history["accuracy"][-1])
        results = { "accuracy": train_accuracy, }

        # Get the parameters after training
        parameters_prime = model.get_model().get_weights()
        # Directly return the parameters and the number of examples trained on
        return parameters_prime, len(self.train_images), results

    def evaluate(self, parameters, config=None):
        global epochs
        global DO_SHUFFLE
        model.get_model().set_weights(parameters)
        test_images, test_labels = self.test_images, self.test_labels
        if DO_SHUFFLE:
            test_images, test_labels = shuffle(
                percentage=args.data_percentage,
                args=(self.test_images, self.test_labels)
            )
        # Evaluate the model and get the loss and accuracy
        loss, eval_accuracy = model.get_model().evaluate(
            test_images, test_labels, batch_size=self.args.batch_size
        )
        if epochs > 0:
            diagnostic_data = [epochs, float(loss), float(eval_accuracy), float(train_accuracy)]
            push_json(JSON_METRICS_PATH, diagnostic_data)
        # Return the loss, the number of examples evaluated on and the accuracy
        return float(loss), len(self.test_images), {"accuracy": float(eval_accuracy)}

# Function to Start the Client
def start_fl_client():
    try:
        make_json(path=JSON_METRICS_PATH, data=args.__dict__)
        client = Client(args).to_client()
        fl.client.start_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


# ============================================ INIT ============================================
if __name__ == "__main__":

    # Globals
    DO_SHUFFLE = False
    COARSE_LEARNING     = args.coarse
    CLIENT_FOLDER       = "results" if args.mode == "local" else "/results"
    CLIENT_PREFIX       = str(args.client_id) if not args.mode == "local" else "-solo"
    WEIGHTS_PATH        = f'{CLIENT_FOLDER}/client{CLIENT_PREFIX}.weights.h5'
    JSON_METRICS_PATH   = f'{CLIENT_FOLDER}/client{CLIENT_PREFIX}.json'

    if args.mode == "legacy":
        wait_for_server: bool = True
        while wait_for_server:
            time.sleep(10)
            try:
                logger.info("Waiting for server...")
                if requests.get(f'http://{args.flask_address}:7272/establish_connection'):
                    wait_for_server = False
            except:
                logger.info("Waiting for server...")
        logger.error(f"Connection established. Initializing client{CLIENT_PREFIX}...")

    # Model
    num_classes = 20 if COARSE_LEARNING else 100
    model = Model(
        learning_rate=args.learning_rate, 
        classes_=num_classes,
        alpha_=args.alpha,
        scale_input=args.scale
    )
    model.compile()

    # Learning
    epochs: int = 0
    train_accuracy: float = 0
    if args.mode == "local":
        make_json(path=JSON_METRICS_PATH, data=args.__dict__)
        c = Client(args)
        params = c.get_parameters()
        while epochs < args.total_epochs:
            logger.info(f"Epoch {epochs}...")
            params, num_examples, results = c.fit(params)
            c.evaluate(params)
        updatePlot(data_path=CLIENT_FOLDER)
    else:
        start_fl_client()