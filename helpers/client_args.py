import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")
# Learning args
parser.add_argument(
    "--batch-size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning-rate", type=float, default=0.1, help="Learning rate for the optimizer"
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
    "--train-split", type=float, default=0.8, help="In which proportion split partition"
)
parser.add_argument(
    "--coarse", default=False, action='store_true', help="Whether to train on fine (superclass) labels or coarse (class) labels"
)
parser.add_argument(
    "--mode", type=str, default="local", choices=["local", "fed", "legacy"], help="Whether to launch locally or as a federated client"
)
parser.add_argument(
    "--split-type", type=str, default="coarse", choices=["coarse", "fine", "none"], help="Whether to launch split data by coarse or fine labels"
)

# Fed args
parser.add_argument(
    "--server-address", type=str, default="server:8080", help="Address of the fed server"
)
parser.add_argument(
    "--flask-address", type=str, default="0.0.0.0", help="Address of the data server"
)
parser.add_argument(
    "--client-id", type=int, default=1, help="Unique ID for the client"
)
parser.add_argument(
    "--total-clients", type=int, default=2, help="Total number of clients"
)

args = parser.parse_args()