import argparse

parser = argparse.ArgumentParser(description="Flower Server")
parser.add_argument(
    "--number-of-rounds",
    type=int,
    default=100,
    help="Number of FL rounds (default: 100)",
)
parser.add_argument(
    "--split-type", 
    type=str, 
    default="fine", 
    choices=["fine", "coarse", "none"], 
    help="Distribute data among nodes by fine or coarse labels"
)
parser.add_argument(
    "--flask-address", type=str, default="0.0.0.0", help="Address of the data server"
)
parser.add_argument(
    "--total-clients", type=int, default="0", help="Number of clients"
)
parser.add_argument(
    "--legacy", default=False, action='store_true', help="Use legacy data distribution"
)

args = parser.parse_args()