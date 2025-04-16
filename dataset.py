import logging
import numpy as np
from pprint import pp
from datasets import ClassLabel, Dataset
from typing import Sequence, Iterable, Generator
from flwr_datasets import FederatedDataset

# Logs
logging.basicConfig(level=logging.INFO)     # Configure logging
logger = logging.getLogger(__name__)        # Create logger for the module

# ======================================= GLOBALS =======================================
dummy_dataset = FederatedDataset(dataset="cifar100", partitioners={"train": 2}) # For deriving meta
meta_features: ClassLabel = dummy_dataset.load_partition(0, "train").features["coarse_label"]
coarse_ids: list[int] = [meta_features.str2int(label) for label in meta_features.names]

# ======================================= UTILITY =======================================
def split_sequence_by_step(seq: Sequence, step: int) -> list[list]:
    """
    Yields sequence of item positions related to given class ids. 
    """
    return [list(seq)[i:i + step] for i in range(0, len(seq), step)]

def get_indicies_of_classes(data: Iterable, classes: list[str]) -> Generator[int, None, None]:
    """
    Yields sequence of item positions related to given class ids. 
    """
    logger.info(f"Searching for {len(classes)} in data...")
    for i in range(len(data)):
        if data[i] in classes:
            yield i

def split_coarse_labels(num_clients: int) -> list[list[int]]:
    """
    Breaks coarse labels into chunks. Number of chunks is equal to number of clients.
    
    It is way simpler than splitting fine labels since 
    we don't have to make sure that each coarse class is represented. 
    """
    global coarse_ids
    labels: Sequence = coarse_ids
    return list(
        split_sequence_by_step(
            seq=labels,
            step=int(np.ceil(len(labels)/num_clients))
        )
    )

def split_fine_labels(num_clients: int) -> list[list]:
    """
    Breaks fine labels into chunks in the way  
    where each chunk has at least one fine label in every superclass.
    
    Number of chunks is equal to number of clients.
    
    If there are less clients than classes within a superclass, 
    then fine classes within every chunk are not repeated. 
    """
    label_map: dict[int, set[int]] = get_label_mapping()    
    fine_sets: Sequence[set[int]] = label_map.values()

    # Restrict number of chunks the way so every chunk gets at least one label
    client_scope: int = min(
        num_clients,
        min(len(fine_set) for fine_set in fine_sets)
    )
    client_chunks: list[list] = [list() for client in range(client_scope)]
    
    # Divide fine_data into chunks
    for fine_set in label_map.values():
        split_step: int = int(np.ceil(len(fine_set) / client_scope))
        fine_chunks = list(split_sequence_by_step(seq=fine_set, step=split_step))
        for i in range(client_scope):
            client_chunks[i] += fine_chunks[i]
    
    # Extrapolate chunks until every client gets one
    i: int = 0
    while len(client_chunks) < num_clients:
        logger.info(f"Duplicate chunk with id {i}: {client_chunks[i]}")
        client_chunks.append(client_chunks[i])
        i = (i+1) % len(client_chunks)
    
    return client_chunks

def get_label_mapping() -> dict[int, set[int]]:
    """
    Returns dict that matches every coarse label to its set of fine labels  
    """
    global coarse_ids
    # Get patrition containing required metadata
    partition: Dataset = dummy_dataset.load_partition(0, "train")
    partition.set_format("numpy")
    # Actual code
    match_labels: dict = dict(zip(coarse_ids, [set() for id in coarse_ids]))
    ds_sz: int = len(partition["img"])
    fine_partition = partition['fine_label']
    coarse_partition = partition['coarse_label']
    for i in range(ds_sz):
        coarse_item: int = coarse_partition[i]
        fine_item:   int = fine_partition[i]
        fine_subset: set = match_labels[coarse_item]
        fine_subset.add(fine_item)
        match_labels.update({coarse_item: fine_subset})
    return match_labels

# ======================================= DATA DISTRIBUTION =======================================
def get_full_dataset(train_split) -> dict:
    """
    Returns dataset represented as dict and meta_partition
    """
    # We can fake total_clients since we grab the whole dataset anyway
    total_clients: int = 2 # total_clients == 1 doesn't work!!! 
    fds = FederatedDataset(dataset="cifar100", partitioners={"train": total_clients})
    x_train, y_train, z_train, x_test, y_test, z_test = (None,)*6 
    def collect(l, r):
        return r if (type(l) == type(None)) else np.append(l, r[:], axis=0)
    for client in range(total_clients):
        partition = fds.load_partition(client, "train")
        partition.set_format("numpy")
        # Divide data on each client: 80% train, 20% test
        partition = partition.train_test_split(test_size=1-train_split, seed=42)
        x_train = collect(x_train,  partition["train"]["img"] / 255.0 )
        y_train = collect(y_train,  partition["train"]["fine_label"]  )
        z_train = collect(z_train,  partition["train"]["coarse_label"])
        x_test  = collect(x_test,   partition["test"]["img"] / 255.0  )
        y_test  = collect(y_test,   partition["test"]["fine_label"]   )
        z_test  = collect(z_test,   partition["test"]["coarse_label"] )
    return { 
        "train": {
            "img"           : x_train, 
            "fine_label"    : y_train,
            "coarse_label"  : z_train,
        }, 
        "test":  {
            "img"           : x_test, 
            "fine_label"    : y_test,
            "coarse_label"  : z_test,
        }, 
    }

def get_label_banks(split_type: str, total_clients: int) -> list[Sequence[int]]:
    """
    Returns a sequence of labels for every client to learn on.
    """
    match split_type:
        case "fine":
            label_banks: list[Sequence[int]] = split_fine_labels(total_clients)
        case "coarse":
            label_banks: list[Sequence[int]] = split_coarse_labels(total_clients)
        case "none":
            label_banks: list[Sequence[int]] = [coarse_ids] * total_clients
        case _:
            raise Exception(f"Wrong split type: {split_type}. Should be: [ fine | coarse | none ]")
    return label_banks

def get_split_partition(dataset: dict, label_banks: Sequence[int], split_type: str, client_id: int) -> Sequence[np.ndarray]:
    """
    Returns a dataset partition of given label ids.
    """
    # Get ids of the dataset part which has the mentioned classes
    if split_type == "none":
        split_type = "coarse"
    print(f"Hello World: {len(label_banks)=} {client_id=}")
    train_partition_indicies: list[int] = list(get_indicies_of_classes(
        data=dataset["train"][f"{split_type}_label"], classes=label_banks[client_id]
    ))
    print(f"{len(train_partition_indicies)=}")
    test_partition_indicies: list[int] = list(get_indicies_of_classes(
        data=dataset["test"][f"{split_type}_label"],  classes=label_banks[client_id]
    ))
    print(f"{len(test_partition_indicies)=}")
    # Derive data of given labels
    return { 
        "train": {
            "img"           : dataset["train"] ["img"         ] [train_partition_indicies], 
            "fine_label"    : dataset["train"] ["fine_label"  ] [train_partition_indicies],
            "coarse_label"  : dataset["train"] ["coarse_label"] [train_partition_indicies],
        }, 
        "test": {
            "img"           : dataset["test"]  ["img"         ] [test_partition_indicies], 
            "fine_label"    : dataset["test"]  ["fine_label"  ] [test_partition_indicies],
            "coarse_label"  : dataset["test"]  ["coarse_label"] [test_partition_indicies],
        }, 
    }

# ======================================= EXECUTION =======================================
def classic_scenario(split_type: str, total_clients: int, client_id: int, train_split: float = 0.8):
    dataset: dict = get_full_dataset(
        train_split=train_split
    )
    label_banks: list[Sequence[int]] = get_label_banks(
        split_type=split_type, total_clients=total_clients
    )
    data = get_split_partition(
        dataset=dataset, label_banks=label_banks, split_type=split_type, client_id=client_id
    )
    return data

if __name__ == "__main__":
    # data = classic_scenario(
    #     split_type="fine", client_id=1, total_clients=2
    # )
    # print(list(data["train"]["img"].shape))
    # from matplotlib import pyplot as plt
    # plt.imshow(data["train"]["img"][0], interpolation='nearest')
    # plt.show()
    pass