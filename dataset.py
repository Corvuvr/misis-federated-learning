import numpy as np
from typing import Sequence, Iterable, Generator
from flwr_datasets import FederatedDataset
# ======================================= UTILITY =======================================
def split_list_by_step(seq: Sequence, step: int) -> Generator[Sequence[int], None, None]:
    seq = list(seq)
    if step <= 0:
        raise Exception(f"Step should be greater than zero! Got: {step}")
    j = 0
    while j < len(seq):
        partition: list = []
        for i in range(len(seq)):
            try:
                partition.append(seq[j+i])
            except:
                j += i + 1
                break
            if i == step - 1:
                j += i + 1
                break
        yield partition
def get_indicies_of_classes(data: Iterable, classes: list[str]) -> Generator[int, None, None]:
    for i in range(len(data)):
        if data[i] in classes:
            yield i
def distribute_labels_across_clients(labels: Sequence, num_clients: int) -> list[list[int]]:
    return list(
        split_list_by_step(
            seq=labels,
            step=np.ceil(len(labels)/num_clients)
        )
    )

# ======================================= DATA DISTRIBUTION =======================================
def get_full_dataset(train_split, total_clients):
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

def get_label_banks(dataset: dict, split_type: str, total_clients: int) -> list[Sequence[int]]:
    match split_type:
        case "fine":
            label_banks: list[Sequence[int]] = distribute_labels_across_clients(
                labels=set(dataset['train']['fine_label']), 
                num_clients=total_clients
            )
        case "coarse":
            label_banks: list[Sequence[int]] = distribute_labels_across_clients(
                labels=set(dataset['train']['coarse_label']), 
                num_clients=total_clients
            )
        case "none":
            label_banks: list[Sequence[int]] = [set(dataset['train']['coarse_label'])] * total_clients
        case _:
            raise Exception(f"Wrong split type: {split_type}. Should be: [ fine | coarse | none ]")
    return label_banks

def get_split_partition(dataset: dict, label_banks: Sequence[int], split_type: str, client_id: int) -> Sequence[np.ndarray]:
    # Get ids of the dataset part which has the mentioned classes
    if split_type == "none":
        split_type = "coarse"
    train_partition_indicies: list[int] = list(get_indicies_of_classes(
        data=dataset["train"][f"{split_type}_label"], classes=label_banks[client_id]
    ))
    test_partition_indicies: list[int] = list(get_indicies_of_classes(
        data=dataset["test"][f"{split_type}_label"],  classes=label_banks[client_id]
    ))
    # Compose 
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
def classic_scenario(split_type: str, total_clients: int, client_id: int):
    dataset: dict = get_full_dataset(
        train_split=0.8, total_clients=total_clients
    )
    label_banks: list[Sequence[int]] = get_label_banks(
        dataset=dataset, split_type=split_type, total_clients=total_clients
    )
    data = get_split_partition(
        dataset=dataset, label_banks=label_banks, split_type=split_type, client_id=client_id
    )
    return data

if __name__ == "__main__":
    data = classic_scenario(
        split_type="coarse", client_id=1, total_clients=2
    )
    print(list(data["train"]["img"].shape))
    from matplotlib import pyplot as plt
    plt.imshow(data["train"]["img"][0], interpolation='nearest')
    plt.show()