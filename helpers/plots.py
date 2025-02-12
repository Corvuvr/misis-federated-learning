import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import ticker
from matplotlib import pyplot as plt

CLIENT_FOLDER = "results"
PLOTS_FOLDER = 'plots'

def single_client(data_path):
    
    with open(data_path, 'r', encoding='utf-8') as f:
        df = np.array(json.load(f)[1:]).astype('float32')
        (epochs, loss, eval_accuracy, train_accuracy) = df.transpose()
    fig = plt.figure()
    
    plt.plot(epochs, eval_accuracy)
    plt.plot(epochs, train_accuracy)
    plt.xlabel("Количество эпох")
    plt.ylabel("Точность")
    plt.title("Динамика обучения локальной модели: точность")
    plt.savefig(f'{Path(data_path).parent}/client-solo-acc.png', dpi=fig.dpi)
    plt.clf()
    
    plt.plot(epochs, loss)
    plt.xlabel("Количество эпох")
    plt.ylabel("Loss-функция")
    plt.title("Динамика обучения локальной модели: функция ошибки")
    plt.savefig(f'{Path(data_path).parent}/client-solo-loss.png', dpi=fig.dpi)
    plt.clf()

def federated_clients(data_path):
    print("LOG: Making Plots...")
    # shape=(3,len(metrics)) cols=10,20,30...len(metrics)
    client_data: list = []
    for i in range(1,4):
        with open(f'{data_path}/client{i}.json', 'r', encoding='utf-8') as f:
            # [ epochs, loss, eval_accuracy, train_accuracy ]
            df = np.array(json.load(f)[1:]).astype('float32').transpose()     
            client_data.append(df)
    
    # Clip clients to the minimal size - SMH client2 got 2040 epochs while others got 840
    min_len = min([len(x[0]) for x in client_data])
    client_data = [el[:, 0:min_len] for el in client_data]
    client_data = np.array(client_data)
    
    # Get columns (epochs)
    columns = [str(num_epochs) for num_epochs in client_data[0][0]]
    # Loss
    loss_across_clients = client_data[:, 1]

    # Client1: acc1, acc2, acc3 ... accN
    # Client2: acc1, acc2, acc3 ... accN
    # Client3: acc1, acc2, acc3 ... accN
    
    # Eval Accuracy
    eval_accuracies_across_clients = client_data[:, 2]
    # Train Accuracy
    train_accuracies_across_clients = client_data[:, 3]

    plotFedData(loss_across_clients, columns, f"{data_path}/loss-boxplot.png")
    plotFedData(eval_accuracies_across_clients, columns, f"{data_path}/eval-acc-boxplot.png")
    plotFedData(train_accuracies_across_clients, columns, f"{data_path}/train-acc-boxplot.png")

    print(list(os.walk(data_path)))


def plotFedData(data, columns, filename: str):
    dataframe = pd.DataFrame(data, columns=columns)
    plot = dataframe.plot(kind="box")
    
    # Set max interval on x axis
    plot.axes.xaxis.set_major_locator(ticker.MaxNLocator(10))

    plt.savefig(filename)
    print(f"LOG: Saved Plot in: {filename}")
    plt.clf()

def substitute_labels_with_step(labels, step):
    new_labels = ...
    return new_labels

def updatePlot(mode: str = "solo", data_path: str = "."):
    if mode=="solo":
        single_client(data_path=data_path)
    else:
        federated_clients(data_path=data_path)

if __name__=="__main__":
    federated_clients(data_path='metrics/data-0302024')
    pass