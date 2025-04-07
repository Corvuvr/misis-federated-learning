import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import ticker
from matplotlib import pyplot as plt

def single_client(data_path):
    
    json_filename = f'{data_path}/client-solo.json'
    
    with open(json_filename, 'r', encoding='utf-8') as f:
        df = np.array(json.load(f)[1:]).astype('float32')
        (epochs, loss, eval_accuracy, train_accuracy) = df.transpose()
    fig = plt.figure()
    
    plt.plot(epochs, eval_accuracy)
    plt.plot(epochs, train_accuracy)
    plt.xlabel("Количество эпох")
    plt.ylabel("Точность")
    plt.title("Динамика обучения локальной модели: точность")
    acc_filename = f'{data_path}/client-solo-acc.png'
    plt.savefig(acc_filename, dpi=fig.dpi)
    print(f"LOG: Saved Plot in: {acc_filename}")
    plt.clf()
    
    plt.plot(epochs, loss)
    plt.xlabel("Количество эпох")
    plt.ylabel("Loss-функция")
    plt.title("Динамика обучения локальной модели: функция ошибки")
    loss_filename = f'{data_path}/client-solo-loss.png'
    plt.savefig(loss_filename, dpi=fig.dpi)
    print(f"LOG: Saved Plot in: {loss_filename}")
    plt.clf()

    print(list(os.walk(data_path)))

    import datetime
    directory_name = f'{data_path}/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}' 
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
        import shutil
        shutil.copy(acc_filename, directory_name)
        shutil.copy(loss_filename, directory_name)
        shutil.copy(json_filename, directory_name)
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def federated_clients(data_path, num_clients = 1):
    print("LOG: Making Plots...")
    # shape=(3,len(metrics)) cols=10,20,30...len(metrics)
    client_data: list = []
    for i in range(1,num_clients+1):
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
    epochs = client_data[:,0][0]
    loss_across_clients = client_data[:, 1]

    # Client1: acc1, acc2, acc3 ... accN
    # Client2: acc1, acc2, acc3 ... accN
    # Client3: acc1, acc2, acc3 ... accN
    
    # Eval Accuracy
    eval_accuracies_across_clients = client_data[:, 2]
    # Train Accuracy
    train_accuracies_across_clients = client_data[:, 3]

    loss_sav = f"{data_path}/loss-fed-plot.png"
    vacc_sav = f"{data_path}/eval-acc-fed-plot.png"
    tacc_sav = f"{data_path}/train-acc-fed-plot.png"

    loss_title = "Функция ошибки среди узлов"
    vacc_title = "Валидационная точность среди узлов"
    tacc_title = "Точность обучения среди узлов"

    if num_clients >= 5:
        plotFedData(loss_across_clients,             columns, loss_sav, loss_title)
        plotFedData(eval_accuracies_across_clients,  columns, vacc_sav, vacc_title)
        plotFedData(train_accuracies_across_clients, columns, tacc_sav, tacc_title)
    else:
        plotMetrics(
            metrics=loss_across_clients, 
            columns=epochs,
            savepath=loss_sav,
            xlabel="Количество эпох",
            ylabel="Ошибка",
            title=loss_title
        )
        plotMetrics(
            metrics=eval_accuracies_across_clients, 
            columns=epochs,
            savepath=vacc_sav,
            xlabel="Количество эпох",
            ylabel="Точность",
            title=vacc_title
        )
        plotMetrics(
            metrics=train_accuracies_across_clients, 
            columns=epochs,
            savepath=tacc_sav,
            xlabel="Количество эпох",
            ylabel="Точность",
            title=tacc_title
        )

    print(list(os.walk(data_path)))

def plotMetrics(metrics, columns, savepath: str, xlabel: str = "", ylabel: str = "", title: str = ""):
    fig = plt.figure()
    for i in range(len(metrics)):
        plt.plot(columns, metrics[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savepath, dpi=fig.dpi)
    print(f"LOG: Saved Plot in: {savepath}")
    plt.clf()   


def plotFedData(data, columns, filename: str, title: str):
    dataframe = pd.DataFrame(data, columns=columns)
    num_clients = len(data)
    print(f"{num_clients=}")
    # Make boxplot if there is a lot of clients
    plot = dataframe.plot(kind="box")

    # Set max interval on x axis
    plot.axes.xaxis.set_major_locator(ticker.MaxNLocator(10))
    plt.title(title)

    plt.savefig(filename)
    print(f"LOG: Saved Plot in: {filename}")
    plt.clf()

def updatePlot(data_path: str = ".", num_clients = 0):
    if num_clients==0:
        single_client(data_path=data_path)
    elif num_clients > 0:
        federated_clients(data_path=data_path, num_clients=num_clients)
    else:
        raise Exception('No clients!')
if __name__=="__main__":
    federated_clients(data_path='results', num_clients=2)
    pass