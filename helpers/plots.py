import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
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

def updatePlot(mode: str = "solo", data_path: str = "."):
    if mode=="solo":
        single_client(data_path=data_path)
    else:
        pass