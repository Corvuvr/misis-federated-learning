#!/bin/bash
copy2path(){
    export_dir=$1
    mkdir $export_dir
    cp .env $export_dir && \
    cp results/server.json $export_dir && \
    cp results/client1.json $export_dir && \
    cp results/client2.json $export_dir && \
    cp results/eval-acc-fed-plot.png $export_dir && \
    cp results/loss-fed-plot.png $export_dir && \
    cp results/train-acc-fed-plot.png $export_dir
}

common_properties="--mode=local --total-epochs=70 --epochs-per-subset=10 --batch-size=64 --learning-rate=0.0016 --data-percentage=1.0 --train-split=0.8 --scale=5 --alpha=0.5"
# python client.py $common_properties  

docker compose up --exit-code-from server
copy2path "./results/19"