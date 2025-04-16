#!/bin/bash
copy2path(){
    # Update counter
    counter=$(($(cat ./results/.counter) + 1))
    echo $counter | tee ./results/.counter 
    # Move files tothe new folder
    export_dir="./results/$counter"
    mkdir $export_dir
    cp .env $export_dir && \
    cp results/server.json $export_dir && rm results/server.json && \
    cp results/client*.json $export_dir && rm results/client*.json && \
    cp results/model_accuracy.json $export_dir && \
    cp results/model_loss.json $export_dir && \
    cp results/eval-acc-fed-plot.png $export_dir && \
    cp results/loss-fed-plot.png $export_dir && \
    cp results/train-acc-fed-plot.png $export_dir && rm results/*.png
}

# common_properties="--coarse --mode=local --scale=10 --alpha=0.5 --train-split=0.8 --learning-rate=0.0006 --batch-size=16 --data-percentage=1.0 --epochs-per-subset=1 --total-epochs=20"
# python client.py $common_properties  

python helpers/generate_docker_compose.py
docker compose up --exit-code-from server
docker compose down --remove-orphans
copy2path