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
    cp results/eval-acc-fed-plot.png $export_dir && \
    cp results/loss-fed-plot.png $export_dir && \
    cp results/train-acc-fed-plot.png $export_dir && rm results/*.png
}

# common_properties="--mode=local --total-epochs=70 --epochs-per-subset=10 --batch-size=24 --learning-rate=0.0016 --data-percentage=1.0 --train-split=0.8 --scale=5 --alpha=0.5"
# python client.py $common_properties  

python helpers/generate_docker_compose.py
docker compose up --exit-code-from server
copy2path