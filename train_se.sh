#!/bin/bash
# nohup bash train_se.sh mf > mf.log 2>&1 &
if [[ $# -eq 0 ]] ; then
    echo 'Please provide model name'
    exit 1
fi
arr=("cc")
data_dir="./data"
epoch=10
device="cuda:1"

for i in "${arr[@]}"; do
    echo ---------------ontology $i----------------
    echo train_mlp.py -m $1 -ont $i -dr $data_dir -ep $epoch -d $device
    python train_mlp.py -m $1 -ont $i -dr $data_dir -ep $epoch -d $device
    echo evaluate.py -m $1 -ont $i -dr $data_dir
    python evaluate.py -m $1 -ont $i -dr $data_dir
done
