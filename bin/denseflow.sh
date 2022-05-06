#!/bin/bash

# the dataset absolute path
dataset_path=$1

# the video list path
video_list_path=$2

# the result path
result_path=$3

# number of GPUs
num_gpu=$4

[ ! -d "$dataset_path" ] && echo "$dataset_path directory not found" && exit
[ ! -f "$video_list_path" ] && echo "$video_list_path file not found" && exit
[ ! -d "$result_path" ] && mkdir -p "$result_path"

sed -i "s/\$DATASET_PATH/\/data/g" "$video_list_path"

sudo docker run --gpus all -it --rm \
                -v "$dataset_path":/data \
                -v "$video_list_path":/video_list.txt \
                -v "$result_path":/result \
                --workdir=/src \
                wizyoung/optical-flow-gpu \
                bash -c "python multi_gpu_extract.py /video_list.txt /result --flow_type tvl1 --out_fmt dir --num_gpu $num_gpu --step 1 --keep_frames True --log /data/log.log"
