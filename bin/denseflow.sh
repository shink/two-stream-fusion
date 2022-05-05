#!/bin/bash

# the dataset absolute path
dataset_path=$1

# the video list path
video_list_path=$2

cd "$dataset_path" || exit
mkdir result

OLDIFS=$IFS
IFS=','

[ ! -f "$video_list_path" ] && echo "$video_list_path file not found" && exit

while read -r label video_path; do
  echo "label: $label"
  echo "path: $video_path"

done <"$video_list_path"

IFS=$OLDIFS

#sudo docker run --gpus all -it -v "$dataset_path":/data wizyoung/optical-flow-gpu bash -c
