#!/bin/bash
download_url=https://huggingface.co/g-luo/news-clippings/resolve/main
root_dir=news_clippings
files=("train" "val" "test")

# Download all the splits. If you would like to specify certain splits, then change the list accordingly.
splits=(
    "merged_balanced"
    "person_sbert_text_text"
    "scene_resnet_place"
    "semantics_clip_text_image"
    "semantics_clip_text_text"
)

for split in "${splits[@]}"
do
    for file in "${files[@]}"
    do
    data_folder=data/${split}
    data_file=${data_folder}/${file}.json
    mkdir -p ${root_dir}/${data_folder}
    wget ${download_url}/${data_file} -O ${root_dir}/${data_file}
    done
done

# Download all the embeddings. If you would like to specify certain embeddings, then change the list accordingly.
embeddings=(
    "clip_image_embeddings"
    "clip_text_embeddings"
    "facenet_embeddings"
    "places_resnet50"
    "sbert_embeddings"
)
for embedding in "${embeddings[@]}"
do
    for file in "${files[@]}"
    do
        embedding_folder=embeddings/${embedding}
        embedding_file=${embedding_folder}/${embedding}_${file}.pkl
        mkdir -p ${root_dir}/${embedding_folder}
        wget ${download_url}/${embedding_file}?download=true -O ${root_dir}/${embedding_file}
    done
done