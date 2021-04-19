#!/bin/bash
root_dir="news_clippings/"

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
wget -np -r -nH -P ${root_dir} --reject="index.html*" http://news_clippings.berkeleyvision.org/data/${split}/
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
wget -np -r -nH -P ${root_dir} --reject="index.html*" http://news_clippings.berkeleyvision.org/embeddings/${embedding}/
done
