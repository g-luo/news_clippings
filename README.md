# NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media

**Grace Luo, Trevor Darrell, Anna Rohrbach**

This repository hosts the training code and dataset of NewsCLIPpings. The dataset contains automatically generated out-of-context image-caption pairs in the news media. You can download and set up the data by following the steps below.

[[`arXiv`](https://arxiv.org/abs/2104.05893)]

## Requirements
Make sure you are running Python 3.6+.

## Getting Started
1. Download the [VisualNews Dataset](https://github.com/FuxiaoLiu/VisualNews-Repository) from [this url](https://www.cs.rice.edu/~vo9/visualnews). 
Place the files under the `visual_news` folder.
2. Run [`./download.sh`](https://github.com/g-luo/news_clippings/blob/master/download.sh) to download our matches and populate the `news_clippings` folder (place into `news_clippings/data/`). 
3. Consider doing analyses of your own using the embeddings we have provided (place into `news_clippings/embeddings/`).

All of the ids and image paths provided in our `data/` folder exactly correspond to those listed in the `data.json` file in VisualNews. 
<!--If you have trouble running our download script, you can find everything at [http://news_clippings.berkeleyvision.org](http://news_clippings.berkeleyvision.org).-->

Your file structure should look like this:

```
news_clippings
│
└── data/
└── embeddings/

visual_news
│
└── origin/
│    └── data.json
│        ...
└── ...
```

## Data Format
The data is ordered such that every even sample is pristine, and the next sample is its associated falsified sample. 

- `id`: the id of the VisualNews sample associated with the caption
- `image_id`: the id of the VisualNews sample associated with the image
- `similarity_score`: the similarity measure used to generate the sample (i.e. `clip_text_image, clip_text_text, sbert_text_text, resnet_place`)
- `falsified`: a binary indicator if the caption / image pair was the original pair in VisualNews or a mismatch we generated
- `source_dataset` (Merged / Balanced only): the index of the sub-split name in `source_datasets`

Here's an example of how you can start using our matches:
```
    import json
    visual_news_data = json.load(open("visualnews/origin/data.json"))
    visual_news_data_mapping = {ann["id"]: ann for ann in visual_news_data}
    
    data = json.load(open("news_clippings/data/merged_balanced/val.json"))
    annotations = data["annotations"]
    ann = annotations[0]
    
    caption = visual_news_data_mapping[ann["id"]]["caption"]
    image_path = visual_news_data_mapping[ann["image_id"]]["image_path"]
    
    print("Caption: ", caption)
    print("Image Path: ", image_path)
    print("Is Falsified: ", ann["falsified"])
```

## Embeddings
We include the following precomputed embeddings:

- `clip_image_embeddings`: 512-dim image embeddings from [CLIP](https://github.com/openai/CLIP) ViT-B/32. <br />
Contains embeddings for samples in all splits.
- `clip_text_embeddings`: 512-dim caption embeddings from [CLIP](https://github.com/openai/CLIP) ViT-B/32. <br />
Contains embeddings for samples in all splits.
- `sbert_embeddings`: 768-dim caption embeddings from [SBERT-WK](https://github.com/BinWang28/SBERT-WK-Sentence-Embedding). <br />
Contains embeddings for samples in all splits.
- `places_resnet50`: 2048-dim image embeddings using ResNet50 trained on [Places365](https://github.com/CSAILVision/places365). <br />
Contains embeddings only for samples in the `scene_resnet_place` split (where [PERSON] entities were not detected in the caption).

The following embedding types were not used in the construction of our dataset, but you may find them useful.
- `facenet_embeddings`: 512-dim embeddings for each face detected in the images using [FaceNet](https://github.com/TIBHannover/cross-modal_entity_consistency/blob/master/visual_descriptors/person_embedding.py). If no faces were detected, returns `None`. <br />
Contains embeddings only for samples in the `person_sbert_text_text` split (where [PERSON] entities were detected in the caption).

All embeddings are dictionaries of {id: numpy array} stored in pickle files for train / val / test. You can access the features for each image / caption by its id like so:

```
    import pickle
    clip_image_embeddings = pickle.load(open("news_clippings/embeddings/clip_image_embeddings/test.pkl", "rb"))
    id = 701864
    print(clip_image_embeddings[id])
```

## Metadata
We have additional metadata, such as the [spaCy](https://spacy.io) and [REL](https://github.com/informagi/REL) named entities, timestamp, location of the original article content, etc.

## Training
To run the benchmarking experiments we reported in our paper, look at the README for `news_clippings_training/`.
 
## Citing
If you find our dataset useful for your research, please, cite the following paper:
```
@inproceedings{luo2021newsclippings,
  title={NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media},
  author={Luo, Grace and Darrell, Trevor and Rohrbach, Anna},
  journal={EMNLP},
  year={2021}
}
```
