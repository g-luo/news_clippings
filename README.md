# NewsCLIPpings Dataset

Our dataset for image-caption mismatch in the news. You can find our paper [here](https://arxiv.org/pdf/2104.05893.pdf). 
For inquiries and requests, please contact graceluo@berkeley.edu

## Getting Started
<!-- Set up MMF
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/torch_stable.html
```
 -->
1. Request the [VisualNews Dataset](https://github.com/FuxiaoLiu/VisualNews-Repository).
2. Download our matches in the `data/` folder of [http://news_clippings.berkeleyvision.org](http://news_clippings.berkeleyvision.org). All of the ids and image paths exactly correspond to those listed in the `data.json` file in VisualNews.
3. Consider doing analyses of your own using the embeddings we have provided in `embeddings/`.

<!-- 3. Example command for training / finetuning with MMF.
```
MMF_USER_DIR="." nohup mmf_run config="./configs/experiments/clip.yaml" model=clip dataset=foil run_type=train > clip_train.out &
``` -->

## Data Format
The data is ordered such that every even sample is pristine, and the next sample is its associated falsified sample. Let's say your file structure looks like this:

```
news_clippings
│
└── data/
└── embeddings/
└── your_notebook.ipynb

visual_news
│
└── origin/
│    └── data.json
│        ...
└── articles/
```

- `id`: the id of the VisualNews sample associated with the caption
- `image_id`: the id of the VisualNews sample associated with the image
- `similarity_score`: the similarity measure used to generate the sample (i.e. `clip_text_image, clip_text_text, sbert_text_text, resnet_place`)
- `falsified`: a binary indicator if the caption / image pair was the original pair in VisualNews or a mismatch we generated
- `source_dataset` (Merged / Balanced only): the index of the sub-split name in `source_datasets`

Here's an example of how you can start using our matches:
```
    import json
    your_path = ""
    visual_news_data = json.load(open(f"{your_path}/visualnews/origin/data.json"))
    visual_news_data_mapping = {ann["id"]: ann for ann in visual_news_data}
    
    data = json.load(open(f"{your_path}/news_clippings/data/merged_balanced/val.json"))
    annotations = data["annotations"]
    ann = annotations[0]
    
    caption = visual_news_data_mapping[ann["id"]]["caption"]
    image_path = visual_news_data_mapping[ann["image_id"]]["image_path"]
    
    print("Caption: ", caption)
    print("Image Path: ", image_path)
    print("Falsified: ", ann["falsified"])
```

## Embeddings
All embeddings are dictionaries of {id: numpy array} stored in pickle files for train / val / test. You can access the features for each image / caption by its id like so:

```
    import pickle
    clip_image_embeddings = pickle.load(open("clip_image_embeddings/test.pkl", "rb"))
    id = 701864
    print(clip_image_embeddings[id])
```

- `clip_image_embeddings`: 512-dim image embeddings from [CLIP](https://github.com/openai/CLIP) ViT-B/32
- `clip_text_embeddings`: 512-dim caption embeddings from [CLIP](https://github.com/openai/CLIP) ViT-B/32
- `sbert_embeddings`: 768-dim caption embeddings from [SBERT-WK](https://github.com/BinWang28/SBERT-WK-Sentence-Embedding)
- `places_resnet50`: 2048-dim image embeddings using ResNet50 trained on [Places365](https://github.com/CSAILVision/places365). 

The following embedding types were not used in the construction of our dataset, but you may find it useful.
- `facenet_embeddings`: 512-dim embeddings for each face detected in the images using [FaceNet](https://github.com/TIBHannover/cross-modal_entity_consistency/blob/master/visual_descriptors/person_embedding.py). If no faces were detected, returns `None`. 

## Available Upon Request
We have additional metadata available upon request, such as the [spaCy](https://spacy.io) and [REL](https://github.com/informagi/REL) named entities, timestamp, location of the original article content, etc.

We also have `sbert_embeddings_dissecting`, which has an embedding for each token and its weighting from running the "dissecting" setting of [SBERT-WK](https://github.com/BinWang28/SBERT-WK-Sentence-Embedding), available upon request. 
 
# Citing
```
@misc{luo2021newsclippings,
      title={NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media}, 
      author={Grace Luo and Trevor Darrell and Anna Rohrbach},
      year={2021},
      eprint={2104.05893},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
<!--
```
@misc{singh2020mmf,
 author =       {Singh, Amanpreet and Goswami, Vedanuj and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and
                Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
 title =        {MMF: A multimodal framework for vision and language research},
 howpublished = {\url{https://github.com/facebookresearch/mmf}},
 year =         {2020}
}
@misc{liu2020visualnews,
      title={VisualNews : Benchmark and Challenges in Entity-aware Image Captioning}, 
      author={Fuxiao Liu and Yinghan Wang and Tianlu Wang and Vicente Ordonez},
      year={2020},
      eprint={2010.03743},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
 @article{zhou2017places,
   title={Places: A 10 million Image Database for Scene Recognition},
   author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   year={2017},
   publisher={IEEE}
}
@misc{wang2020sbertwk,
      title={SBERT-WK: A Sentence Embedding Method by Dissecting BERT-based Word Models}, 
      author={Bin Wang and C. -C. Jay Kuo},
      year={2020},
      eprint={2002.06652},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
-->
