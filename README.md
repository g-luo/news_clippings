# [NewsCLIPpings Dataset](https://arxiv.org/abs/2104.05893)

[![DOI](https://zenodo.org/badge/355308357.svg)](https://zenodo.org/badge/latestdoi/355308357)

Our dataset with automatically generated out-of-context image-caption pairs in the news media. 
For inquiries and requests, please contact graceluo@berkeley.edu.

## Requirements
Make sure you are running Python 3.6+.

## Getting Started
1. Request the [VisualNews Dataset](https://github.com/FuxiaoLiu/VisualNews-Repository). 
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

<!-- Set up MMF
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/torch_stable.html
```
 -->
<!-- 3. Example command for training / finetuning with MMF.
```
MMF_USER_DIR="." nohup mmf_run config="./configs/experiments/clip.yaml" model=clip dataset=foil run_type=train > clip_train.out &
``` -->

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

## Available Upon Request
We have additional metadata available upon request, such as the [spaCy](https://spacy.io) and [REL](https://github.com/informagi/REL) named entities, timestamp, location of the original article content, etc.

We also have `sbert_embeddings_dissecting`, which has an embedding for each token and its weighting from running the "dissecting" setting of [SBERT-WK](https://github.com/BinWang28/SBERT-WK-Sentence-Embedding), available upon request. 
 
## Citing
If you find our dataset useful for your research, please, cite the following paper:
```
@article{luo2021newsclippings,
  title={NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media},
  author={Luo, Grace and Darrell, Trevor and Rohrbach, Anna},
  journal={arXiv:2104.05893},
  year={2021}
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
