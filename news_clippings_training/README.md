# NewsCLIPpings Training Code

## Getting Started
To get started, install MMF. Here we provide suggested versions of MMF, torch, and torchvision that are known to be compatible with CLIP.
```
  git clone https://github.com/facebookresearch/mmf
  cd mmf
  git checkout 08f062ef8cc605eed4a5dba729899c1cfc88a23b
  pip install --editable .

  pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## Setup
Download the dataset as instructed by our [dataset repo](https://github.com/g-luo/news_clippings#data-format). Also download the VisualNews dataset for the captions and images. The folder structure should look something like this:

```
${data_dir : ~/.cache/torch/mmf/data/datasets}
└── news_clippings
    └── data/
└── visual_news
    └── origin/
```

Set the environment variables by running `source scripts/environment.sh`. 

## Repo Structure
`datasets/`
* `builder.py`: Tells MMF where to find the dataloader associated with `dataset=news_clippings`.
* `database.py`: Tells MMF how to load the annotations.
* `news_clippings.py`: Defines main dataloader including how to handle image / feature paths. Defines the formattting for predictions from `mmf_predict`.
* `processors.py`: Contains data preprocessing functions such as tokenization.

`models/`
* `clip.py`: Contains a model that featurizes image and text using CLIP then passes it through an MLP classifier.
* `CLIP`: Is a copy of the original [CLIP repo](https://github.com/openai/CLIP.git). However, the file `simple_tokenizer` has been modified to return both the tokens and chunked words themselves.
* `utils.py`: Contains util functions for customizing learning rate and layer freezing.

`configs/`
* `news_clippings.yaml`: Contains default data paths and training parameters. It is recommended not to modify this file. It is recommended *not* to modify this file.
* `models/`: Contains default parameters for models. It is recommended *not* to modify these files.
* `experiments/`: Contains final configs used in training / inference. It is recommended to modify these files and take inspiration from the defaults above.

`scripts/`
* `environment.sh`: Sets environment variables used in other scripts.
* `train.sh`: Default training script.
* `test.sh`: Default inference script.

## Running Experiments
You can choose to run your own custom configs by changing the flag to `config=configs/experiments/clip.yaml`. You can tune hyperparameters for model fitting using the fields `optimizer`, `training`, `scheduler`. You can also modify architecture choices in `model_config`.

## Training 
Run training in the main `news_clippings_training/` folder using `scripts/train.sh`.

## Inference
Run inference in the main `news_clippings_training/` folder using `scripts/test.sh`.

Make sure `checkpoint.resume_pretrained` is set to False otherwise MMF will try to continue training your model during inference. To output predictions by sample run `mmf_predict`, else just output the accuracy with `mmf_run`.

