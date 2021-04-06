# NewsCLIPpings Dataset

Our dataset for image-caption mismatch in the news.

# Getting Started
<!-- Set up MMF
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/torch_stable.html
```
 -->
1. Request the [VisualNews Dataset](https://github.com/FuxiaoLiu/VisualNews-Repository) from fl3es@virginia.edu.
2. Download our matches in the data/ folder. All of the ids and image paths exactly correspond to those listed in the data.json file in VisualNews.

<!-- 3. Example command for training / finetuning with MMF.
```
MMF_USER_DIR="." nohup mmf_run config="./configs/experiments/clip.yaml" model=clip dataset=foil run_type=train > clip_train.out &
``` -->

# Citations
<!-- @misc{singh2020mmf,
  author =       {Singh, Amanpreet and Goswami, Vedanuj and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and
                 Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  title =        {MMF: A multimodal framework for vision and language research},
  howpublished = {\url{https://github.com/facebookresearch/mmf}},
  year =         {2020}
}
 -->
```
@misc{liu2020visualnews,
      title={VisualNews : Benchmark and Challenges in Entity-aware Image Captioning}, 
      author={Fuxiao Liu and Yinghan Wang and Tianlu Wang and Vicente Ordonez},
      year={2020},
      eprint={2010.03743},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```