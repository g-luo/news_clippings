#!/bin/bash
MMF_USER_DIR="." \
  nohup mmf_run config=configs/experiments/clip.yaml \
  model=clip \
  dataset=news_clippings \
  run_type=train \
  checkpoint.resume_pretrained=False \
  env.save_dir=runs/${split}_${model} \
  env.tensorboard_logdir=runs/${split}_${model} \
  > ${split}_${model}_train.out &
