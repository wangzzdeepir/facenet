#!/bin/bash

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/log.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")

set -x
time python src/zz_train_classifier.py \
  --logs_base_dir experiments/tensorboard \
  --models_base_dir output \
  --data_dir /data/face-recognition/casia-webface/CASIA-maxpy_mtcnnpy_182 \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --lfw_dir /data/face-recognition/lfw/lfw_mtcnnpy_160 \
  --optimizer RMSPROP \
  --learning_rate -1 \
  --max_nrof_epochs 80 \
  --batch_size 100 \
  --keep_probability 0.8 \
  --random_crop \
  --random_flip \
  --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
  --weight_decay 5e-5 \
  --center_loss_factor 1e-4 \
  --center_loss_alfa 0.9 \
  --gpu_memory_fraction 0.95 \
  --filter_filename data/filtering_metrics.h5 \
  --filter_percentile 80 \
  --filter_min_nrof_images_per_class 2
