GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONUNBUFFERED="True"

cd ..
python src/zz_train_classifier.py --logs_base_dir /data/scratch/wzz/tflog \
  --models_base_dir /home/wzz/dev/facenet/models \
  --data_dir /data/face-recognition/casia-webface/CASIA-maxpy_mtcnnpy_182 \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --lfw_dir /data/face-recognition/lfw/lfw_mtcnnpy_160 \
  --optimizer RMSPROP \
  --learning_rate -1 \
  --max_nrof_epochs 80 \
  --keep_probability 0.8 \
  --random_crop \
  --random_flip \
  --learning_rate_schedule_file /home/wzz/dev/facenet/data/learning_rate_schedule_classifier_casia.txt \
  --weight_decay 5e-5 \
  --center_loss_factor 1e-4 \
  --center_loss_alfa 0.9 \
  --gpu_memory_fraction 0.95 \
  --filter_filename data/filtering_metrics.h5 \
  --filter_percentile 90 \
  --filter_min_nrof_images_per_class 2 \

#python facenet_train_classifier.py --logs_base_dir /data/scratch/wzz/tflog --models_base_dir /home/wzz/dev/facenet/models/20170131-234652 --data_dir /data/face-recognition/inke_10k_50pp/inke10k50pp_mtcnnpy_182 --image_size 160 --model_def models.inception_resnet_v1 --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 80 --keep_probability 0.8 --random_crop --random_flip --learning_rate_schedule_file /home/wzz/dev/facenet/data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-4 --center_loss_alfa 0.9 --lfw_dir /data/face-recognition/lfw/lfw_mtcnnpy_160 --pretrained_model /home/wzz/dev/facenet/models/20170215-000945

#python train_copy.py --logs_base_dir /data/scratch/wzz/tflog --models_base_dir /home/wzz/dev/facenet/models/20170131-234652 --data_dir /data/face-recognition/casia-webface/CASIA-maxpy_mtcnnpy_182 --image_size 160 --model_def models.inception_resnet_v1 --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 80 --keep_probability 0.8 --random_crop --random_flip --learning_rate_schedule_file /home/wzz/dev/facenet/data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-4 --center_loss_alfa 0.9
