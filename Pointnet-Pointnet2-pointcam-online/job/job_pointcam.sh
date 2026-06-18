# PointNet
python train_semseg.py --model pointnet_sem_seg_pointcam --test_area 5 --data_split s3dis_1 --use_cutmix --use_pointcam --open_eval pointcam --log_dir pointnet_sem_seg_split1_pointcam --select_ratio 0.6 --alpha 5.0
python train_semseg.py --model pointnet_sem_seg_pointcam --test_area 5 --data_split s3dis_3 --use_cutmix --use_pointcam --open_eval pointcam --log_dir pointnet_sem_seg_split3_pointcam --select_ratio 0.6 --alpha 5.0


# PointNet2
python train_semseg.py --model pointnet2_sem_seg_pointcam --test_area 5 --data_split s3dis_1 --use_cutmix --use_pointcam --open_eval pointcam --log_dir pointnet2_sem_seg_split1_pointcam --select_ratio 0.6 --alpha 5.0
python train_semseg.py --model pointnet2_sem_seg_pointcam --test_area 5 --data_split s3dis_3 --use_cutmix --use_pointcam --open_eval pointcam --log_dir pointnet2_sem_seg_split3_pointcam --select_ratio 0.6 --alpha 5.0
