# PointNet
python train_semseg.py --model pointnet_sem_seg --test_area 5 --data_split s3dis_1 --use_cutmix --open_eval msp --log_dir pointnet_sem_seg_split1_cutmix --select_ratio 0.6
python train_semseg.py --model pointnet_sem_seg --test_area 5 --data_split s3dis_3 --use_cutmix --open_eval msp --log_dir pointnet_sem_seg_split3_cutmix --select_ratio 0.6


# PointNet2
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --data_split s3dis_1 --use_cutmix --open_eval msp --log_dir pointnet2_sem_seg_split1_cutmix --select_ratio 0.6
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --data_split s3dis_3 --use_cutmix --open_eval msp --log_dir pointnet2_sem_seg_split3_cutmix --select_ratio 0.6
