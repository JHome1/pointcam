# PointNet
python train_semseg.py --model pointnet_sem_seg --test_area 5 --data_split s3dis_1 --open_eval msp --log_dir pointnet_sem_seg_split1_msp
python train_semseg.py --model pointnet_sem_seg --test_area 5 --data_split s3dis_3 --open_eval msp --log_dir pointnet_sem_seg_split3_msp 


# PointNet2
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --data_split s3dis_1 --open_eval msp --log_dir pointnet2_sem_seg_split1_msp
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --data_split s3dis_3 --open_eval msp --log_dir pointnet2_sem_seg_split3_msp
