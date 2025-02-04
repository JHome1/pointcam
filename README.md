# PointCaM
This repository is for PointCaM introduced in the following paper:  

**PointCaM: Cut-and-Mix for Open-Set Point Cloud Learning**  
Jie Hong*, Shi Qiu*, Weihao Li, Saeed Anwar, Mehrtash Harandi, Nick Barnes, Lars Petersson
## Paper and Citation
The paper can be downloaded from [arXiv](https://arxiv.org/abs/2212.02011).  
If you find our paper/code is useful, please cite:

        @article{hong2023pointcam,
          title={PointCaM: Cut-and-Mix for Open-Set Point Cloud Learning},
          author={Hong, Jie and Qiu, Shi and Li, Weihao and Anwar, Saeed and Harandi, Mehrtash and Barnes, Nick and Petersson, Lars},
          journal={arXiv preprint arXiv:2212.02011},
          year={2023}
        }
        
## Datasets and Environments
* PointTransformer: \
Download the datasets and set the environments following the project [```point-transformer```](https://github.com/POSTECH-CVLab/point-transformer). The experiments are running on 4 NVIDIA GeForce RTX 3090.

## Running the Code
### PointTransformer:
* Training \
Name ```exp``` of ```exp_dir``` in ```./tool/train.sh```; \
Configure ```data_root```, ```test_list```, ```test_list_full```, and ```names_path``` in ```./config/s3dis/s3dis_pointtransformer_repro.yaml```; \
Configure ```cutmix```, ```data_split```, ```open_eval```, ```alpha```, and ```select_ratio``` in ```./config/s3dis/s3dis_pointtransformer_repro.yaml```. 
```
sh train.sh
```

* Testing \
Name ```exp``` of ```exp_dir``` in ```./tool/test.sh```; \
MSP and MaxLogits methods share the same training process; \
Even ```open_eval``` is set to "msp" in the training stage, \
configure ```open_eval``` to "maxlogit" for evaluating MaxLogits. 
```
sh test.sh
```
