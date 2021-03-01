# SRNet
Pose-guided Part Matching Network via Shrinking and Reweighting for Occluded Person
Re-identification

### Update
2021-03-01: Code is available.



### Set Up
```shell script
conda create -n srnet python=3.7
conda activate srnet
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
# GPU Memory >= 10G, Memory >= 20G
```


### Preparation
* Dataset: Occluded DukeMTMC-reID ([Project](https://github.com/lightas/Occluded-DukeMTMC-Dataset))
* Pre-trained Pose Model ([pose_hrnet_w48_256x192.pth](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC), 
please download it to path ```./core/models/model_keypoints/pose_hrnet_w48_256x192.pth```)


### Trained Model 
* BaiDuDisk (comming soon)
* Google Drive (comming soon)

### Train
```
python main.py --mode train \
--dataset_path path/to/occluded/duke \
--output_path  ./results   \
--train_dataset duke
```

### Test with Trained Model
```
!python main.py --mode test \
--dataset_path path/to/occluded/duke \
--output_path ./results   \
--resume_test_path  ./results/models   \
--train_dataset duke \
--resume_test_epoch   119
```

## Contacts
If you have any question about the project, please feel free to contact me.

E-mail: 417545906@qq.com

