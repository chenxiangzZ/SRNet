# SRNet
Pose-guided Part Matching Network via Shrinking and Reweighting for Occluded Person
Re-identification(Accepted)

### env setting
```
conda create -n srnet python=3.7
conda activate srnet
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
```
GPU Memmory >= 10G, Memory >= 20G(LINUX with CUDN)

pay attention to the pretrianed model
```
r50_ibn_a.pth : core/models/model_reid.py line 71
pose_hrnet_w48_256x192.pth: core/models/model_keypoints/config/default.py line 133
```
if you cannot download from the Internet, mail me!
### Update
2021-03-01: We will open source when paper is accepted.  
2021-05-05：Happy News, Our paper is accepted.  
2021-11-04：update env setting.  


## Contacts
If you have any question about the project, please feel free to contact me.

E-mail: 417545906@qq.com

