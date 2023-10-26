# Diversifying Spatial-Temporal Perception for Video Domain Generalization
PyTorch implementation of the NeurIPS-2023 paper titled "Diversifying Spatial-Temporal Perception for Video Domain Generalization". 

## Environments
- Python: 3.7.3
- PyTorch: 1.3.1

## Training
#### Data preparation
First, create a new directory named "data". Then put the video datasets (video frames) into the directory according to the lists in the directory "datalists". Each list file consists of lines in the form of (path, video_len, label_id) or (path, video_start_idx, video_end_idx, label_id). 

#### Run training scripts
```
# For the UCF->HMDB benchmark
bash scripts/train_ucfhmdb.sh
```

## Acknowledgement 
- This repository is based on the codebases [zhoubolei/TRN-pytorch](https://github.com/zhoubolei/TRN-pytorch) and [mit-han-lab/temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module). 

- If you find this paper/code useful, please consider citing our paper:
```
@inproceedings{lin2023diversify,
  author       = {Kun-Yu Lin and Jia-Run Du and Yipeng Gao and Jiaming Zhou and Wei-Shi Zheng},
  title        = {Diversifying Spatial-Temporal Perception for Video Domain Generalization},
  booktitle    = {NeurIPS 2023},
  year         = {2023},
}
```
