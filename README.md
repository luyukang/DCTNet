

# DCTNet: Depth-Cooperated Trimodal Network for Video Salient Object Detection

This repository provides PyTorch code implementation for DCTNet: Depth-Cooperated Trimodal Network for Video Salient Object Detection [[Arxiv](https://arxiv.org/pdf/2202.06060.pdf)]

## Requirements

- Python 3.7
- Pytorch 1.6.0
- Torchvision 0.7.0
- Cuda 9.2
- Ubuntu16.04

## Usage

### Training

#### Pretrain RGB, Flow and Depth streams

1. Download the pre_trained ResNet34 [backbone](https://download.pytorch.org/models/resnet34-333f7ec4.pth) to './model/resnet/pre_train/'.
2. Download the train dataset (containing DAVIS16, DAVSOD, FBMS and DUTS-TR) from [Baidu Driver]() (PSW: ) and save it at './dataset/train/*'. 
3. Following the the instruction of [RAFT](https://github.com/princeton-vl/RAFT) to prepare the optical flow and the instruction of [DPT](https://github.com/isl-org/DPT) to prepare the synthetic depth map.
4. Organize each dataset according to the organization format in the './dataset/train/DAVIS/'.
5. Our pretraining pipeline consists of three steps:
   - First, train the RGB stream model using the combination of static SOD dataset (i.e., DUTS)  and VSOD datasets (i.e., DAVIS16 & FBMS & DAVSOD).
     - Set `--mode='pretrain_rgb'` and run `python main.py` in terminal
   - Second, train the Flow stream model using the optical flow map of VSOD datasets (i.e., DAVIS16 & FBMS & DAVSOD).
     - Set `--mode='pretrain_flow'` and run `python main.py` in terminal
   - Third, train the Depth stream model using the depth map of VSOD datasets (i.e., DAVIS16 & FBMS & DAVSOD).
     - Set `--mode='pretrain_depth'` and run `python main.py` in terminal

#### Train the entire network 

Modify the `--spatial_ckpt ` ,  `--flow_ckpt ` ,  `--depth_ckpt `  to the path of pretrained saved checkpoints. 

Last, the training of entire DCTNet is implemented on two NVIDIA TiTAN X GPUs. 

- Set `--mode='train'` 
- run `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py` in terminal

## Testing

1. Download the test data (containing DAVIS, DAVSOD, FBMS, SegTrack-V2, VOS) from [Baidu Driver](https://pan.baidu.com/s/1u1qOWkv5WbovwWKogXwZQw) (PSW: 8uh3) and save it at './dataset/test/*'
2. Download trained model from [Baidu Driver](https://pan.baidu.com/s/1Z8Sut8bOGOwbUBf0Tmhm4w) (PSW: lze1) and modify the  `model_path` to its saving path in the `main.py`.
3. Set `--mode='test'` and run `python main.py` in terminal

We also provide the version with different training set of video dataset, including DAVIS + FBMS, DAVIS + DAVSOD. 

### DAVIS + FBMS 

Models with "*" are traditional methods, MGAN and FSNet are trained and finetuned on the DAVIS and FBMS. The comparison result is below. Download the trained model from [Baidu Driver]()

| Datasets    | Metrics   | MSTM* | STBP* | SFLR* | SCOM* | MGAN      | FSNet | Ours      |
| ----------- | :-------- | ----- | ----- | ----- | ----- | --------- | ----- | --------- |
| DAVIS       | maxF      | 0.395 | 0.485 | 0.698 | 0.746 | 0.893     | 0.907 | **0.913** |
|             | S-measure | 0.566 | 0.651 | 0.771 | 0.814 | 0.913     | 0.920 | **0.926** |
|             | MAE       | 0.174 | 0.105 | 0.060 | 0.055 | 0.022     | 0.020 | **0.013** |
| DAVSOD      | maxF      | 0.347 | 0.408 | 0.482 | 0.473 | 0.662     | 0.685 | **0.691** |
|             | S-measure | 0.530 | 0.563 | 0.622 | 0.603 | 0.757     | 0.773 | **0.777** |
|             | MAE       | 0.214 | 0.165 | 0.136 | 0.219 | 0.079     | 0.072 | **0.071** |
| FBMS        | maxF      | 0.500 | 0.595 | 0.660 | 0.797 | **0.909** | 0.888 | 0.899     |
|             | S-measure | 0.613 | 0.627 | 0.699 | 0.794 | **0.912** | 0.890 | 0.907     |
|             | MAE       | 0.177 | 0.152 | 0.117 | 0.079 | **0.026** | 0.041 | **0.026** |
| SegTrack-V2 | maxF      | 0.526 | 0.640 | 0.745 | 0.764 | **0.840** | 0.806 | 0.831     |
|             | S-measure | 0.643 | 0.735 | 0.804 | 0.815 | **0.895** | 0.870 | 0.887     |
|             | MAE       | 0.114 | 0.061 | 0.037 | 0.030 | **0.024** | 0.025 | 0.033     |
| VOS         | maxF      | 0.567 | 0.526 | 0.546 | 0.690 | 0.743     | 0.659 | **0.761** |
|             | S-measure | 0.657 | 0.576 | 0.624 | 0.712 | 0.807     | 0.703 | **0.826** |
|             | MAE       | 0.144 | 0.163 | 0.145 | 0.162 | 0.069     | 0.103 | **0.062** |

### DAVIS + DAVSOD 

PCSA and TENet are trained and finetuned on the DAVIS and FBMS. The comparison result is below. Download the trained model from [Baidu Driver]()

| Datasets    | Metrics   | MSTM* | STBP* | SFLR* | SCOM* | SSAV  | PCSA  | TENet | Ours      |
| ----------- | :-------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --------- |
| DAVIS       | maxF      | 0.395 | 0.485 | 0.698 | 0.746 | 0.861 | 0.880 | 0.894 | **0.904** |
|             | S-measure | 0.566 | 0.651 | 0.771 | 0.814 | 0.893 | 0.902 | 0.905 | **0.917** |
|             | MAE       | 0.174 | 0.105 | 0.060 | 0.055 | 0.028 | 0.022 | 0.021 | **0.016** |
| DAVSOD      | maxF      | 0.347 | 0.408 | 0.482 | 0.473 | 0.603 | 0.656 | 0.648 | **0.695** |
|             | S-measure | 0.530 | 0.563 | 0.622 | 0.603 | 0.724 | 0.741 | 0.753 | **0.778** |
|             | MAE       | 0.214 | 0.165 | 0.136 | 0.219 | 0.092 | 0.086 | 0.078 | **0.069** |
| FBMS        | maxF      | 0.500 | 0.595 | 0.660 | 0.797 | 0.865 | 0.837 | 0.887 | 0.883     |
|             | S-measure | 0.613 | 0.627 | 0.699 | 0.794 | 0.879 | 0.868 | 0.910 | 0.886     |
|             | MAE       | 0.177 | 0.152 | 0.117 | 0.079 | 0.040 | 0.040 | 0.027 | 0.032     |
| SegTrack-V2 | maxF      | 0.526 | 0.640 | 0.745 | 0.764 | 0.798 | 0.811 | **    | **0.839** |
|             | S-measure | 0.643 | 0.735 | 0.804 | 0.815 | 0.851 | 0.866 | **    | **0.886** |
|             | MAE       | 0.114 | 0.061 | 0.037 | 0.030 | 0.023 | 0.024 | **    | **0.014** |
| VOS         | maxF      | 0.567 | 0.526 | 0.546 | 0.690 | 0.742 | 0.747 | **    | **0.772** |
|             | S-measure | 0.657 | 0.576 | 0.624 | 0.712 | 0.819 | 0.828 | **    | **0.837** |
|             | MAE       | 0.144 | 0.163 | 0.145 | 0.162 | 0.074 | 0.065 | **    | **0.058** |



## For evaluation:

1. The saliency maps can be download from [Baidu Driver](https://pan.baidu.com/s/10i5ADy4iSSwydy04Enf27w) (PSW: wfqc)
2. Evaluation Toolbox: We use the standard evaluation toolbox from [DAVSOD benchmark](https://github.com/DengPingFan/DAVSOD).