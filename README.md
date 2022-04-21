

Pytorch implementation for DCTNet: Depth-Cooperated Trimodal Network for Video Salient Object Detection [[PDF](https://arxiv.org/pdf/2202.06060.pdf)]

# Requirements

- Python 3.7
- Pytorch 1.6.0
- Torchvision 0.7.0
- Cuda 9.2
- Ubuntu16.04

# Usage

## To Train

### For pretrain different streams

1. Download the pre_trained ResNet34 [backbone](https://download.pytorch.org/models/resnet34-333f7ec4.pth) to './model/resnet/pre_train/'.
2. Organize each dataset according to the organization format in the './dataset/DAVIS'.
3. Download the train dataset from [Baidu Driver]() (PSW: u01t) and save it at `./dataset/train/*`. Our pretraining pipeline consists of three steps:
   - First, train the RGB stream model using the combination of static SOD dataset (i.e., DUTS)  and VSOD datasets (i.e., DAVIS16 & FBMS & DAVSOD).
     - Set `--mode='pretrain_rgb'` and run `python main.py` in terminal
   - Second, train the FLOW stream model using the optical-flow map of VSOD datasets (i.e., DAVIS16 & FBMS & DAVSOD).
     - Set `--mode='pretrain_flow'` and run `python main.py` in terminal
   - Third, train the DEPTH stream model using the depth map of VSOD datasets (i.e., DAVIS16 & FBMS & DAVSOD).
     - Set `--mode='pretrain_depth'` and run `python main.py` in terminal

### For train final network 

Last, the training of final DCTNet is implemented on two NVIDIA Titan X GPUs. Modify the `--spatial_ckpt ` , `--flow_ckpt ` ,  `--depth_ckpt `  with the path of pretrained saved checkpoints.

- Set `--mode='train'` 
- run `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py` in terminal

# Trained model for testing

1. Download the test data (containing DAVIS, DAVSOD, FBMS, SegTrack-V2, VOS) from [Here]() [PSW: ], trained model (DCTNet_model.pth) from [Here]() [PSW: ]
2. Set `--mode='test'` and run `python main.py` in terminal

We also provide the version with different training set of video dataset, including DAVIS and FBMS. MGAN and FSNet are trained and finetuned on the DAVIS and FBMS. The comparsion result is below. Download the trained model(DCTNet_DAFB_model.pth)  

## DAVIS + FBMS 

| Datasets    | Metrics   | MSTM* | STBP* | SFLR* | SCOM* | MGAN      | FSNet     | Ours      | Ours(MIN) |
| ----------- | :-------- | ----- | ----- | ----- | ----- | --------- | --------- | --------- | --------- |
| DAVIS       | maxF      | 0.395 | 0.485 | 0.698 | 0.746 | 0.893     | 0.907     | **0.910** | 0.913     |
|             | S-measure | 0.566 | 0.651 | 0.771 | 0.814 | 0.913     | 0.920     | **0.922** | 0.926     |
|             | MAE       | 0.174 | 0.105 | 0.060 | 0.055 | 0.022     | 0.020     | **0.014** | 0.013     |
| DAVSOD      | maxF      | 0.347 | 0.408 | 0.482 | 0.473 | 0.662     | **0.685** | 0.678     | 0.691     |
|             | S-measure | 0.530 | 0.563 | 0.622 | 0.603 | 0.757     | **0.773** | 0.771     | 0.777     |
|             | MAE       | 0.214 | 0.165 | 0.136 | 0.219 | 0.079     | **0.072** | 0.077     | 0.071     |
| FBMS        | maxF      | 0.500 | 0.595 | 0.660 | 0.797 | **0.909** | 0.888     | 0.892     | 0.899     |
|             | S-measure | 0.613 | 0.627 | 0.699 | 0.794 | **0.912** | 0.890     | 0.905     | 0.907     |
|             | MAE       | 0.177 | 0.152 | 0.117 | 0.079 | **0.026** | 0.041     | 0.031     | 0.026     |
| SegTrack-V2 | maxF      | 0.526 | 0.640 | 0.745 | 0.764 | **0.840** | 0.806     | 0.826     | 0.831     |
|             | S-measure | 0.643 | 0.735 | 0.804 | 0.815 | **0.895** | 0.870     | 0.887     | 0.887     |
|             | MAE       | 0.114 | 0.061 | 0.037 | 0.030 | **0.024** | 0.025     | 0.032     | 0.033     |
| VOS         | maxF      | 0.567 | 0.526 | 0.546 | 0.690 | 0.743     | 0.659     | **0.751** | 0.761     |
|             | S-measure | 0.657 | 0.576 | 0.624 | 0.712 | 0.807     | 0.703     | **0.815** | 0.826     |
|             | MAE       | 0.144 | 0.163 | 0.145 | 0.162 | **0.069** | 0.103     | 0.071     | 0.062     |

The version with different training set of video dataset, including DAVIS and DAVSOD. SSAV, PCSA and TENet are trained and finetuned on the DAVIS and FBMS. The comparsion result is below. Download the trained model(DCTNet_DADS_model.pth)  

## DAVIS + DAVSOD 

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



### For evaluation:

1. The saliency maps can be download in 
2. Evaluation Toolbox: We use the standard evaluation toolbox from [DAVSOD benchmark](https://github.com/DengPingFan/DAVSOD).