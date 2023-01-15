# CrossPoint_reappear
The codes are tested on Ubuntu 16.04, PyTorch 1.9 and Python 3.8.
## [Reference github](https://github.com/MohamedAfham/CrossPoint) | [Original Paper Link](https://arxiv.org/abs/2203.00680) | [Original Project Page](https://mohamedafham.github.io/CrossPoint/) 
## Data Dependency
Datasets are available [here](https://drive.google.com/drive/folders/1dAH9R3XDV0z69Bz6lBaftmJJyuckbPmR?usp=sharing), you can download them, then put them in data directory. Alternatively, Run the command below to download all the datasets (ShapeNetRender, ModelNet40, ScanObjectNN, ShapeNetPart) to reproduce the results.
```
cd data
source download_data.sh
```
Data file structure should look like this:
```
./
├── train_crosspoint_update.py
├── train_crosspoint.py
├── train_partseg.py
├── ...
└── data/
    ├── ShapeNetRendering
    ├── ScanObjectNN
    ├── ...
    └── download_data.sh
```
## Usage
sign up a wandb account so that you can easily visualize the output.
### Our revised code: 
```
python train_crosspoint_update.py --model dgcnn --epochs 100 --lr 0.001 --exp_name crosspoint_revise --batch_size 32 --print_freq 200 --k 15
```
You can arbitrarily choose ```exp_name```, and the output will automatically save in wandb and checkpoints directory. See code in details.

### Traning CrossPoint for classification
```
python train_crosspoint.py --model dgcnn --epochs 100 --lr 0.001 --exp_name crosspoint_dgcnn_cls --batch_size 20 --print_freq 200 --k 15
```
### Training CrossPoint for part-segmentation
```
python train_crosspoint.py --model dgcnn_seg --epochs 100 --lr 0.001 --exp_name crosspoint_dgcnn_seg --batch_size 20 --print_freq 200 --k 15
```
### Fine-tuning for part-segmentation
```
python train_partseg.py --exp_name dgcnn_partseg --pretrained_path ./models/dgcnn_partseg_best.pth --batch_size 8 --k 40 --test_batch_size 8 --epochs 300
```
