# Robust Positive-Unlabeled Learning via Self-selection of Unlabeled Negatives
This repo contains the official **PyTorch** code for Robust-PU.

## Introduction

<img src="figures/pipline.png" alt="pipline" style="zoom: 25%;" />

We utilize a novel "hardness" measure to distinguish unlabeled samples with a high chance of being negative from unlabeled samples with large label noise. An iterative training strategy is then implemented to fine-tune the selection of negative samples during the training process in an iterative manner to include more "easy" samples in the early stage of training.

## Usage

### Requirements

Python 3.9

1. ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   ```

2. ```bash
   pip install -r requirements.txt
   ```

### Experiments

To train the model on MNIST with prior 0.6:

```bash
python main.py --dataset mnist --prior 0.6 --pre_lr 1e-3 --pre_epochs 100 --pre_batch_size 128 --pre_wd 1e-4 --lr 1e-2 --inner_epochs 20 --wd 0 --batch_size 64 --scheduler_type_n linear --alpha_n 0.11 --max_thresh_n 1.0 --grow_steps_n 5 --temper_n 1.3 --scheduler_type_p linear --alpha_p 0.1 --max_thresh_p 1.0 --grow_steps_p 5 --temper_p 1.0 --hardness logistic
```

`${scheduler_type_n}`: `const`, `linear`, `convex`, `concave`, `exp`

`${scheduler_type_p}`: `const`, `linear`, `convex`, `concave`, `exp`

`${hardness}`: `logistic`, `sigmoid`

## Contact

If you have any question, please feel free to contact the authors. Zhangchi Zhu: zczhu@stu.ecnu.edu.cn

## Citation

If you find our work is useful in your research, please consider citing:

