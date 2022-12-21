# Content

- [2DUNet Descriptions](#2dunet-descriptions)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
- [Script Parameters](#script-parameters)
- [Training Process](#training-process)
- [Evaluation Process](#evaluation-process)
  - [Evaluation](#evaluation)
- [Model Description](#model-description)
  - [Performance](#performance)
    - [Training Performance](#training-performance)
    - [Inference Performance](#inference-performance)



## 2DUNet Descriptions

2DUNet was proposed in 2015, it is a typoe of neural network that directly consumes 2D images. The u-net archiecture ahiceves very good performance on very different biomedical segmentation applications.  Without strong data augmentations, it only needs very few annotated images and has a ver reasonable trainning time.

[2D Unet Paper](https://arxiv.org/pdf/1505.04597.pdf): Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical image computing and computer-assisted intervention* (pp. 234-241). Springer, Cham.

[Lung segmentation Paper](https://arxiv.org/pdf/2001.11767.pdf): Hofmanninger J, Prayer F, Pan J, Röhrich S, Prosch H, Langs G. Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. European Radiology Experimental. 2020 Dec;4(1):1-3.




## Model Architecture

The 2DUNet segementation network takes n 2D images as input, applies input and feature transformations. BN is introdued before each ReLU. 


## Dataset

Chest CT image dataset （in 2D format）

Dataset size: 1.2G

* Train: 60M, 1713 images and corresponding labels.
* Test: 15M, 429  images and corresponding labels.

## Environment Requirements

- Hardware（Ascend）
  - Prepare hardware environment with Ascend processor.
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

* suwen package

  ```bash
  pip install -r requirements.txt
  pip install ./suwen-1.0.1-py3-none-any.whl
  ```

  

## Quick Start

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
# enter script dir, train PointNet
sh run_train_ascend.sh

# enter script dir, evaluate PointNet
sh run_eval.sh
```



## Script Description

```
.
├── README.md
├── UNet2d.ckpt
├── eval.py
├── eval_log.txt
├── imgs
│   └── 2dUNet.png
├── requirements.txt
├── scripts
│   ├── run_eval.sh
│   └── run_train_ascend.sh
├── src
│   ├── __pycache__
│   │   ├── cross_entropy_with_logits.cpython-37.pyc
│   │   ├── dataset.cpython-37.pyc
│   │   ├── dice_loss.cpython-37.pyc
│   │   └── dice_metric.cpython-37.pyc
│   ├── cross_entropy_with_logits.py
│   ├── dataset.py
│   ├── dice_loss.py
│   └── dice_metric.py
├── suwen-1.0.1-py3-none-any.whl
├── train.py
└── train_log.txt

4 directories, 19 files
```



## Script Parameters

```
Major parameters in train.py are as follows:

--data_path: The absolute full path to the train and evaluation datasets.
--seg_path : The absolute full path to the train and evaluation segmentation labels.
--ckpt_path: The absolute full path to the checkpoint file saved after training.
```

More hyperparamteters can be modified in src/config.py.



## Training Process

* running on Ascend

  ```
  sh run_train_ascend.sh
  ```

  After training, the loss value will be achieved as what in train_log.txt

  The model checkpoint will be saved in the current ckpt directory.
  
  
## Evaluation Process

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```
  sh scripts/run_eval.sh
  ```
  
  You can view the results through the file "eval_log". The accuracy of the test dataset will be as what in eval_log.txt.
  
  

## Model Description

### Performance

#### Training Performance

| Parameters                 |                                               |
| -------------------------- | --------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 24cores; Memory, 96G |
| uploaded Date              | 11/22/2021 (month/day/year)                   |
| MindSpore Version          | 1.3.0                                         |
| Dataset                    | MM-WHS                                        |
| Training Parameters        | epoch=600                                     |
| Optimizer                  | Adam                                          |
| Loss Function              | Softmax Cross Entropy                         |
| outputs                    | probability                                   |
| Loss                       | SoftmaxCrossEntropyWithLogits                 |
| Speed                      | -ms/step-                                     |
| Total time                 | -ms                                           |
| Checkpoint for Fine tuning | 23M (.ckpt file)                              |

#### Inference Performance

| Parameters        |                                               |
| ----------------- | --------------------------------------------- |
| Resource          | Ascend 910; CPU 2.60GHz, 24cores; Memory, 96G |
| uploaded Date     | 11/22/2021 (month/day/year)                   |
| MindSpore Version | 1.3.0                                         |
| Dataset           | MM-WHS                                        |
| batch_size        | 1                                             |
| outputs           | probability                                   |
| Dice              | 75.86%                                        |
