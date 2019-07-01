# GCNet for Object Detection

By [Yue Cao](http://yue-cao.me), [Jiarui Xu](http://jerryxu.net), [Stephen Lin](https://scholar.google.com/citations?user=c3PYmxUAAAAJ&hl=en), Fangyun Wei, [Han Hu](https://sites.google.com/site/hanhushomepage/).

This repo is a official implementation of ["GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond"](https://arxiv.org/abs/1904.11492) on COCO object detection based on open-mmlab's [mmdetection](https://github.com/open-mmlab/mmdetection).
Many thanks to mmdetection for their simple and clean framework. 

*Update on 2019/07/01*

The code is refactored. 
More results are provided and all configs could be found in `configs/gcnet`.

**Notes**: Both PyTorch official SyncBN and Apex SyncBN have some stability issues. 
During training, mAP may drops to zero and back to normal during last few epochs. 

*Update on 2019/06/03*

GCNet is supported by the official mmdetection repo [here](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet). 
Thanks again for open-mmlab's work on open source projects.

## Introduction

**GCNet** is initially described in [arxiv](https://arxiv.org/abs/1904.11492). Via absorbing advantages of Non-Local Networks (NLNet) and Squeeze-Excitation Networks (SENet),  GCNet provides a simple, fast and effective approach for global context modeling, which generally outperforms both NLNet and SENet on major benchmarks for various recognition tasks.

## Citing GCNet

```
@article{cao2019GCNet,
  title={GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond},
  author={Cao, Yue and Xu, Jiarui and Lin, Stephen and Wei, Fangyun and Hu, Han},
  journal={arXiv preprint arXiv:1904.11492},
  year={2019}
}
```

## Main Results

|  Backbone |       Model      | Backbone Norm |       Heads      |     Context    | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:---------:|:----------------:|:-------------:|:----------------:|:--------------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
|  R-50-FPN |       Mask       |     fixBN     |    2fc(w/o BN)   |        -       |    1x   |    3.9   |        0.453        |      10.6      |  37.3  |   34.2  | [model]()|
|  R-50-FPN |       Mask       |     fixBN     |    2fc(w/o BN)   | GC(c3-c5, r16) |    1x   |    4.5   |        0.533        |      10.1      |  38.5  |   35.1  | [model]()|
|  R-50-FPN |       Mask       |     fixBN     |    2fc(w/o BN)   |  GC(c3-c5, r4) |    1x   |    4.6   |        0.533        |       9.9      |  38.9  |   35.5  | [model]()|
|  R-50-FPN |       Mask       |     fixBN     |    2fc(w/o BN)   |        -       |    2x   |     -    |          -          |        -       |  38.2  |   34.9  | [model]()|
|  R-50-FPN |       Mask       |     fixBN     |    2fc(w/o BN)   | GC(c3-c5, r16) |    2x   |     -    |          -          |        -       |  39.7  |   36.1  | [model]()|
|  R-50-FPN |       Mask       |     fixBN     |    2fc(w/o BN)   |  GC(c3-c5, r4) |    2x   |     -    |          -          |        -       |  40.0  |   36.2  | [model]()|
|  R-50-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   |        -       |    1x   |    3.9   |        0.543        |      10.2      |  37.2  |   33.8  | [model]()|
|  R-50-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   | GC(c3-c5, r16) |    1x   |    4.5   |        0.547        |       9.9      |  39.4  |   35.7  | [model]()|
|  R-50-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   |  GC(c3-c5, r4) |    1x   |    4.6   |        0.603        |       9.4      |  39.9  |   36.2  | [model]()|
|  R-50-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   |        -       |    2x   |    3.9   |        0.543        |      10.2      |  37.7  |   34.3  | [model]()|
|  R-50-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   | GC(c3-c5, r16) |    2x   |    4.5   |        0.547        |       9.9      |  39.7  |   36.0  | [model]()|
|  R-50-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   |  GC(c3-c5, r4) |    2x   |    4.6   |        0.603        |       9.4      |  40.2  |   36.3  | [model]()|
|  R-50-FPN |       Mask       |     SyncBN    | 4conv1fc(SyncBN) |        -       |    1x   |     -    |          -          |        -       |  38.8  |   34.6  | [model]()|
|  R-50-FPN |       Mask       |     SyncBN    | 4conv1fc(SyncBN) | GC(c3-c5, r16) |    1x   |     -    |          -          |        -       |  41.0  |   36.5  | [model]()|
|  R-50-FPN |       Mask       |     SyncBN    | 4conv1fc(SyncBN) |  GC(c3-c5, r4) |    1x   |     -    |          -          |        -       |  41.4  |   37.0  | [model]()|
| R-101-FPN |       Mask       |     fixBN     |    2fc(w/o BN)   |        -       |    1x   |    5.8   |        0.571        |       9.5      |  39.4  |   35.9  | [model]()|
| R-101-FPN |       Mask       |     fixBN     |    2fc(w/o BN)   | GC(c3-c5, r16) |    1x   |    7.0   |        0.731        |       8.6      |  40.8  |   37.0  | [model]()|
| R-101-FPN |       Mask       |     fixBN     |    2fc(w/o BN)   |  GC(c3-c5, r4) |    1x   |    7.1   |        0.747        |       8.6      |  40.8  |   36.9  | [model]()|
| R-101-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   |        -       |    1x   |    5.8   |        0.665        |       9.2      |  39.8  |   36.0  | [model]()|
| R-101-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   | GC(c3-c5, r16) |    1x   |    7.0   |        0.778        |       9.0      |  41.1  |   37.4  | [model]()|
| R-101-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   |  GC(c3-c5, r4) |    1x   |    7.1   |        0.786        |       8.9      |  41.7  |   37.6  | [model]()|
| X-101-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   |        -       |    1x   |    7.1   |        0.912        |       8.5      |  41.2  |   37.3  | [model]()|
| X-101-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   | GC(c3-c5, r16) |    1x   |    8.2   |        1.055        |       7.7      |  42.4  |   38.0  | [model]()|
| X-101-FPN |       Mask       |     SyncBN    |    2fc(w/o BN)   |  GC(c3-c5, r4) |    1x   |    8.3   |        1.037        |       7.6      |  42.9  |   38.5  | [model]()|
| X-101-FPN |   Cascade Mask   |     SyncBN    |    2fc(w/o BN)   |        -       |    1x   |     -    |          -          |        -       |  44.7  |   38.3  | [model]()|
| X-101-FPN |   Cascade Mask   |     SyncBN    |    2fc(w/o BN)   | GC(c3-c5, r16) |    1x   |     -    |          -          |        -       |  45.9  |   39.3  | [model]()|
| X-101-FPN |   Cascade Mask   |     SyncBN    |    2fc(w/o BN)   |  GC(c3-c5, r4) |    1x   |     -    |          -          |        -       |  46.5  |   39.7  | [model]()|
| X-101-FPN | DCN Cascade Mask |     SyncBN    |    2fc(w/o BN)   |        -       |    1x   |     -    |          -          |        -       |  47.1  |   40.4  | [model]()|
| X-101-FPN | DCN Cascade Mask |     SyncBN    |    2fc(w/o BN)   | GC(c3-c5, r16) |    1x   |     -    |          -          |        -       |  47.9  |   40.9  | [model]()|
| X-101-FPN | DCN Cascade Mask |     SyncBN    |    2fc(w/o BN)   |  GC(c3-c5, r4) |    1x   |     -    |          -          |        -       |  47.9  |   40.8  | [model]()|
**Notes:**

- `GC` denotes Global Context (GC) block is inserted after 1x1 conv of backbone. 
- `DCN` denotes replace 3x3 conv with 3x3 Deformable Convolution in `c3-c5` stages of backbone.
- `r4` and `r16` denote ratio 4 and ratio 16 in GC block respectively. 
- Some of models are trained on 4 GPUs with 4 images on each GPU.

## Requirements

- Linux(tested on Ubuntu 16.04)
- Python 3.6+
- PyTorch 1.1.0
- Cython
- [apex](https://github.com/NVIDIA/apex) (Sync BN)

## Install

a. Install PyTorch 1.1 and torchvision following the [official instructions](https://pytorch.org/).

b. Install latest apex with CUDA and C++ extensions following this [instructions](https://github.com/NVIDIA/apex#quick-start). 
The [Sync BN](https://nvidia.github.io/apex/parallel.html#apex.parallel.SyncBatchNorm) implemented by apex is required.

c. Clone the GCNet repository. 

```bash
 git clone https://github.com/xvjiarui/GCNet.git 
```

d. Compile cuda extensions.

```bash
cd GCNet
pip install cython  # or "conda install cython" if you prefer conda
./compile.sh  # or "PYTHON=python3 ./compile.sh" if you use system python3 without virtual environments
```

e. Install GCNet version mmdetection (other dependencies will be installed automatically).

```bash
python(3) setup.py install  # add --user if you want to install it locally
# or "pip install ."
```

Note: You need to run the last step each time you pull updates from github. 
Or you can run `python(3) setup.py develop` or `pip install -e .` to install mmdetection if you want to make modifications to it frequently.

Please refer to mmdetection install [instruction](https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md) for more details.

## Environment

### Hardware

- 8 NVIDIA Tesla V100 GPUs
- Intel Xeon 4114 CPU @ 2.20GHz

### Software environment

- Python 3.6.7
- PyTorch 1.1.0
- CUDA 9.0
- CUDNN 7.0
- NCCL 2.3.5

## Usage

### Train

As in original mmdetection, distributed training is recommended for either single machine or multiple machines.

```bash
./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> [optional arguments]
```

Supported arguments are:

- --validate: perform evaluation every k (default=1) epochs during the training.
- --work_dir <WORK_DIR>: if specified, the path in config file will be replaced.

### Evaluation

To evaluate trained models, output file is required.

```bash
python tools/test.py <CONFIG_FILE> <MODEL_PATH> [optional arguments]
```

Supported arguments are:

- --gpus: number of GPU used for evaluation
- --out: output file name, usually ends wiht `.pkl`
- --eval: type of evaluation need, for mask-rcnn, `bbox segm` would evaluate both bounding box and mask AP. 
