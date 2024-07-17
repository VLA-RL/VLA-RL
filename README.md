# VLA-RL
UofT ECE 1508 Reinforcement Learning Ropo

## Guides

<!-- - Getting Started: [Installation](#installation), [Quickstart](#quickstart), [Checkpoints and Pre-Generated Datasets](#download), [Model Card](model-card.md)
- Data Generation: [Data Generation](#data-generation)
- Training & Evaluation: [Multi-Task Training and Evaluation](#training-and-evaluation), [Gotchas](#gotchas)
- Miscellaneous: [Recording Videos](#recording-videos), [Notebooks](#notebooks), [Disclaimers](#disclaimers-and-limitations), [FAQ](#faq), [Docker Guide](#docker-guide), [Licenses](#licenses)
- Acknowledgements: [Acknowledgements](#acknowledgements), [Citations](#citations) -->


## Installation

### Prerequisites

<!-- PerAct is built-off the [ARM repository](https://github.com/stepjam/ARM) by James et al. The prerequisites are the same as ARM.  -->

#### 1. Environment

<!-- ```bash
# setup a virtualenv with whichever package manager you prefer
virtualenv -p $(which python3.8) --system-site-packages VLA-RL-env
source peract_env/bin/activate
pip install --upgrade pip
``` -->

#### 2. PyRep and Coppelia Simulator

<!-- Follow instructions from the official [PyRep](https://github.com/stepjam/PyRep) repo; reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/previousVersions#)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd <install_dir>
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -r requirements.txt
pip install .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

If you encounter errors, please use the [PyRep issue tracker](https://github.com/stepjam/PyRep/issues). -->

#### 3. RLBench

<!-- VLA-RL uses my [RLBench fork](https://github.com/VLA-RL/RLBench/tree/VLA-RL). 

```bash
cd <install_dir>
git clone -b VLA-RL https://github.com/VLA-RL/RLBench.git # note: 'VLA-RL' branch

cd RLBench
pip install -r requirements.txt
python setup.py develop
```

For [running in headless mode](https://github.com/MohitShridhar/RLBench/tree/peract#running-headless), tasks setups, and other issues, please refer to the [official repo](https://github.com/stepjam/RLBench). -->

### VLA-RL Repo
Clone:
```bash
cd <install_dir>
git clone https://github.com/VLA-RL/VLA-RL.git
```

Install:
```bash
cd VLA-RL
pip install -r requirements.txt

export VLA_RL_ROOT=$(pwd)  # mostly used as a reference point for tutorials
# python setup.py develop
```


<!-- **Note**: You might need versions of `torch==1.7.1` and `torchvision==0.8.2` that are compatible with your CUDA and hardware. Later versions should also be fine (in theory).  -->

## Quick Start

## Download

### Pre-Trained Checkpoints

#### [OpenVLA - 7B](https://huggingface.co/openvla/openvla-7b)

```bash
huggingface-cli download openvla/openvla-7b --local-dir $MODEL_ROOT/openvla-7b
```

## Data generation

```bash
python rlbench_data_generator/dataset_generator.py \
                      --save_path '/your/savepath' \
                      --task 'pick_up_cup' \
                      --image_size 224 224 \
                      --renderer 'opengl' \ 
                      --processes 3 \
                      --episodes_per_task 3 \
                      --variations 10
```

```bash
python rlbench_data_generator/data_processing.py
```

## Download pretrained model



## Training and Evaluation

### Finetuning

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 finetune/finetune_l1.py
```

#### TODO list
- [x] ActionTokenizer (trans, sin(theta) rotation)
- [x] argmax is not differentiable, change to softmax @ decode
- [x] accumulation steps
- [x] train, test, valid split
- [ ] Split trans, Rotation, grip loss
- [ ] chain of thought
- [ ] RL 

### Experiments

1. single task vs multi tasks
2. accumulation steps
3. different quantization config
4. chain of thought
5. few shot rl (num of demo)

#### Abalation Expoeriments




## Citations 

<!-- **OpenVLA**
```
@article{kim2024openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and Xiao, Ted and Balakrishna, Ashwin and Nair, Suraj and Rafailov, Rafael and Foster, Ethan and Lam, Grace and Sanketi, Pannag and others},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}
```

**PerAct**
```
@inproceedings{shridhar2022peract,
  title     = {Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
  year      = {2022},
}
```

**RLBench**
```
@article{james2020rlbench,
  title={Rlbench: The robot learning benchmark \& learning environment},
  author={James, Stephen and Ma, Zicong and Arrojo, David Rovick and Davison, Andrew J},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={3019--3026},
  year={2020},
  publisher={IEEE}
}
``` -->

