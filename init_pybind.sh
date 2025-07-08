#!/bin/bash

# pybind依赖安装脚本，基于notebook 910B云环境，python3.9，cann8.0.0.alpha003
# 910B云环境可直接执行脚本，基础赛道根据自身的设备参考安装。

pip3 install pip==24.1
pip3 install pybind11==2.13.1 numpy==1.24 expecttest
pip3 install tensorflow==2.18.0

# 下载torch 2.5.1
#wget https://download.pytorch.org/whl/cpu/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# 下载torch_npu 2.5.1
#wget https://gitee.com/ascend/pytorch/releases/download/v7.0.0-pytorch2.5.1/torch_npu-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch_npu-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

