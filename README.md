Image Patch Semantic Learning

### Install
- create env
```shell
# 创建conda虚拟环境
conda create -n ispl python=3.10 -y
# 激活虚拟环境
conda activate ispl

# 安装最新的 torch 和 torchvision
pip install torch torchvision
# 验证torch安装是否成功
python -c 'import torch;print(torch.__version__)'  # 2.6.0+cu124

# 安装 mmengine
# pip install mmengine-lite  # 安装轻量级版本，只有fileio、registry 和 config 模块
pip install mmengine
# 验证mmebgine安装是否成功
python -c 'import mmengine;print(mmengine.__version__)'  # 0.10.7

# 安装 mmcv, mmdet, mmpretrain
pip install mmcv==2.1.0 mmdet mmpretrain
pip install git+https://github.com/cocodataset/panopticapi.git.
```