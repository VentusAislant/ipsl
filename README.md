Image Patch Semantic Learning

### Install
- create env
    ```shell
    # 创建conda虚拟环境
    conda create -n ipsl python=3.10 -y
    # 激活虚拟环境
    conda activate ipsl
    
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
  
- prepare data and pretrained models
```shell
coco_path="/home/wind/Disks/16t/Datasets/CV/COCO/"
open_ai_cit_path="/home/wind/Disks/16t/Models/vision_models/openai/"
llm_path="/home/wind/Disks/16t/Models/LLMs/vicuna/vicuna-7b-v1.5/"
mask2former_path="/home/wind/Disks/16t/Models/vision_models/mask2former/"
swin_transformer_path="/home/wind/Disks/16t/Models/vision_models/swin_transformer/"
mkdir data
mkdir pretrained_models
mkdir work_dirs

ln -s $coco_path ./data
ln -s $open_ai_cit_path ./pretrained_models
ln -s $llm_path ./pretrained_models
ln -s $mask2former_path ./pretrained_models
ln -s $swin_transformer_path ./pretrained_models
 
```