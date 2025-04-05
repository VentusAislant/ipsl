export CUDA_VISIBLE_DEVICES=0
export COCO_DATASET="data/COCO/"

type="mask2former"

CFG_BASE="configs/models/${type}"

cfg_name="mask2former_r50_8xb2-lsj-50e_coco-panoptic"
#cfg_name="mask2former_r50_8xb2-lsj-50e_coco"
#cfg_name="mask2former_r101_8xb2-lsj-50e_coco"
#cfg_name="mask2former_r101_8xb2-lsj-50e_coco-panoptic"
#cfg_name="mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic"
#cfg_name="mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic"
#cfg_name="mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic"
#cfg_name="mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic"
#cfg_name="mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco"
#cfg_name="mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic"
#cfg_name="mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco"


ckpt_path="pretrained_models/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.pth"

cfg_path="${CFG_BASE}/${cfg_name}.py"

work_dir="work_dirs/test/${type}/${cfg_name}"

python -m torch.distributed.launch --nproc_per_node=gpu \
    ipsl/test.py $cfg_path $ckpt_path \
      --launcher pytorch \
      --cfg-options \
        work_dir=$work_dir \
        test_dataloader.batch_size=10