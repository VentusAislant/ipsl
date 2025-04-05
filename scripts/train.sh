export CUDA_VISIBLE_DEVICES=0
export COCO_DATASET="data/COCO/"

export TORCH_DISTRIBUTED_DEBUG="INFO"

BATCH_SIZE=1

type="mask2former_ipsl"

CFG_BASE="configs/models/${type}"
cfg_name="mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"
#cfg_name="mask2former_r50_8xb2-lsj-50e_coco.py"
#cfg_name="mask2former_r101_8xb2-lsj-50e_coco.py"
#cfg_name="mask2former_r101_8xb2-lsj-50e_coco-panoptic.py"
#cfg_name="mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic.py"
#cfg_name="mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic.py"
#cfg_name="mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.py"
#cfg_name="mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic.py"
#cfg_name="mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"
#cfg_name="mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic.py"
#cfg_name="mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco.py"
cfg_path="${CFG_BASE}/${cfg_name}"

work_dir="work_dirs/train/${type}/${cfg_name}"

python -m torch.distributed.launch --nproc_per_node=gpu \
    ipsl/train.py $cfg_path \
      --launcher pytorch \
      --cfg-options \
        work_dir=$work_dir \
        train_dataloader.batch_size=$BATCH_SIZE \
        val_dataloader.batch_size=$BATCH_SIZE \
        test_dataloader.batch_size=$BATCH_SIZE