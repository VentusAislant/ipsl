from typing import Tuple, List, Dict
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.maskformer import MaskFormer


@MODELS.register_module()
class IpslMask2Former(MaskFormer):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        losses = self.panoptic_head.loss(x, batch_data_samples, batch_inputs)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        feats = self.extract_feat(batch_inputs)
        mask_cls_results, mask_pred_results = self.panoptic_head.predict(
            feats, batch_data_samples, batch_inputs)
        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results,
            mask_pred_results,
            batch_data_samples,
            rescale=rescale)
        results = self.add_pred_to_datasample(batch_data_samples, results_list)

        return results

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            tuple[List[Tensor]]: A tuple of features from ``panoptic_head``
            forward.
        """
        feats = self.extract_feat(batch_inputs)  # backbone -> neck
        results = self.panoptic_head.forward(feats, batch_data_samples, batch_inputs)
        return results
