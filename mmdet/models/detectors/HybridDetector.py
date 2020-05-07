import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
import torch
import torch.nn as nn

from __future__ import division

import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin

from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector

@DETECTORS.register_module
class HybridDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(HybridDetector, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.bbox_head = builder.build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(HybridDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)

        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)

        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)

        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)

        losses = dict()

        '''
            获取其他模型的rois,roi_feats,bbox_pred,cls_score
        '''
        other_rois, other_roi_feats, other_bbox_pred, other_cls_score = self.Ensemble_load_tensor(img_metas)


        # bbox head forward and loss
        if self.with_bbox:

            cls_score, bbox_pred = self.bbox_head(other_roi_feats)
            # sampling_results？？？？？？？？
            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)


        return losses

    def Ensemble_load_tensor(self,
                             img_metas=None
                             ):
        '''
        载入其他模型最终识别出来的框所对应的几个张量 rois,roi_feats,bbox_pred,cls_score

        :param img_metas:
        :return:
        '''
        save_path = "/content/drive/My Drive/detect-tensor/" + "FasterRCNN" + "/"

        images_name = img_metas[0]['filename'].split("/")[-1].split(".")[0]
        # 保存框对应的rois(rois是用来作为roi_extractor的输入)张量
        save_path = save_path + images_name + "-rois.pt"
        rois = torch.load(save_path)
        # 保存roi_extractor的输出张量
        save_path = save_path + images_name + "-roi_feats.pt"
        roi_feats = torch.load(save_path)
        # 保存预测框张量
        save_path = save_path + images_name + "-bbox_pred.pt"
        bbox_pred = torch.load(save_path)
        # 保存预测框分数
        save_path = save_path + images_name + "-cls_score.pt"
        cls_score = torch.load(save_path)
        return rois,roi_feats,bbox_pred,cls_score

    def simple_test(self, img, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        other_rois, other_roi_feats, other_bbox_pred, other_cls_score = self.Ensemble_load_tensor(img_metas)
        rois = other_rois
        roi_feats = other_roi_feats

        ###
        cls_score, bbox_pred = self.bbox_head(roi_feats)

        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        rcnn_test_cfg = self.test_cfg.rcnn

        ###
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,               # 原来的参数 必须保证load的.pt文件和这个维度一致，所以这里也需要保存后load
            cls_score,          # 原来的参数 必须保证load的.pt文件和这个维度一致，所以这里也需要保存后load
            bbox_pred,          # 原来的参数 必须保证load的.pt文件和这个维度一致，所以这里也需要保存后load
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg,
            )

        # 返回一个列表,不是tensor
        # 用浮点数表示，且是一个列表
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
