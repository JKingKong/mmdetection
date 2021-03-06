import logging
import sys

import torch

from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, merge_aug_proposals, multiclass_nms)

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class RPNTestMixin(object):

    if sys.version_info >= (3, 7):

        async def async_test_rpn(self, x, img_metas, rpn_test_cfg):
            sleep_interval = rpn_test_cfg.pop('async_sleep_interval', 0.025)
            async with completed(
                    __name__, 'rpn_head_forward',
                    sleep_interval=sleep_interval):
                rpn_outs = self.rpn_head(x)

            proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)

            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
            return proposal_list

    def simple_test_rpn(self, x, img_metas, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(imgs_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, rpn_test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals


class BBoxTestMixin(object):

    if sys.version_info >= (3, 7):

        async def async_test_bboxes(self,
                                    x,
                                    img_metas,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False,
                                    bbox_semaphore=None,
                                    global_lock=None):
            """Async test only det bboxes without augmentation."""
            rois = bbox2roi(proposals)
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)

            async with completed(
                    __name__, 'bbox_head_forward',
                    sleep_interval=sleep_interval):
                cls_score, bbox_pred = self.bbox_head(roi_feats)

            img_shape = img_metas[0]['img_shape']
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            return det_bboxes, det_labels

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           mode_name=None,
                           save_mode=False,
                           ):
        ####====================================================================================
        """Test only det bboxes without augmentation."""
        '''
        此处可以载入其他模型的roi、roi_feats  进行特征加入融合
        '''
        rois = bbox2roi(proposals)
        # 调用 mmdet/models/roi_extractors/single_level.py 下得到
        # 1、得到roi_feats
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)

        # 利用roi_feats得到cls_score, bbox_pred
        # mmdet/models/bbox_heads/convfc_bbox_head.py  下的forward方法的返回值 分类过后的分数,回归框后的参数
        # cls_score的shape: box数 * 2 (第0列是背景分数,会被直接忽略)
        # bbox_pred的shape：box数 * 8
        '''
        此处可以载入其他模型的roi_feats 进行 特征加入融合
        '''
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']

        # 载入roi_feats张量(已完成小score过滤,nms抑制)
        # root_path = "/content/mmdetection/"
        # picture_name = "Z108"
        # save_path = root_path + picture_name + "_filter_final_roi_feats.pt"
        # roi_feats = torch.load(save_path)
        # save_path = root_path + picture_name + "_filter_final_rois.pt"
        # rois = torch.load(save_path)
        # save_path = root_path + picture_name + "_filter_final_bbox_pred.pt"
        # bbox_pred = torch.load(save_path)
        # save_path = root_path + picture_name + "_filter_final_cls_score.pt"
        # cls_score = torch.load(save_path)

        # det_bboxes的shape:NMS抑制后的数量 * 5
        '''
        # print(img_metas)
        此处可以传入img_metas:
        [{
            'filename': '/content/mmdetection/data/coco/val2017/Z107.jpg',
            'ori_shape': (720, 1280, 3),
            'img_shape': (750, 1333, 3),
            'pad_shape': (768, 1344, 3),
            'scale_factor': 1.04140625,
            'flip': False,
            'img_norm_cfg': {
                'mean': array([123.675, 116.28, 103.53], dtype = float32),
                'std': array([58.395, 57.12, 57.375], dtype = float32),
                'to_rgb': True
            }
        }]
        '''
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,               # 原来的参数 必须保证load的.pt文件和这个维度一致，所以这里也需要保存后load
            cls_score,          # 原来的参数 必须保证load的.pt文件和这个维度一致，所以这里也需要保存后load
            bbox_pred,          # 原来的参数 必须保证load的.pt文件和这个维度一致，所以这里也需要保存后load
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg,
            save_mode=save_mode,
            roi_feats=roi_feats,  # 新加入的参数   为了得到预测框所对应的特征图
            img_metas=img_metas,
            mode_name=mode_name
            )
        # print()
        # print("===================****************=====================")
        # print("--- current function from ", sys._getframe().f_code.co_filename)
        # print("--- current function is      ", sys._getframe().f_code.co_name)
        # print()
        # print("--- called from file           ", sys._getframe().f_back.f_code.co_filename)
        # print("--- called by function      ", sys._getframe().f_back.f_code.co_name)
        # print("--- called at line               ", sys._getframe().f_back.f_lineno)
        # print("===================****************=====================")
        # print()


        # print()
        # print("--------------------------------test_mixins.py------------------------------------------------------")
        # print("===roi_feats:",roi_feats.shape)
        # print()
        # print("===cls_score:",cls_score.shape)
        # print("===bbox_pred:",bbox_pred.shape)
        # print()
        # print("===det_bboxes:",det_bboxes.shape)
        # print("===det_labels:",det_labels.shape)
        # print("--------------------------------------------------------------------------------------")
        # print()
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    if sys.version_info >= (3, 7):

        async def async_test_mask(self,
                                  x,
                                  img_metas,
                                  det_bboxes,
                                  det_labels,
                                  rescale=False,
                                  mask_test_cfg=None):
            # image shape of the first image in the batch (only one)
            ori_shape = img_metas[0]['ori_shape']
            scale_factor = img_metas[0]['scale_factor']
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head.num_classes - 1)]
            else:
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)

                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                if mask_test_cfg and mask_test_cfg.get('async_sleep_interval'):
                    sleep_interval = mask_test_cfg['async_sleep_interval']
                else:
                    sleep_interval = 0.035
                async with completed(
                        __name__,
                        'mask_head_forward',
                        sleep_interval=sleep_interval):
                    mask_pred = self.mask_head(mask_feats)
                segm_result = self.mask_head.get_seg_masks(
                    mask_pred, _bboxes, det_labels, self.test_cfg.rcnn,
                    ori_shape, scale_factor, rescale)
            return segm_result

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                                       det_labels,
                                                       self.test_cfg.rcnn,
                                                       ori_shape, scale_factor,
                                                       rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                mask_pred = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.test_cfg.rcnn)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg.rcnn,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result
