import torch

from mmdet.ops.nms import nms_wrapper


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   Ensemble_Test=False,   # 识别图片时的是否使用多模型集成识别？
                   save_mode=False,
                   roi_feats=None, # 新加入的参数   为了得到预测框所对应的map
                   rois=None,
                   bbox_pred=None,
                   cls_score=None,
                   img_metas=None,
                   mode_name=None,
                   ):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    # 排除背景类之后的剩余类数量
    # size(int) 沿着某个轴计算size
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    #
    if multi_bboxes.shape[1] > 4:
        # 前4个列算作背景类去掉,后边的是物体
        # shape: [multi_bboxes数量,物体类数*4]
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)

    #去除第一列的背景分数, 保留的列是各类物体的分数
    # shape: [multi_bboxes数量,物体类数]
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    ''' 
        # 低分数过滤
        # 过滤掉分数 < score_thr的行
        valid_mask打印：
        tensor([[ True],
                [ True],
                [ True],
                [False],
        shape: [multi_bboxes数量,1]
    '''

    '''
    1、排除score过小的框
    
    '''
    valid_mask = scores > score_thr
    # bboxes对应scores保留行
    # shape: [过滤后保留行数,物体类数*4]
    bboxes = bboxes[valid_mask]

    if score_factors is not None:
        # 默认为空 不会执行 fcos_head.py的时候会用上
        scores = scores * score_factors[:, None]
    # 过滤掉小于score_thr的行
    scores = scores[valid_mask]
    # 保留非零行
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        # bboxes数为0时,直接返回
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # 同一类物体的框重叠过多就会被NMS抑制
    # print()
    # print("------------------------------------bbox_nms.py  1111---------------------------------")
    # print("===multi_bboxes:", multi_bboxes.shape)
    # print("===multi_scores:", multi_scores.shape)
    # print("====roi_feats[0]:",roi_feats[0].shape)
    # print("===roi_feats:",roi_feats.shape)
    # print("===filter_low_score_roi_feats",filter_low_score_roi_feats.shape)
    # print()
    # print("===valid_mask:", valid_mask.shape)
    # print("===bboxes:", bboxes.shape)
    # print("===labels:", labels.shape)
    # print("--------------------------------------------------------------------------------------")
    # print()

    '''
    2、nms抑制

    '''
    # 模型集成性能测试
    if Ensemble_Test == True:
        bboxes,scores,labels = Ensemble_bboxes_union(
            img_metas=img_metas,
            mode_name=mode_name,
            cur_bboxes=bboxes,
            cur_scores=scores,
            cur_labels=labels
        )

        # bboxes, scores, labels = Ensembel_bboxes_intersection(
        #     img_metas=img_metas,
        #     mode_name=mode_name,
        #     cur_bboxes=bboxes,
        #     cur_scores=scores,
        #     cur_labels=labels
        # )


    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    # nms_wrapper就是nms抑制的处理逻辑
    nms_op = getattr(nms_wrapper, nms_type)
    # nms_op：NMS操作(具体注释,输入,输出格式 进入上边一行的nms_wrapper里看)
    # dets是NMS抑制后留下的bbox, keep是保留的行索引
    # dets使用科学计数法
    dets, keep = nms_op(
        torch.cat([bboxes_for_nms, scores[:, None]], 1),  # 这个cat操作将bbox和score按列拼接到一起 (?,4) + (?,1) ---> (?,5)
        **nms_cfg_)
    bboxes = bboxes[keep]
    scores = dets[:, -1]  # soft_nms will modify scores
    labels = labels[keep]




    # 3、框数量超过设定值,则要按照置信度取Top-max_num
    final_bboxes=bboxes # 要使用深拷贝
    final_scores=scores
    final_labels=labels
    top_max_inds = None
    if keep.size(0) > max_num:
        # 保存前 max_num个框
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        top_max_inds = inds

    if save_mode == True:
        # 开启保存模式  --- 自定义函数
        # 保存最后识别的框 和 特征
        save_tensor(   valid_mask=valid_mask,
                       roi_feats=roi_feats,      # 保存
                       rois=rois,                # 保存
                       bbox_pred=bbox_pred,      # 保存
                       cls_score=cls_score,      # 保存

                       bboxes=final_bboxes,      # 保存,为了方便使用自己编写的Ensemble_union 和 Ensemble_intersection函数
                       scores=final_scores,      # 保存,为了方便使用自己编写的Ensemble_union 和 Ensemble_intersection函数
                       labels=final_labels,      # 保存,为了方便使用自己编写的Ensemble_union 和 Ensemble_intersection函数

                       top_max_inds=top_max_inds,
                       img_metas=img_metas,
                       mode_name=mode_name
                       )

    #
    # print()
    # print("------------------------------------bbox_nms.py  2222---------------------------------")
    # print("===max_coordinate:", max_coordinate)
    # print("===offsets:", offsets)
    # print("===bboxes_for_nms:", bboxes_for_nms)
    # print("===nms_cfg_:", nms_cfg_ )
    # print("===nms_type:", nms_type)
    # print("===nms_op:", nms_op)
    # print("--------")
    # print("===dets:", dets.shape, dets)
    # print("===keep(NMS的 inds):", keep.shape, keep)
    # print("--------")
    # print("===scores:", scores.shape, scores)
    # print("===labels:",labels.shape,labels)
    # print("===bboxes:", bboxes.shape,bboxes)
    # print("===final_roi_feats",final_roi_feats.shape)
    # print("===final_rois:",final_rois.shape)
    # print("--------------------------------------------------------------------------------------")
    # print()

    return torch.cat([bboxes, scores[:, None]], 1), labels

def save_tensor(valid_mask = None,  # 过滤掉低分
               rois = None,         # 保存
               roi_feats = None,    # 保存
               bbox_pred = None,    # 保存
               cls_score = None,    # 保存
               bboxes=None,         # 保存
               scores=None,          # 保存
               labels=None,         # 保存

               top_max_inds=None, # 根据参数保留前top_max个框
               img_metas=None,
               mode_name=None
                   ):

    # 自定义函数
    # 保存最后识别的框 和 特征

    '''
    *********此处自己加上的
    # 过滤低分数的框后保留对应的roi_feats、roi
    '''

    i = 0
    # 1、去掉分数过低的框
    idns = [] # 保留此索引所对应的行
    for one_row in valid_mask:
        if(one_row[0] == True):
            idns.append(i)
            i = i + 1
    filter_low_score_roi_feats = roi_feats[idns]
    filter_low_score_rois = rois[idns]
    filter_low_score_bbox_pred = bbox_pred[idns]
    filter_low_score_cls_score = cls_score[idns]
    '''
    *********
    '''

    # 为了创建引用
    final_rois = filter_low_score_rois
    final_roi_feats = filter_low_score_roi_feats
    final_bbox_pred = filter_low_score_bbox_pred
    final_cls_score = filter_low_score_cls_score
    final_bboxes = bboxes
    final_scores = scores
    final_labels = labels

    # 2、去掉NMS抑制不通过的框
    if top_max_inds is not None:
        final_roi_feats = filter_low_score_roi_feats[top_max_inds]
        final_rois = filter_low_score_rois[top_max_inds]
        final_bbox_pred = filter_low_score_bbox_pred[top_max_inds]
        final_cls_score = filter_low_score_cls_score[top_max_inds]
        final_bboxes = final_bboxes[top_max_inds]
        final_scores = final_scores[top_max_inds]
        final_labels = final_labels[top_max_inds]


    # 3、保存经过过滤后的预测框的张量
    # 更好的方案是把这几个张量用list保存到一个.pt张量文件里,读取之时按照下标取对应的张量对象,不然Io可能会很长时间
    # 保存到一个文件里的问题：张量的维度不同...???
    tensor_list = []
    tensor_list.append()

    save_path = "/content/drive/My Drive/detect-tensor/" + mode_name + "/"
    # img_metas[0]['filename']: 类似'/content/mmdetection/data/coco/val2017/Z107.jpg'
    images_name = img_metas[0]['filename'].split("/")[-1].split(".")[0]

    '''
    为了方便融合特征
    '''
    # 保存框对应的rois(rois是用来作为roi_extractor的输入)张量
    save_path = save_path + images_name + "-rois.pt"
    torch.save(final_rois,save_path)
    # 保存roi_extractor的输出张量
    save_path = save_path + images_name + "-roi_feats.pt"
    torch.save(final_roi_feats,save_path)
    # 保存预测框张量
    save_path = save_path + images_name + "-bbox_pred.pt"
    torch.save(final_bbox_pred,save_path)
    # 保存预测框分数
    save_path = save_path + images_name + "-cls_score.pt"
    torch.save(final_cls_score,save_path)

    '''
    为了方便使用自己编写的Ensemble_union 和 Ensemble_intersection函数
    '''
    # 保存bboxes,为了方便使用自己编写的Ensemble_union 和 Ensemble_intersection函数
    save_path = save_path + images_name + "-bboxes.pt"
    torch.save(final_bboxes,save_path)

    save_path = save_path + images_name + "-scores.pt"
    torch.save(final_scores, save_path)

    save_path = save_path + images_name + "-labels.pt"
    torch.save(final_labels,save_path)

def Ensemble_bboxes_union(
                   img_metas=None,
                   mode_name=None,
                   cur_bboxes=None,
                   cur_scores=None,
                   cur_labels=None
                    ):
    '''

    :param img_metas:
    :param mode_name:
    :return: 不同模型检测框的并集,而后再经过NMS抑制最终返回结果
    '''
    save_path = "/content/drive/My Drive/detect-tensor/" + mode_name + "/"

    images_name = img_metas[0]['filename'].split("/")[-1].split(".")[0]
    # 保存框对应的rois(rois是用来作为roi_extractor的输入)张量
    # 1、读取其他模型的tensor
    save_path = save_path + images_name + "-bboxes.pt"
    other_bboxes = torch.load(save_path)

    save_path = save_path + images_name + "-scores.pt"
    other_scores = torch.load(save_path)

    save_path = save_path + images_name + "-labels.pt"
    other_labels = torch.load(save_path)

    # 2、并集融合
    bboxes = torch.cat((cur_bboxes,other_bboxes),0)
    labels = torch.cat((cur_labels, other_labels), 0)
    scores = torch.cat((cur_scores, other_scores), 0)

    return bboxes,scores,labels

def Ensembel_bboxes_intersection(
        img_metas=None,
        mode_name=None,
        cur_bboxes=None,
        cur_scores=scores,
        cur_labels=None):
    '''

     :param img_metas:
     :param mode_name:
     :return: 不同模型检测框的交集,而后再经过NMS抑制最终返回结果
     '''

    save_path = "/content/drive/My Drive/detect-tensor/" + mode_name + "/"

    images_name = img_metas[0]['filename'].split("/")[-1].split(".")[0]
    # 保存框对应的rois(rois是用来作为roi_extractor的输入)张量
    # 1、读取其他模型的tensor
    save_path = save_path + images_name + "-bboxes.pt"
    other_bboxes = torch.load(save_path)

    save_path = save_path + images_name + "-scores.pt"
    other_scores = torch.load(save_path)

    save_path = save_path + images_name + "-labels.pt"
    other_labels = torch.load(save_path)

    # 2、交集融合
    bboxes = torch.cat((cur_bboxes, other_bboxes), 0)
    labels = torch.cat((cur_labels, other_labels), 0)
    scores = torch.cat((cur_scores, other_scores), 0)

    return bboxes, scores, labels