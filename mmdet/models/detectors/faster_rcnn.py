from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class FasterRCNN(TwoStageDetector):#继承了双阶段的检测器TwoStageDetector，MASKRCNN也继承了它

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 mode_name="FasterRCNN"
                 ):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            mode_name="FasterRCNN"
        )
