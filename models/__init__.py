from .s4dl import S4DL

from configs import CFG
from models.backbone import build_backbone, ImageClassifier


def build_model(num_channels, num_classes):
    if CFG.MODEL.BACKBONE:
        backbone_ = build_backbone(num_channels, CFG.MODEL.BACKBONE)
    else:
        raise NotImplementedError('invalid backbone: {} or experts: {}'.format(CFG.MODEL.BACKBONE, CFG.MODEL.EXPERTS))
    # build model
    if CFG.MODEL.NAME == 's4dl':
        return S4DL(num_classes, backbone_, CFG.HYPERPARAMS)
    raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))
