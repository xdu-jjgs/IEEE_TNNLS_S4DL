from .classifier import ImageClassifier
from .extractor import RevFeatureExtractor

__all__ = [
    ImageClassifier,
    RevFeatureExtractor
]


def build_backbone(num_channels, model_name):
    if model_name == 'fe_rev':
        return RevFeatureExtractor(num_channels)
    raise NotImplementedError('invalid model: {}'.format(model_name))
