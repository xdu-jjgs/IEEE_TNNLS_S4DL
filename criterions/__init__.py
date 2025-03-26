from .bce import BCELoss, SigmoidBCELoss
from .ce import CELoss, SoftmaxCELoss
from .decomposed import OrthogonalDecomposedLoss


def build_criterion(name):
    if name == 'ce':
        criterion = CELoss()
    elif name == 'softmax+ce':
        criterion = SoftmaxCELoss()
    elif name == 'bce':
        criterion = BCELoss()
    elif name == 'sigmoid+bce':
        criterion = SigmoidBCELoss()
    elif name == 'orthogonal':
        return OrthogonalDecomposedLoss()
    else:
        raise NotImplementedError('invalid criterion: {}'.format(name))
    return criterion
