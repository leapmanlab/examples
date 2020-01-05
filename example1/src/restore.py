"""Restore a trained LCIMB segmentation model.

"""

from .GeneNet import GeneNet


def restore(model_dir: str) -> GeneNet:
    """Restore a trained model from a saved model dir.

    Args:
        model_dir (str):

    Returns:
        (GeneNet): The restored model

    """
    net = GeneNet.restore(model_dir, build_now=True)
    return net
