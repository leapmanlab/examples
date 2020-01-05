"""Merge tensors by concatenating a collection of them along their
convolution kernel axis.

"""
import tensorflow as tf

from typing import Optional, Sequence


def merge_tensors(tensors: Sequence[tf.Tensor],
                  name: Optional[str]=None) -> tf.Tensor:
    """Merge tensors by concatenating a collection of them along
    their convolution kernel axis

    Args:
        tensors (Sequence[tf.Tensor]): A collection of Tensors.
        name (str): Operation name.

    Returns:
        (tf.Tensor): Single Tensor containing the merged input Tensors.

    """
    # No need to sum anything with just one Tensor
    if len(tensors) == 1:
        return tensors[0]

    # Create a list of Tensor shapes
    shapes = [t.get_shape().as_list() for t in tensors]
    # Create a list of Tensor ranks
    ranks = [len(shape) for shape in shapes]
    # Input Tensors must have the same ranks
    if min(ranks) != max(ranks):
        raise AttributeError('All input Tensors must have the same rank')
    # Since they all have the same rank
    rank = ranks[0]
    # Index of the feature axis
    kernel_axis = 1

    # Tensors may not all have the same spatial size, so we need to perform a
    # center crop. Compute its size
    crop_shape = []
    for i in range(rank):
        if i in [0, kernel_axis]:
            # Keep everything along the feature axis
            crop_shape.append(-1)
        else:
            # Keep the smallest shape along the i^th axis.
            crop_shape.append(min(shape[i] for shape in shapes))
    # Perform cropping as necessary
    crops = []
    for (tensor, shape) in zip(tensors, shapes):
        if shape != crop_shape:
            # Compute the starting point for the center crop along the
            # spatial dimensions
            starts = [0]
            for i in range(1, rank):
                if i == kernel_axis:
                    # Keep everything, start at the start
                    starts.append(0)
                else:
                    start = (shape[i] - crop_shape[i]) // 2
                    starts.append(start)
            # Append the appropriate slice of the Tensor to crops
            crops.append(tf.slice(tensor, starts, crop_shape))
        else:
            # Shapes match, no cropping necessary
            crops.append(tensor)

    # Concatenate the Tensors along the feature axis
    return tf.concat(crops, kernel_axis, name=name)
