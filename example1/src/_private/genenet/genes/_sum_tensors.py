"""Sum tensors elementwise, despite possibly mismatched shape.

Perform a center spatial crop along spatial axes to get the spatial shapes
matching, then sum lazily along the convolution kernel axis.

"""
import tensorflow as tf

from typing import Sequence


def sum_tensors(tensors: Sequence[tf.Tensor]) -> tf.Tensor:
    """Sum a collection of tensors, but in a very permissive way to
    accommodate mismatched tensor shapes.

    Tensors have shape [1, c, x, y] (2D spatial shape) or [1, c, z, x, y]
    (3D spatial shape), where c indicates the size of the convolution kernel
    axis, i.e. the number of convolution features in the Tensor. The following
    exposition assumes a 3D spatial shape, but the same principles apply in
    2D. Given a collection of n tensors [T1, T2, ..., Tn] such that
    shape[Ti] = [1, ci, zi, xi, yi], define zm = min(z1, z2, ..., zn),
    xm = min(x1, x2, ..., xn), ym = min(y1, y2, ..., yn). Center crop all
    tensors along their spatial dimensions to get a collection of tensors
     S1, S2, ..., Sn] such that shape(Si) = [1, ci, zm, xm, ym].

    Handling mismatched spatial dimensions is easy, since we always crop to
    the smallest size present. Handling mismatched convolution kernel
    dimensions is trickier. Instead, specify a target number of kernels in
    the sum with the parameter `n_kernels_out` := max(c1, ..., cn),
    let S = Si+ Sj, where shape(Si) = [1, ci, zm, xm, ym] and
    shape(Sm) = [1, cj, zm, xm, ym]. If ci < cj, then S[..., 0:ci] = Si + Sj[
    ..., 0:ci] and S[..., ci:] = Sj[..., ci:]. If ci > cj, then S = Si[...,
    0:cj] + Sj.

    Args:
        tensors (Sequence[tf.Tensor]): A collection of tensors to sum.

    Returns:
        (tf.Tensor): Single Tensor containing the summed input Tensors.

    """
    # No need to sum anything with just one Tensor
    if len(tensors) == 1:
        return tensors[0]

    # Channels-first format
    kernel_axis = 1
    # Sort tensors by convolution kernel axis size - largest to smallest
    tensors = sorted(tensors,
                     key=lambda t: t.get_shape().as_list()[kernel_axis],
                     reverse=True)
    # Create a list of Tensor shapes
    shapes = [t.get_shape().as_list() for t in tensors]
    # Create a list of Tensor ranks
    ranks = [len(shape) for shape in shapes]
    # Input Tensors must have the same ranks
    if min(ranks) != max(ranks):
        raise AttributeError('All input Tensors must have the same rank')
    rank = ranks[0]

    # Max kernel axis size needs to be at least the target output kernel axis
    # size
    n_kernels_out = shapes[0][kernel_axis]


    # Perform center crop
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

    # Perform kernel axis summation

    n_kernels = [s[kernel_axis] for s in shapes]

    # TODO: Make this use kernel_axis instead of hardcoding it
    # First Tensor has size >= n_kernels_out, so truncate it
    tensor_sum = crops[0][:, 0:n_kernels_out, ...]
    # Keep adding
    for k, t in zip(n_kernels[1:], crops[1:]):
        if k >= n_kernels_out:
            # More kernels than kernels out. Truncate
            tensor_sum += t[:, 0:n_kernels_out, ...]
        else:
            # Fewer kernels than kernels out. Partial addition. Since Tensors
            # don't support item assignment, accomplish this with concatenation
            tensor_sum = tf.concat([tensor_sum[:, 0:k, ...] + t,
                                    tensor_sum[:, k:, ...]], axis=kernel_axis)

    return tensor_sum
