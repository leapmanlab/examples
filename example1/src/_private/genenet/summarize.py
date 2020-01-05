"""Simple TensorFlow Tensor summary control for GeneNet computation graphs.

"""
import tensorflow as tf

from .util import colorize

from typing import Any, Dict, Optional


def image_summary(t: tf.Tensor,
                  params: Dict[str, Any],
                  name: Optional[str]=None):
    """Create an image summary of a tensor with spatial structure, such as a
    convolutional tensor.

    Args:
        t (tf.Tensor): A TensorFlow Tensor.
        params (Dict[str, Any]): Image summary generation parameters. Keys:
            cmap (str): A valid cmap named for use with matplotlib's
                `get_cmap`. Default is 'gray'.
            data_format (str): Tensor data format, either 'channels_first'
                (convolutional feature channels are along axis 1) or
                'channels_last' (convolutional feature channels are along
                axis -1). Default is 'channels_first'
            idxs_batch (Sequence[int]): Indices along the tensor minibatch
                axis (axis 0) indicating which minibatch elements to make
                image summaries for. Default is [0].
            idxs_feature (Sequence[int]): Indices along the tensor feature
                channel axis indicating which feature channels to make image
                summaries for. Default is [0].
            idxs_x (Sequence[int]): Indices along the tensor x spatial axis
                indicating which yz planar slices to make image summaries for.
                Ignored if input tensor has 2D spatial structure. Default is [].
            idxs_y (Sequence[int]): Indices along the tensor y spatial axis
                indicating which  xz planar slices to make image summaries for.
                Ignored if input tensor has 2D spatial structure. Default is []
            idxs_z (Sequence[int]): Indices along the tensor z spatial axis
                indicating which xy planar slices to make image summaries for.
                Ignored if input tensor has 2D spatial structure. Default is
                [0].
            vmax (float): the maximum value of the range used for
                normalization. (Default: image maximum).
            vmin (float): the minimum value of the range used for
                normalization. (Default: image minimum).
        name (Optional[str]): A name for the generated node. Will also serve
            as a series name in TensorBoard. Default is t.name.

    Returns: None

    """
    # Default name if none is supplied
    if name is None:
        name = t.name
    # Get parameter values
    cmap = 'gray'
    if 'cmap' in params:
        cmap = params['cmap']
    data_format = 'channels_first'
    if 'data_format' in params:
        data_format = params['data_format']
    # Make sure we received a valid value
    if data_format not in ['channels_first', 'channels_last']:
        raise ValueError(f'Received data format {data_format}, value must be'
                         f'\'channels_first\' or \'channels_last\'.')
    idxs_batch = [0]
    if 'idxs_batch' in params:
        idxs_batch = params['idxs_batch']
    idxs_feature = [0]
    if 'idxs_feature' in params:
        idxs_feature = params['idxs_feature']
    idxs_x = []
    if 'idxs_x' in params:
        idxs_x = params['idxs_x']
    idxs_y = []
    if 'idxs_y' in params:
        idxs_y = params['idxs_y']
    idxs_z = [0]
    if 'idxs_z' in params:
        idxs_z = params['idxs_z']
    vmax = None
    if 'vmax' in params:
        vmax = params['vmax']
    vmin = None
    if 'vmin' in params:
        vmin = params['vmin']

    # Determine whether the tensor has a 2D or 3D spatial structure.
    ndims = t.get_shape().ndims
    if ndims == 3:
        # 2D, and also missing a dimension
        is_3d = False
        with tf.name_scope('add_dim'):
            t = tf.expand_dims(t, 0, name='add_dim')
    elif ndims == 4:
        is_3d = False
    elif ndims == 5:
        is_3d = True
    else:
        raise ValueError(f'Input tensor has rank {ndims}, must be 3, 4, or 5.')

    # Determine whether the tensor has channels first
    channels_first = data_format == 'channels_first'

    def log_image(s_, name_, vmin_, vmax_, cmap_):
        """Create a colored image, log it with a TF summary.

        Args:
            s_ (tf.Tensor): 2D tensor slice.
            name_: A name for the generated node. Will also serve as a series
                name in TensorBoard.
            vmin_: See parent function arg `params` key 'vmin'.
            vmax_: See parent function arg `params` key 'vmax'.
            cmap_: See parent function arg `params` key 'cmap'.

        Returns: None

        """
        with tf.name_scope('summaries'):
            # Apply a color map, then append a singleton batch dimension out
            # front for tf.summary.image
            with tf.name_scope('colorize'):
                color_image = tf.expand_dims(colorize(s_, vmin_, vmax_, cmap_),
                                             axis=0)
            # Create the summary
            tf.summary.image(name_, color_image, max_outputs=100)

        pass

    for bi in idxs_batch:
        for fi in idxs_feature:
            if is_3d:
                # Create an image for each entry in `idxs_z`, `idxs_x`,
                # and `idxs_y`. Assumes tensor spatial layout is ZXY
                for zi in idxs_z:
                    if channels_first:
                        s = t[bi, fi, zi, :, :]
                    else:
                        s = t[bi, zi, :, :, fi]
                    # Give the slice a unique summary name
                    slice_name = f'{name}_b{bi}_f{fi}_z{zi}'
                    # Create and log the slice image
                    log_image(s, slice_name, vmin, vmax, cmap)

                for xi in idxs_x:
                    if channels_first:
                        s = t[bi, fi, :, xi, :]
                    else:
                        s = t[bi, :, xi, :, fi]
                    # Give the slice a unique summary name
                    slice_name = f'{name}_b{bi}_f{fi}_x{xi}'
                    # Create and log the slice image
                    log_image(s, slice_name, vmin, vmax, cmap)

                for yi in idxs_y:
                    if channels_first:
                        s = t[bi, fi, :, :, yi]
                    else:
                        s = t[bi, :, :, yi, fi]

                    # Give the slice a unique summary name
                    slice_name = f'{name}_b{bi}_f{fi}_y{yi}'
                    log_image(s, slice_name, vmin, vmax, cmap)

            else:
                # Tensor has a 2D spatial structure, there is just one image
                # possible given batch and feature indices
                if channels_first:
                    s = t[bi, fi, :, :]
                else:
                    s = t[bi, :, :, fi]
                # Give the slice a unique summary name
                slice_name = f'{name}_b{bi}_f{fi}'
                # Create and log the slice image
                log_image(s, slice_name, vmin, vmax, cmap)

    pass


def variable_summary(t: tf.Tensor):
    """Attach variable value summaries to a tensor for TensorBoard
    visualization.

    Args:
        t (tf.Tensor): A TensorFlow Tensor.

    Returns: None

    """
    with tf.name_scope('summaries'):
        # mean = tf.reduce_mean(t)
        # tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(t - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(t))
        # tf.summary.scalar('min', tf.reduce_min(t))
        tf.summary.histogram('histogram', t)
