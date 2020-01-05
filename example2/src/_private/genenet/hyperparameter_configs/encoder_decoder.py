"""Create a HyperparameterConfig for encoder decoder networks.

"""
import metatree as mt
import tensorflow as tf

from .values import ACTIVATION_FUNCTIONS, PADDING_TYPES, POOLING_TYPES


def encoder_decoder() -> mt.HyperparameterConfig:
    """Create a HyperparameterConfig for building and training
    encoder-decoder networks for EM image segmentation.

    Specifies a collection of hyperparameter keys and values, as well as root
    and descendant feasible ranges for each hyperparameter - see
    metatree.HyperparameterConfig for more information on hyperparameters.

    Returns:
        (mt.HyperparameterConfig): HyperparameterConfig for an
            encoder-decoder segmentation network.

    """
    # Create a dict of hyperparameters and their feasible ranges
    hyperparam_dict = {

        # Network architecture hyperparameters
        # Spatial mode (0=2d, 1=3d with kxkxk filters, 2=3d with 1xkxk filters)
        'spatial_mode': 0,
        # If True, pooling layers pool along the z axis, too
        'pool_z': False,
        # Dilation Rate
        'dilation_rate': 1,
        # Number of convolution kernels in a convolution layer
        'n_kernels': 64,
        # Number of convolution layers in a convolution block
        'n_comps': 2,
        # Number of convolution blocks in a blockset
        'n_blocks': 2,
        # Data spatial scale
        'spatial_scale': 0,

        # Computation hyperparameters
        # Padding type used by convolution operations
        'padding_type': ('valid', PADDING_TYPES),
        # Operation used by pooling layers
        'pooling_type': ('maxpool', POOLING_TYPES),
        # Layer activation function
        'f_activate': (tf.nn.relu, ACTIVATION_FUNCTIONS),
        # Use batch normalization?
        'batch_norm': False,
        # Convolution kernel support width (in all spatial dimensions)
        'k_width': 3,

        # Regularization hyperparameters
        # log10 of layer activation L1 regularization hyperparameter
        'log_lambda1': -11.,
        # log10 of layer activation L2 regularization hyperparameter
        'log_lambda2': -11.,
        # log10 of convolution kernel L1 regularization hyperparameter
        'log_gamma1': -11.,
        # log10 of convolution kernel L2 regularization hyperparameter
        'log_gamma2': -11.
    }

    return mt.HyperparameterConfig(hyperparam_dict)
