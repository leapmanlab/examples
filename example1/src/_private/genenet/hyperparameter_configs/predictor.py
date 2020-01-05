"""Create a HyperparameterConfig for class predictor genes.

"""
import metatree as mt
import tensorflow as tf

from .values import ACTIVATION_FUNCTIONS, PADDING_TYPES, POOLING_TYPES


def predictor() -> mt.HyperparameterConfig:
    """Create a HyperparameterConfig for a PredictorGene.

    Returns:
        (mt.HyperparameterConfig): HyperparameterConfig for a class
            prediction operation.

    """
    hyperparam_dict = {
        # Spatial Mode (0=2D, 1=3d with kxkxk filters, 2=3d with 1xkxk filters)
        'spatial_mode': 0,
        # Network architecture hyperparameters
        # If True, pooling layers pool along the z axis, too
        'pool_z': False,
        # Dilation Rate
        'dilation_rate': 1,
        # Number of convolution layers in a convolution block
        'n_comps': 2,
        # Number of convolution blocks in a blockset
        'n_blocks': 2,
        # Data spatial scale
        'spatial_scale': 0,
        # Computation hyperparameters
        # Use batch normalization?
        'batch_norm': False,
        # Convolution kernel support width (in all spatial dimensions)
        'k_width': 3,
        # Number of convolution kernels
        'n_kernels': 64,
        # Padding type used by convolution operations
        'padding_type': ('valid', PADDING_TYPES),
        # Operation used by pooling layers
        'pooling_type': ('maxpool', POOLING_TYPES),
        # Layer activation function
        'f_activate': (tf.nn.relu, ACTIVATION_FUNCTIONS),
        # log10 of L1 regularization weight for layer activations
        'log_lambda1': -11.,
        # log10 of L2 regularization weight for layer activations
        'log_lambda2': -11.,
        # log10 of L1 regularization weight for layer kernels
        'log_gamma1': -11.,
        # log10 of L2 regularization weight for layer kernels
        'log_gamma2': -11.,

    }

    return mt.HyperparameterConfig(hyperparam_dict)
