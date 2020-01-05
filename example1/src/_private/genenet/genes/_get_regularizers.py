"""Create L1 and L2 regularizers for convolution kernels and activations.

"""
import metatree as mt
import tensorflow as tf

from typing import Callable, Tuple, Union


# If the log10 of a regularization value is below this cutoff, just return a
# regularization weight of 0
REG_CUTOFF = -10


def get_regularizers(node: Union[mt.Gene, mt.Edge],
                     do_kernel: bool=True,
                     do_activity: bool=True) -> Tuple[Callable, Callable]:
    """Create L1 and L2 regularizers for convolution kernels and activations.

    Args:
        node (mt.Gene): Gene to get regularizers for.
        do_kernel (bool): If True, create a kernel regularizer.
        do_activity (bool): If True, create an activity regularizer.

    Returns:
        kernel_regularizer (Callable): Regularizer for convolution kernels.
        activity_regularizer (Callable): Regularizer for the outputs of a
            convolution layer's activation function.

    """
    def reg_weight(name: str) -> float:
        """Compute the regularization weight from the given hyperparameter name.

        Args:
            name (str): The hyperparameter name.

        Returns:
            (float): The corresponding regularization weight.

        """
        logvalue = node.hyperparam(name)
        return 10**logvalue if logvalue > REG_CUTOFF else 0.

    def sum_regularizer(l1_weight: float, l2_weight: float) \
            -> Union[Callable, None]:
        """Wrapper around tf.contrib.sum_regularizer that still works
        properly if one of the summands has weight 0.

        Args:
            l1_weight (float): L1 regularization weight.
            l2_weight (float): L2 regularization weight.

        Returns:
            (Union[Callable, None]): A regularization function like
                tf.contrib.sum_regularizer, or None if both weights are zero.

        """
        if l1_weight == 0 and l2_weight == 0:
            return None

        if l1_weight == 0:
            return tf.contrib.layers.l2_regularizer(l2_weight)
        elif l2_weight == 0:
            return tf.contrib.layers.l1_regularizer(l1_weight)
        else:
            return tf.contrib.layers.sum_regularizer([
                tf.contrib.layers.l1_regularizer(l1_weight),
                tf.contrib.layers.l2_regularizer(l2_weight)])

    if do_kernel:
        # Gamma terms regularize kernels
        gamma_weights = [reg_weight(name) for name in
                         ['log_gamma1', 'log_gamma2']]
        kernel_regularizer = sum_regularizer(gamma_weights[0], gamma_weights[1])
    else:
        kernel_regularizer = None

    if do_activity:
        # Lambda terms regularize activations
        lambda_weights = [reg_weight(name) for name in ['log_lambda1',
                                                        'log_lambda2']]
        activity_regularizer = sum_regularizer(lambda_weights[0],
                                               lambda_weights[1])
    else:
        activity_regularizer = None

    return kernel_regularizer, activity_regularizer
