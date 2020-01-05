"""Object values for HyperparameterConfigs built by the
genenet.hyperparameter_configs module.

"""
import tensorflow as tf

# Possible activation functions
ACTIVATION_FUNCTIONS = [tf.nn.relu,
                        tf.nn.relu6,
                        tf.nn.crelu,
                        tf.nn.elu,
                        tf.nn.softplus,
                        tf.nn.tanh,
                        tf.nn.sigmoid,
                        tf.nn.leaky_relu,
                        tf.identity]

# Convolution padding types
PADDING_TYPES = ['valid', 'same']

# Pooling types
POOLING_TYPES = ['maxpool',
                 'conv']
