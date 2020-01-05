"""GeneNet class for building TensorFlow computation graphs from GeneGraphs.

"""
import json
import logging
import os
import pickle
import random
import time

import genenet as gn
import metatree as mt
import numpy as np
import tensorflow as tf

from sklearn.metrics.cluster import adjusted_rand_score
from tensorflow.python.client import device_lib

from genenet.genes import ConvolutionGene
from genenet.summarize import image_summary
from genenet.summarize import variable_summary
from genenet._words import WORDS

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, \
    Union


# Image settings type: To record an image summary of a Gene's output tensor,
# create a tuple where the first element is the Gene, and the second element
# is a dictionary of image summary parameters as defined in the `params` arg
# of `genenet.summarize.image_summary`. Additional special tensors can be
# summarized using special string keys: 'input', 'classes',
# and 'probabilities', which are associated with the input tensor,
# class prediction tensor, and class probability tensor respectively
ImageSettings = Tuple[Union[str, mt.Gene], Dict[str, Any]]
# Dictionary of string keys mapping to mt.HyperparameterConfig objects
HyperparamConfigs = Dict[str, mt.HyperparameterConfig]

logger = logging.getLogger('genenet.GeneNet')

# Name for an eval history JSON file
_EVAL_FILE = 'file_history.json'
# Name for a pickled GeneGraph copy
_GENEGRAPH_FILE = 'gene_graph.pkl'


class GeneNet(gn.GeneNet):
    def model_fn(self,
                 features: Dict[str, tf.Tensor],
                 labels: Dict[str, tf.Tensor],
                 mode: tf.estimator.ModeKeys,
                 params: Dict[str, Any]) -> tf.estimator.EstimatorSpec:
        """Model creation function for a GeneNet segmentation network.

        Args:
            features (Dict[str, tf.Tensor]): Dictionary of input Tensors.
            labels (Dict[str, tf.Tensor]): Dictionary of label Tensors.
            mode (tf.estimator.ModeKeys): Estimator mode.
            params (Dict[str, Any]): Additional model hyperparameters.

        Returns:
            (tf.estimator.EstimatorSpec): GeneNet network EstimatorSpec.
        """
        logger.debug(f'Creating a model_fn on device {self.device}')
        with tf.device(self.device):
            # Get batch size from input shape
            batch_size = tf.shape(features['input'])[0]

            # Build from the GeneGraph. `tensor_map` maps `self.gene_graph`
            # Genes, as well as the special strings 'input', 'classes',
            # and 'probabilities' to TensorFlow Tensors.
            with tf.variable_scope(self.name):
                tensor_map = self.gene_graph.build(features, mode)

            # Update the trainable parameter count
            self.n_trainable_params = int(
                np.sum([np.prod(v.get_shape().as_list())
                        for v in tf.trainable_variables(self.name)]))

            if params['make_summaries']:
                # If True, attach variable summaries to each ConvolutionGene
                for key in tensor_map:
                    if isinstance(key, ConvolutionGene):
                        variable_summary(tensor_map[key])

            # Get the output from the last gene in the gene graph's `genes`
            # OrderedDict
            output_gene = list(self.gene_graph.genes.values())[-1]
            n_classes = output_gene.n_classes
            # Logits are the output of the final PredictorGene in the GeneGraph
            logits = tensor_map[output_gene]

            logger.debug(f'Received logits with shape {logits.get_shape()}')
            if mode != tf.estimator.ModeKeys.PREDICT:
                logger.debug(f'Received labels with shape '
                             f'{labels["label"].get_shape()}')

            with tf.name_scope('classes'):
                classes = tf.argmax(input=logits, axis=1, name='classes')
                tensor_map['classes'] = classes

            with tf.name_scope('probabilities'):
                probabilities = tf.nn.softmax(logits,
                                              axis=1,
                                              name='probabilities')
                tensor_map['probabilities'] = probabilities

            # Both predictions (for PREDICT and EVAL modes)
            predictions = {'classes': classes,
                           'probabilities': probabilities}

            # Create summary ops for the tensor associated with each Gene or
            # str in params['image_settings'].
            image_settings: Sequence[ImageSettings] = []
            if 'image_settings' in params:
                image_settings = params['image_settings']
            for key, summary_params in image_settings:
                # Get the tensor associated with each key
                target_tensor = tensor_map[key]
                # Create an image summary
                image_summary(target_tensor, summary_params)

            # For a forward pass, no need to build optimization ops
            if mode == tf.estimator.ModeKeys.PREDICT:

                # Add the corner feature to predictions, for easy reconstruction
                # of large images from patches in PREDICT mode
                predictions['corner'] = features['corners']

                # Create an EstimatorSpec
                spec = tf.estimator.EstimatorSpec(mode=mode,
                                                  predictions=predictions)

                logger.debug('Created PREDICT EstimatorSpec')

                return spec

            # Calculate loss: per-voxel weighted cross-entropy
            with tf.name_scope('loss'):
                # Cross-entropy from logits. Note the transpose to convert
                # channels-first data into channels-last data

                if mode == tf.estimator.ModeKeys.TRAIN:
                    # During training, use per-voxel cross-entropy weighting
                    # plus regularization terms

                    c_lr_2d = params['c_lr_2d']

                    # Get the output from 'predictor_2d' Gene
                    predictor_2d_gene = list(self.gene_graph.genes.values())[1]
                    logits_2d = tensor_map[predictor_2d_gene]

                    losses = []
                    loss_weights = [1, c_lr_2d]

                    for l, logit in enumerate([logits, logits_2d]):

                        # Experiment-specific tweak: add a loss term from the 2d
                        # predictor as well

                        if logit.get_shape().ndims == 4:
                            xentropy = \
                                tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=labels['label'],
                                    logits=tf.transpose(logit,
                                                        [0, 2, 3, 1],
                                                        name=f'transpose_{l}'),
                                    name=f'softmax_xentropy_{l}')
                        else:
                            xentropy = \
                                tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=labels['label'],
                                    logits=tf.transpose(logit, [0, 2, 3, 4, 1],
                                                        name=f'transpose_{l}'),
                                    name=f'softmax_xentropy_{l}')

                        # Impose a weight floor
                        # Get weight floor
                        weight_floor = features['weight_floor'][0]
                        # Treat zeroed areas differently - they shouldn't be
                        # included in loss calculations
                        weight = labels['weight']
                        nonzero_weights = tf.cast(tf.not_equal(weight, 0),
                                                  dtype=weight.dtype)
                        weight += tf.multiply(nonzero_weights, weight_floor)
                        weights = tf.add(labels['weight'], weight_floor)
                        weighted_xentropy = tf.multiply(
                            weights,
                            xentropy,
                            name=f'weighted_xentropy_{l}')
                        # Sum voxel loss values
                        losses.append(loss_weights[l] * tf.reduce_sum(
                            weighted_xentropy,
                            name=f'sum_xentropy_{l}'))

                    loss = tf.math.add_n(
                        losses,
                        name='total_xentropy')
                    # Add regularization terms, weighted to ignore zeroed-out
                    # areas
                    frac_nonzero = tf.reduce_mean(nonzero_weights)
                    loss += frac_nonzero * tf.losses.get_regularization_loss()

                else:
                    # Experiment-specific tweak: add a loss term from the 2d
                    # predictor as well

                    if logits.get_shape().ndims == 4:
                        xentropy = \
                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=labels['label'],
                                logits=tf.transpose(logits,
                                                    [0, 2, 3, 1],
                                                    name='transpose'),
                                name='softmax_xentropy')
                    else:
                        xentropy = \
                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=labels['label'],
                                logits=tf.transpose(logits, [0, 2, 3, 4, 1],
                                                    name='transpose'),
                                name='softmax_xentropy')

                    # For eval, use per-voxel cross entropy summed across all
                    # voxels
                    loss = tf.reduce_sum(xentropy, name='sum_xentropy')

            # Build training op
            if mode == tf.estimator.ModeKeys.TRAIN:
                # Get training hyperparameters
                learning_rate = 10 ** params['log_learning_rate']
                decay_steps = 10 ** params['log_decay_steps']
                exponential_decay_rate = params['exponential_decay_rate']
                beta1 = 1 - 10 ** params['log_alpha1']
                beta2 = 1 - 10 ** params['log_alpha2']
                epsilon = 10 ** params['log_epsilon']

                with tf.name_scope('train'):
                    lr = tf.train.exponential_decay(
                        learning_rate=learning_rate,
                        global_step=tf.train.get_global_step(),
                        decay_steps=decay_steps,
                        decay_rate=exponential_decay_rate,
                        staircase=True)
                    optimizer = tf.train.AdamOptimizer(learning_rate=lr,
                                                       beta1=beta1,
                                                       beta2=beta2,
                                                       epsilon=epsilon,
                                                       name='adam')
                    train_op = optimizer.minimize(
                        loss=loss,
                        global_step=tf.train.get_global_step())

                    # Create an EstimatorSpec
                    spec = tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      train_op=train_op)

                    logger.debug('Created TRAIN EstimatorSpec')

                    # Minimize the loss in TRAIN mode
                    return spec

            # Build evaluation op
            with tf.name_scope('eval'):

                # Add evaluation metrics
                # noinspection PyUnresolvedReferences
                flat_labels = tf.layers.flatten(labels['label'])
                flat_labels = \
                    tf.reshape(
                        tensor=flat_labels,
                        shape=[batch_size*flat_labels.get_shape()[1]])
                flat_predictions = tf.layers.flatten(predictions['classes'])
                flat_predictions = tf.reshape(
                    tensor=flat_predictions,
                    shape=[batch_size * flat_predictions.get_shape()[1]])

                eval_ops = {
                    'accuracy': tf.metrics.accuracy(
                        labels=labels['label'],
                        predictions=predictions['classes'],
                        name='accuracy'),

                    'mean_iou': tf.metrics.mean_iou(
                        labels=labels['label'],
                        predictions=predictions['classes'],
                        num_classes=n_classes),

                    'adj_rand_idx': _adj_rand_idx_metric_op(flat_labels,
                                                            flat_predictions)
                }

                # Create an EstimatorSpec
                spec = tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_ops)

                logger.debug('Created EVAL EstimatorSpec')

                return spec


def _adj_rand_idx_metric_op(label: np.ndarray, predict: np.ndarray) -> \
        Tuple[tf.Tensor, tf.Operation]:
    """ Wrapper around sklearn.adjusted_rand_score, in order to be able to use
    it with the estimator's eval metric ops.

    Args:
        label (np.ndarray): Flattened array of true labels of shape [batch_size]
        predict (np.ndarray): Flattened array of predictions of shape [
           batch_size]

    Returns:
        adj_rand_idx (Tensor): Return the adjusted rand index as computed by
            sklearn.
        op (tf.Operation): An operation that increments the total and count
            variables appropriately and whose value matches mean_value.
    """
    value = tf.py_func(func=adjusted_rand_score,
                       inp=[label, predict],
                       Tout=tf.float64)
    adj_rand_idx, op = tf.metrics.mean(value)

    return adj_rand_idx, op


def _random_name(n_words: int = 4) -> str:
    """Generate a random name for a GeneNet.

    Args:
        n_words (int): Number of words to use in the name.

    Returns:
        (str): A name.

    """
    name_words = []
    for n in range(n_words):
        name_words.append(random.choice(WORDS).capitalize())
    return ''.join(name_words)
