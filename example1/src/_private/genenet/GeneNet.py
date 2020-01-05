"""GeneNet class for building TensorFlow computation graphs from GeneGraphs.

"""
import json
import logging
import os
import pickle
import random
import time

import metatree as mt
import numpy as np
import tensorflow as tf

from sklearn.metrics.cluster import adjusted_rand_score
from tensorflow.python.client import device_lib

from .genes import ConvolutionGene
from .summarize import image_summary
from .summarize import variable_summary
from ._words import WORDS

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, \
    TypeVar, Union


# Image instance_settings type: To record an image summary of a Gene's output tensor,
# create a tuple where the first element is the Gene, and the second element
# is a dictionary of image summary parameters as defined in the `params` arg
# of `genenet.summarize.image_summary`. Additional special tensors can be
# summarized using special string keys: 'input', 'classes',
# and 'probabilities', which are associated with the input tensor,
# class prediction tensor, and class probability tensor respectively
ImageSettings = TypeVar(Tuple[Union[str, mt.Gene], Dict[str, Any]])
# Dictionary of string keys mapping to mt.HyperparameterConfig objects
HyperparamConfigs = TypeVar(Dict[str, mt.HyperparameterConfig])

logger = logging.getLogger('genenet.GeneNet')

# Name for an eval history JSON file
_EVAL_FILE = 'file_history.json'
# Name for a pickled GeneGraph copy
_GENEGRAPH_FILE = 'gene_graph.pkl'


class GeneNet:
    """Builds a TensorFlow computation graph from a GeneGraph using an
    Estimator.

    TODO
    Attributes:

    """
    def __init__(self,
                 gene_graph: mt.GeneGraph,
                 name: Optional[str]=None,
                 save_dir: Optional[str]=None,
                 param_seed: Optional[int]=None,
                 image_settings: Optional[Sequence[ImageSettings]]=None,
                 make_summaries: bool=False,
                 device: Optional[str]=None,
                 build_now: bool=True):
        """Constructor.

        Args:
            gene_graph (mt.GeneGraph): A GeneGraph that specifies the
                architecture of this GeneNet's computation graph.
            name (Optional[str]): Unique name for this network. If two
                GeneNets with the same name are run in the same TensorFlow
                session, variable scope conflicts may occur.
            save_dir (Optional[str]): Directory where GeneNet information is
                saved.
            param_seed (Optional[int]): Random seed for network weight
                parameter initialization.
            image_settings (Optional[Sequence[ImageSettings]]): A collection of
                ImageSettings tuples, specifying which tensors should generate
                image summaries.
            make_summaries (bool): If True, attach variable summaries to each
                ConvolutionGene in this GeneNet's gene graph.
            device (Optional[str]): TensorFlow Device on which to build the
                `model_fn`'s computation graph. By default, will use the
                first available GPU (if present), else the CPU.
            build_now (bool): If True, instantiate an Estimator now.
        """
        # Network GeneGraph
        self.gene_graph = gene_graph
        # Input data spatial shape
        self.input_shape = gene_graph.input_shape
        # Output data spatial shape
        self.output_shape = gene_graph.output_shape()
        # Output data spatial size
        self.output_size = 1
        for s in self.output_shape:
            self.output_size *= s
        # Number of trainable parameters. Not set until self.model_fn() is
        # called
        self.n_trainable_params = None

        # Unique identifier to use when name scoping network variables
        if name is None:
            # Load existing name is an eval file already exists in the save
            # directory
            if os.path.isfile(os.path.join(save_dir, _EVAL_FILE)):
                eval_history_file = os.path.join(save_dir, _EVAL_FILE)
                with open(eval_history_file) as fl:
                    eval_history = json.load(fl)
                last_eval = eval_history[-1]
                self.name = last_eval['name']
            else:
                # Create a new random name
                self.name = _random_name()
        else:
            self.name = name

        # Create a save directory, if supplied
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.model_dir = os.path.join(self.save_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)

        # Generate a weight parameter seed if none is supplied
        if param_seed is None:
            param_seed = np.random.randint(np.iinfo(np.int32).max)
        self.param_seed = param_seed

        if image_settings is None:
            image_settings = []

        # Keep a history of evaluation results, for easier visualization
        # since Tensorboard is a pain
        self.eval_history: List[Dict[str, Any]] = []
        # Load history if any exists
        eval_history_file = os.path.join(self.save_dir, _EVAL_FILE)
        if os.path.isfile(eval_history_file):
            with open(eval_history_file) as fl:
                self.eval_history = json.load(fl)

        # Create a dict of additional model generation hyperparameters for an
        # Estimator `model_fn`
        # ADAM optimizer hyperparams
        params = self.gene_graph.hyperparameter_configs['optim']
        # Image generation options
        params['image_settings'] = image_settings
        params['make_summaries'] = make_summaries

        # Device setup
        if device is None:
            # All computational devices available to TensorFlow
            devices = [d.name for d in device_lib.list_local_devices()]
            # Available GPUs, under the assumption that a device name
            # contains 'GPU' iff it is a GPU
            gpus = [d for d in devices if 'GPU' in d]
            # Use a GPU if available, else use the last available device.
            # There is always at least one, the CPU.
            if len(gpus) > 0:
                # Use the last GPU (on the assumption that other GPU
                # processes are usually happening on GPU 0)
                device = gpus[-1]
            else:
                device = devices[0]
        self.device = device

        self.build_params = params

        # GeneNet Estimator creation
        self.estimator = None
        if build_now:
            self.build()

        pass

    def build(self):
        run_config = tf.estimator.RunConfig(save_summary_steps=1000,
                                            keep_checkpoint_max=1,
                                            tf_random_seed=self.param_seed)
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                model_dir=self.model_dir,
                                                config=run_config,
                                                params=self.build_params)

    def evaluate(self,
                 input_fn: Callable[[], None],
                 hooks: Optional[Sequence[tf.train.SessionRunHook]] = None) \
            -> Dict[str, Any]:
        """Evaluate this GeneNet's Estimator.

        Args:
            input_fn (Callable[[], None]): Input function for
                tf.estimator.Estimator.evaluate().
            hooks (Optional[Sequence[tf.train.SessionRunHook]]): List of
                `SessionRunHook` subclass instances. Used for callbacks inside
                the evaluation call.

        Returns:
            (Dict[str, Any]): A dict containing the evaluation metrics
                specified in `model_fn` keyed by name, as well as an entry
                `global_step` which contains the value of the global step for
                which this evaluation was performed.

        """
        if hooks is None:
            hooks = []

        # Run evaluation
        eval_results = self.estimator.evaluate(input_fn,
                                               hooks=hooks)

        # Convert to list
        eval_list = {key: eval_results[key].tolist() for key in eval_results}
        # Add some constant info, just to store it
        eval_list['name'] = self.name
        eval_list['param_seed'] = self.param_seed
        # Add a time stamp to track when things were last modified
        eval_list['last_eval_time'] = time.time()
        eval_list['n_trainable_params'] = self.n_trainable_params

        # Add eval results to eval_history
        self.eval_history.append(eval_list)

        # Save updated eval results to disk
        eval_history_file = os.path.join(self.save_dir, _EVAL_FILE)
        with open(eval_history_file, 'w') as fl:
            json.dump(self.eval_history, fl)

        # Save a pickled copy of the gene graph
        gene_graph_pickle_file = os.path.join(self.save_dir, _GENEGRAPH_FILE)
        with open(gene_graph_pickle_file, 'wb') as fl:
            pickle.dump(self.gene_graph, fl)

        return eval_list

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

                if logits.get_shape().ndims == 4:
                    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels['label'],
                        logits=tf.transpose(logits,
                                            [0, 2, 3, 1],
                                            name='transpose'),
                        name='softmax_xentropy')
                else:
                    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels['label'],
                        logits=tf.transpose(logits, [0, 2, 3, 4, 1],
                                            name='transpose'),
                        name='softmax_xentropy')

                if mode == tf.estimator.ModeKeys.TRAIN:
                    # During training, use per-voxel cross-entropy weighting
                    # plus regularization terms

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
                    weighted_xentropy = tf.multiply(weights,
                                                    xentropy,
                                                    name='weighted_xentropy')
                    # Sum voxel loss values
                    loss = tf.reduce_sum(weighted_xentropy,
                                         name='sum_xentropy')

                    # Add regularization terms, weighted to ignore zeroed-out
                    # areas
                    frac_nonzero = tf.reduce_mean(nonzero_weights)
                    loss += frac_nonzero * tf.losses.get_regularization_loss()

                else:
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

    def predict(self,
                input_fn: Callable[[], None],
                hooks: Optional[Sequence[tf.train.SessionRunHook]]=None) -> \
            Dict[str, tf.Tensor]:
        """Forward prediction pass through this GeneNet's estimator.

        Basically a wrapper around tf.estimator.Estimator.predict().

        Args:
            input_fn (Callable[[], None]): Input function for
                tf.estimator.Estimator.evaluate().
            hooks (Optional[Sequence[tf.train.SessionRunHook]]): List of
                `SessionRunHook` subclass instances. Used for callbacks inside
                the evaluation call.

        Returns:
            (Dict[str, tf.Tensor]): Dictionary of forward pass output tensors.

        """
        if hooks is None:
            hooks = []
        return self.estimator.predict(input_fn,
                                      hooks=hooks)

    @staticmethod
    def restore(net_dir: str,
                hyperparam_configs: Optional[HyperparamConfigs]=None,
                image_settings: Optional[Sequence[ImageSettings]]=None,
                make_summaries: bool=False,
                device: Optional[str]=None,
                build_now: bool=True):
        """Restore a GeneNet from a save dir.

        Args:
            net_dir (str): Save directory containing the GeneNet to be restored.
            hyperparam_configs (Optional[HyperparamConfigs]): Used to update
                the hyperparameter configs of the restored gene graph.
                This may be useful when a restored GeneNet used an older
                version of the mt.HyperparameterConfig that one is
                currently using. For each key in `hyperparam_configs`,
                assign hyperparam_configs[key] to
                gene_graph.hyperparameter_configs[key] and
                gene.hyperparameter_config for each gene in the tree with root
                gene_graph.genes[key].
            image_settings (Optional[Sequence[ImageSettings]]): A collection of
                ImageSettings tuples, specifying which tensors should generate
                image summaries.
            make_summaries (bool): If True, attach variable summaries to each
                ConvolutionGene in this GeneNet's gene graph.
            device (Optional[str]): TensorFlow name for the computational
                device (CPU/GPU) to restore the GeneNet onto.
            build_now (bool): If True, build the restored GeneNet now.

        Returns:
            (GeneNet): The restored GeneNet.

        """
        # Load the network's Gene graph
        gene_graph_file = os.path.join(net_dir, 'gene_graph.pkl')
        with open(gene_graph_file, 'rb') as fl:
            gene_graph: mt.GeneGraph = pickle.load(fl)

        if hyperparam_configs is not None:
            # Do hyperparameter config replacement
            for key in hyperparam_configs:
                config = hyperparam_configs[key]
                # Update the gene_graph's copy of the hyperparam config
                gene_graph.hyperparameter_configs[key] = config
                # Update the hyperparam configs in the tree of genes and edges
                # associated with the same key
                genes, edges = gene_graph.genes[key].all_descendants()
                for node in genes + edges:
                    node.hyperparameter_config = config
                    for k in config.names():
                        if k not in node.deltas:
                            # Use default instance_settings for any new hyperparams
                            # i.e. default hyperparam value delta at the
                            # root, and a delta of 0 otherwise
                            if node.level == 0:
                                value = config.values()[k]
                                # Add key to deltas
                                node.deltas[k] = 0
                                settings = {k: value}
                                node.set(**settings)
                            else:
                                node.deltas[k] = 0

        # Get the network's name and param seed
        eval_history_file = os.path.join(net_dir, _EVAL_FILE)
        with open(eval_history_file) as fl:
            eval_history = json.load(fl)
        last_eval = eval_history[-1]
        name = last_eval['name']
        param_seed = last_eval['param_seed']

        # Restore the net
        net = GeneNet(gene_graph,
                      name,
                      net_dir,
                      param_seed,
                      image_settings,
                      make_summaries,
                      device,
                      build_now)

        return net

    def train(self,
              input_fn: Callable[[], None],
              max_steps: Optional[int]=None,
              hooks: Optional[Sequence[tf.train.SessionRunHook]]=None):
        """Train this GeneNet's TF Graph.

        Args:
            input_fn (Callable[[], None]): TensorFlow Estimator input_fn.
            max_steps (Optional[int]): If supplied, limits training to this
                number of steps. Otherwise, use the input_fn's num_epochs to
                determine how many times to iterate through training data.
            hooks (Optional[Sequence[tf.train.SessionRunHook]]): Training
                hooks to add to the training session.

        Returns: None

        """
        if hooks is None:
            hooks = []

        # Train the model
        self.estimator.train(
            input_fn=input_fn,
            max_steps=max_steps,
            hooks=hooks)

        pass


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


def _random_name(n_words: int=4) -> str:
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
