"""Create a network, prep for training.

"""
import os

# from _private
import genenet as gn

from typing import Any, Dict, Tuple


def create_network(instance_settings: Dict[str, Any]) -> \
        Tuple[gn.GeneNet, Dict[str, Any], Dict[str, Any]]:
    """Create a TensorFlow computation graph to train the U-Net on the
    platelet segmentation task.

    Args:
        instance_settings (Dict[str, Any]): Dictionary of example instance_settings that controls
            the creation and training of a TensorFlow network.

    Returns:
        net (gn.GeneNet): An in-lab wrapper around a TensorFlow computation
            graph that implements training, eval, and inference with a
            segmentation neural network.
        train_settings (Dict[str, Any]): Settings for training `net`.
        eval_settings (Dict[str, Any]): Settings for evaluating `net`.
    """
    # See 'example1_train_unet.ipynb' for setting definitions
    data_dir = instance_settings['data_dir']
    instance_dir = instance_settings['instance_dir']
    input_shape = instance_settings['input_shape']
    n_epochs = instance_settings['n_epochs']
    stop_criterion = instance_settings['stop_criterion']
    weight_seed = instance_settings['weight_seed']
    data_seed = instance_settings['data_seed']
    train_window_spacing = instance_settings['train_window_spacing']
    net_settings = instance_settings['net_settings']
    optim_settings = instance_settings['optim_settings']

    # Create a DataHandler, load data
    data_handler = gn.DataHandler(
        train_data_dir=data_dir,
        train_file='train-images.tif',
        train_label_file='train-labels.tif',
        train_weight_file='train-error-weights.tif',
        eval_file='eval-images.tif',
        eval_label_file='eval-labels.tif')

    # Create a GeneGraph for an encoder-decoder segmentation network
    gene_graph = gn.gene_graph(input_shape,
                               data_handler.n_classes,
                               net_settings=net_settings,
                               predictor_settings={},
                               optim_settings=optim_settings)

    # Image summary instance_settings
    image_settings = [
        ('input', {}),
        ('classes', {'cmap': 'jet', 'vmin': 0,
                     'vmax': data_handler.n_classes}),
        ('probabilities', {'cmap': 'pink', 'vmin': 0, 'vmax': 1})
    ]

    # Create a GeneNet
    net = gn.GeneNet(gene_graph,
                     name='U-Net',
                     save_dir=instance_dir,
                     param_seed=weight_seed,
                     image_settings=image_settings)

    # Media directory
    eval_media_dir = os.path.join(instance_dir, 'media')
    os.makedirs(eval_media_dir, exist_ok=True)
 
    # Network snapshot info. Create images on the eval volume.
    sample_image = data_handler.eval_volume
    sample_label = data_handler.eval_label_volume
    image_cmaps = {'data': 'gray',
                   'segmentation': 'jet',
                   'prob_maps': 'pink'}

    # Package training info into a dict
    train_settings = {
        'data_handler': data_handler,
        'n_epochs': n_epochs,
        'max_steps': None,
        'stop_criterion': stop_criterion,
        'data_seed': data_seed,
        'window_spacing': train_window_spacing
    }

    # Package eval and media generation info into a dict
    eval_settings = {
        'instance_dir': instance_dir,
        'eval_media_dir': eval_media_dir,
        'sample_image': sample_image,
        'sample_label': sample_label,
        'image_cmaps': image_cmaps
    }

    return net, train_settings, eval_settings
