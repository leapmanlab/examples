"""Evaluate a network on a dataset, saving one or more outputs of the process.

"""
import json
import os

import genenet as gn
import tensorflow as tf

from .checkpoints import restore_from_checkpoint

from typing import Dict, Optional, Union


def evaluate(
        net_dir: str,
        image_file: str,
        label_file: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        segmentation_out_tif: Optional[str] = None,
        eval_out_json: Optional[str] = None,
        save_prob_maps: bool = False,
        device: Optional[str] = None) -> Dict[str, Union[float, str]]:
    """

    Args:
        net_dir:
        image_file:
        label_file:
        checkpoint_dir:
        segmentation_out_tif:
        eval_out_json:
        save_prob_maps:
        device:

    Returns:

    """

    do_eval = label_file is not None
    save_image = segmentation_out_tif is not None
    save_eval = eval_out_json is not None
    if save_eval:
        eval_dir = os.path.dirname(eval_out_json)
        if eval_dir != '':
            os.makedirs(eval_dir, exist_ok=True)

    net = restore_from_checkpoint(net_dir, checkpoint_dir, device=device)

    eval_dir = os.path.dirname(image_file)
    image_name = os.path.basename(image_file)
    if do_eval:
        label_rel_name = os.path.relpath(label_file, eval_dir)
        data_handler = gn.DataHandler(eval_data_dir=eval_dir,
                                      eval_file=image_name,
                                      eval_label_file=label_rel_name)
    else:
        data_handler = gn.DataHandler(eval_data_dir=eval_dir,
                                      eval_file=image_name)

    eval_input_fn = data_handler.input_fn(
        mode=tf.estimator.ModeKeys.EVAL,
        graph_source=net.gene_graph)

    eval_results = net.evaluate(eval_input_fn)

    if save_eval:
        with open(eval_out_json, 'w') as fd:
            json.dump(eval_results, fd)

    if save_image:
        # Segmentation coloration info
        image_cmaps = {'data': 'gray',
                       'segmentation': 'jet',
                       'prob_maps': 'pink'}

        media_dir = os.path.dirname(segmentation_out_tif)
        os.makedirs(media_dir, exist_ok=True)
        tif_name = os.path.splitext(os.path.basename(segmentation_out_tif))[0]
        gn.segment(net,
                   data_handler=data_handler,
                   data_vol=data_handler.eval_volume,
                   label_vol=data_handler.eval_label_volume,
                   save_dir=media_dir,
                   image_cmaps=image_cmaps,
                   save_name=tif_name,
                   draw_prob_maps=save_prob_maps)

    return eval_results
