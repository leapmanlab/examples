"""Segment an image volume using a GeneNet.

"""
import logging
import os
import time

import genenet as gn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tifffile as tif

from .DataHandler import DataHandler
from .GeneNet import GeneNet

from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger('genenet.segment')


def segment(nets: Union[GeneNet, Sequence[GeneNet], str, Sequence[str]],
            data_vol: np.ndarray,
            data_handler: DataHandler,
            save_dir: Optional[str] = None,
            image_cmaps: Optional[Dict[str, str]] = None,
            draw_prob_maps: bool = False,
            label_vol: Optional[np.ndarray] = None,
            save_name: Optional[str] = None,
            forward_window_overlap: Optional[Sequence[int]] = None) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Segment some data using one or more GeneNets, and possibly save some
    images of
    the results.

    Args:
        nets (Union[GeneNet, Sequence[GeneNet], str, Sequence[str]): One or
            more GeneNets or GeneNet save directories.
        data_vol (np.ndarray): Snapshot data volume.
        data_handler (DataHandler): A DataHandler for input_fn generation.
        save_dir (Optional[str]): Directory where snapshot images get saved.
            If none is supplied, no images are saved.
        image_cmaps (Optional[Dict[str, str]]): Colormaps for snapshot
            images, represented as names for matplotlib cmaps. Default is
            {'data': 'gray', 'segmentation': 'jet'}. To save probability
            map images, add a 'prob_maps' key to `image_cmaps`, I
            recommend the value 'pink'.
        draw_prob_maps (bool): If True, assemble a probability map
            volume during the segmentation of `data_vol`.
        label_vol (Optional[np.ndarray])=None: Snapshot label volume. If
            supplied, snapshot will include an image of the label volume.
        save_name (Optional[str]): Snapshot image base save name. If none
            is provided, one is generated automatically.
        forward_window_overlap (Sequence[int]): Overlap between
                successive windows during forward (inference) passes through
                a network. Used to mitigate edge effects caused by
                partitioning a large volume into independent windows for
                segmentation. Default is no overlap.


    Returns:
        segmentation (np.ndarray): The segmented `data_vol`.
        prob_maps (np.ndarray): Only returned if `draw_prob_maps`, this is
            a probability map volume output by `net` during segmentation.

    """
    if save_name is None:
        save_name = time.strftime('%Y%m%d-%H%M%S')
    if image_cmaps is None:
        image_cmaps = {'data': 'gray',
                       'segmentation': 'jet'}

    ndim_data = data_vol.ndim

    all_prob_maps = []
    if not isinstance(nets, List) and not isinstance(nets, Tuple):
        net_sources = [nets]
    else:
        net_sources = nets
    for net in net_sources:
        if isinstance(net, str):
            net = gn.GeneNet.restore(net)

        output_shape = net.gene_graph.output_shape()
        net_is_3d = len(output_shape) == 3
        # Shape of the segmentation volume is same as data volume if
        # single-channel, else ignore the channel dimension
        if ndim_data > 3:
            vol_shape = data_vol.shape[1:]
        else:
            vol_shape = data_vol.shape
        # Initialize the volumes
        # segmentation = np.zeros(vol_shape)
        # Shape of the probability map volume: one map per class
        prob_shape = [data_handler.n_classes] + list(vol_shape)
        prob_maps = np.zeros(prob_shape)
        # Create an input_fn
        predict_input_fn = data_handler.input_fn(
            mode=tf.estimator.ModeKeys.PREDICT,
            graph_source=net,
            forward_window_overlap=forward_window_overlap,
            prediction_volume=data_vol)
        # Inference pass result generator
        results: Iterator[Dict[str, np.ndarray]] = net.predict(predict_input_fn)

        for r in results:
            patch_prob = r['probabilities']
            patch_corner = r['corner']
            if net_is_3d:
                z0 = patch_corner[0]
                z1 = z0 + output_shape[0]
                x0 = patch_corner[1]
                x1 = x0 + output_shape[1]
                y0 = patch_corner[2]
                y1 = y0 + output_shape[2]

                prob_maps[:, z0:z1, x0:x1, y0:y1] = patch_prob
            else:
                z0 = patch_corner[0]
                x0 = patch_corner[1]
                x1 = x0 + output_shape[0]
                y0 = patch_corner[2]
                y1 = y0 + output_shape[1]
                if ndim_data > 2:
                    prob_maps[:, z0, x0:x1, y0:y1] = patch_prob
                else:
                    prob_maps[:, x0:x1, y0:y1] = patch_prob
        all_prob_maps.append(prob_maps)

    prob_map_mean = np.mean(all_prob_maps, axis=0)
    segmentation = np.argmax(prob_map_mean, axis=0)

    def tif_cmap(c):
        """Convert a matplotlib colormap into a tifffile colormap.

        """
        a = plt.get_cmap(c)(np.arange(256))
        return np.swapaxes(255 * a, 0, 1)[0:3, :].astype('u1')

    # Save a bunch of images, if `save_dir` was supplied
    if save_dir is not None:
        # Create a data volume image
        # For multichannel 3D data, only use the first channel, under the
        # assumption that that is actual image data
        # TODO: Find a more robust solution for multichannel data
        if ndim_data == 4:
            data_vol = data_vol[0, ...]
        # Generate a file name and path
        data_fname = f'train-data.tif'
        data_fpath = os.path.join(save_dir, data_fname)
        # Create a colormap compatible with tifffile's save function
        data_tcmap = tif_cmap(image_cmaps['data'])

        # Convert data volume to the right type
        data_image = (255. * (data_vol - data_vol.min()) /
                      (data_vol.max() - data_vol.min())).astype(np.uint8)
        # Save
        tif.imsave(data_fpath, data_image, colormap=data_tcmap)

        # Create a label volume image, if supplied
        if label_vol is not None:
            # Generate a file name and path
            label_fname = f'train-label.tif'
            label_fpath = os.path.join(save_dir, label_fname)
            # Create a colormap compatible with tifffile's save function
            label_tcmap = tif_cmap(image_cmaps['segmentation'])
            # Convert and scale the label volume
            label_image = (255. / (data_handler.n_classes - 1) *
                           label_vol).astype(np.uint8)
            # Save
            tif.imsave(label_fpath, label_image, colormap=label_tcmap)

        # Create a segmentation volume image
        # Generate a file name and path
        seg_fname = f'segmentation_{save_name}.tif'
        seg_fpath = os.path.join(save_dir, seg_fname)
        # Create a colormap compatible with tifffile's save function
        seg_tcmap = tif_cmap(image_cmaps['segmentation'])
        # Convert and scale the segmentation volume
        seg_image = (255. / (data_handler.n_classes - 1) *
                     segmentation).astype(np.uint8)
        # Save
        tif.imsave(seg_fpath, seg_image, colormap=seg_tcmap)

        # Create probability map images, if specified
        if draw_prob_maps:
            for i in range(data_handler.n_classes):
                # Generate a file name and path
                pmap_fname = f'prob_map_{i}_{save_name}.tif'
                pmap_fpath = os.path.join(save_dir, pmap_fname)
                # Create a colormap compatible with tifffile's save function
                pmap_tcmap = tif_cmap(image_cmaps['prob_maps'])
                # Convert and scale the probability map
                pmap_image = (255. * prob_map_mean[i, ...]).astype(np.uint8)
                # Save
                tif.imsave(pmap_fpath, pmap_image, colormap=pmap_tcmap)

    if draw_prob_maps:
        return segmentation, prob_map_mean
    else:
        return segmentation
