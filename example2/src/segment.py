"""Use a trained network to segment an image.

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tifffile as tif

# From _private
import genenet as gn
import leapmanlab as lab

from functools import lru_cache

from scipy.ndimage.morphology import distance_transform_edt

from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union


def segment(
        net_sources: Union[gn.GeneNet, str, List[str]],
        image_source: Union[np.ndarray, str],
        output_dir: Optional[str] = None,
        label_source: Optional[Union[np.ndarray, str]] = None,
        device: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        net_sources (Union[gn.GeneNet, str, List[str]]): A GeneNet or one or more
            paths to directories containing saved model files. If more than one
            source is supplied, the resulting segmentation is an ensemble of the
            outputs from each source.
        image_source (Union[np.ndarray, str]): Either an image as a numpy array,
            or a path to a saved image. This image is segmented by the net(s).
        output_dir (Optional[str]): Directory for saving the output of the segmentation
            process, if supplied.
        label_source (Optional[Union[np.ndarray, str]]):
        device:

    Returns:
        segmentation (np.ndarray): Segmentation of `image_source`'s image. Has
            the same shape as the image.
        prob_maps (np.ndarray): Per-voxel probability maps for each class across
            the input image. `prob_maps[i, ...]` is the per-voxel probability
            map for class `i`, and has the same shape as the input image.

    """
    if isinstance(net_sources, str) or isinstance(net_sources, gn.GeneNet):
        net_sources = [net_sources]

    if label_source and type(image_source) != type(label_source):
        raise TypeError('label_source must have same type as image_source if'
                        ' supplied')

    if isinstance(image_source, str):
        eval_dir = os.path.dirname(image_source)
        image_file = os.path.basename(image_source)
        if label_source:
            label_rel_name = os.path.relpath(label_source, eval_dir)
        else:
            label_rel_name = None
    else:
        eval_dir = None
        image_file = image_source
        if label_source:
            label_rel_name = label_source

    data_handler = gn.DataHandler(
        eval_data_dir=eval_dir,
        eval_file=image_file,
        eval_label_file=label_rel_name)
    data_vol = data_handler.eval_volume

    image_cmaps = {'data': 'gray',
                   'segmentation': 'jet',
                   'prob_maps': 'pink'}
    save_name = 'prediction.tif'

    ndim_data = data_vol.ndim

    all_prob_maps = []
    for source in net_sources:
        if isinstance(source, str):
            ckpt_dir = os.path.join(source, 'model', 'checkpoints', 'best')
            if os.path.exists(ckpt_dir):
                net = lab.restore_from_checkpoint(source, ckpt_dir, device)
            else:
                net = lab.restore_from_checkpoint(source, None, device)
        else:
            net = source

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
        prob_map_update_dist = np.zeros(vol_shape, dtype=np.int)
        # Create an input_fn
        # Check if SAME padding is used
        gene0 = list(net.gene_graph.genes.items())[0][1]
        padding = gene0.hyperparam('padding_type')
        is_same_padded = padding.lower() == 'same'
        if is_same_padded:
            forward_window_overlap = [1] * (3 - len(output_shape)) + [s // 3 for s in output_shape]
        else:
            forward_window_overlap = [0] * 3

        predict_input_fn = data_handler.input_fn(
            mode=tf.estimator.ModeKeys.PREDICT,
            graph_source=net,
            forward_window_overlap=forward_window_overlap,
            prediction_volume=data_vol)
        # Inference pass result generator
        results: Iterator[Dict[str, np.ndarray]] = net.predict(predict_input_fn)

        # TODO: pass that distance scale triplet as a parameter instead of hard-coding
        if net_is_3d:
            distance_scale = (4, 1, 1)
        else:
            distance_scale = (1, 1)

        for r in results:
            patch_prob = r['probabilities']
            patch_dist = memoized_distance_transform(patch_prob.shape[1:], distance_scale)
            patch_corner = r['corner']

            if net_is_3d:
                z0 = patch_corner[0]
                z1 = z0 + output_shape[0]
                x0 = patch_corner[1]
                x1 = x0 + output_shape[1]
                y0 = patch_corner[2]
                y1 = y0 + output_shape[2]
                region = (slice(z0, z1), slice(x0, x1), slice(y0, y1))
            else:
                z0 = patch_corner[0]
                x0 = patch_corner[1]
                x1 = x0 + output_shape[0]
                y0 = patch_corner[2]
                y1 = y0 + output_shape[1]
                if ndim_data > 2:
                    region = (slice(z0, z0 + 1), slice(x0, x1), slice(y0, y1))
                else:
                    region = (slice(x0, x1), slice(y0, y1))

            if is_same_padded:
                parts_to_update = prob_map_update_dist[region] < patch_dist

                padded_parts_to_update = np.zeros_like(prob_map_update_dist, dtype=np.bool)
                padded_parts_to_update[region] = parts_to_update

                prob_maps[:, padded_parts_to_update] = patch_prob[:, parts_to_update]
                prob_map_update_dist[padded_parts_to_update] = patch_dist[parts_to_update]
            else:
                if net_is_3d:
                    prob_maps[:, z0:z1, x0:x1, y0:y1] = patch_prob
                else:
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
    if output_dir is not None:
        # Create a data volume image
        # For multichannel 3D data, only use the first channel, under the
        # assumption that that is actual image data
        # TODO: Find a more robust solution for multichannel data
        if ndim_data == 4:
            data_vol = data_vol[0, ...]
        # Generate a file name and path
        data_fname = f'train-data.tif'
        data_fpath = os.path.join(output_dir, data_fname)
        # Create a colormap compatible with tifffile's save function
        data_tcmap = tif_cmap(image_cmaps['data'])

        # Convert data volume to the right type
        data_image = (255. * (data_vol - data_vol.min()) /
                      (data_vol.max() - data_vol.min())).astype(np.uint8)
        # Save
        tif.imsave(data_fpath, data_image, colormap=data_tcmap)

        # Create a segmentation volume image
        # Generate a file name and path
        seg_fname = save_name
        seg_fpath = os.path.join(output_dir, seg_fname)
        # Create a colormap compatible with tifffile's save function
        seg_tcmap = tif_cmap(image_cmaps['segmentation'])
        # Convert and scale the segmentation volume
        seg_image = (255. / (data_handler.n_classes - 1) *
                     segmentation).astype(np.uint8)
        # Save
        tif.imsave(seg_fpath, seg_image, colormap=seg_tcmap, compress=7)

    return segmentation, prob_map_mean


@lru_cache(maxsize=128)
def memoized_distance_transform(
        shape: Sequence[int],
        distance_scale: Optional[Sequence[int]] = None) -> np.ndarray:
    """

    Args:
        shape:
        distance_scale

    Returns:
        (np.ndarray)
    """
    if not distance_scale:
        distance_scale = [1 for s in shape]

    expanded_shape = [s + 2 for s in shape]

    arr_padded = np.zeros(expanded_shape)
    center_slice = (slice(1, -1),) * len(shape)
    arr_padded[center_slice] = 1
    dist_padded = distance_transform_edt(arr_padded, sampling=distance_scale)
    return dist_padded[center_slice]