"""Visual demos of various parts of the training process

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif

# From _private
import genenet as gn

from .segment import segment

from typing import List, Optional, Union


def demo_data(
        data_dir: str,
        z: Optional[int] = None,
        dpi: float = 110):
    """Demo of the dataset used in (Guay et al., 2019).

    Args:
        data_dir (str):
        z (Optional[int]):
        dpi (float):

    Returns: None

    """
    ''' Training data images '''

    z_train = z if z else 15
    train_img = tif.imread(os.path.join(data_dir, 'train-images.tif'))[z_train]
    train_lab = tif.imread(os.path.join(data_dir, 'train-labels.tif'))[z_train]
    train_w = tif.imread(os.path.join(
        data_dir,
        'train-error-weights.tif'))[z_train]

    img_y = int(train_img.shape[0] / dpi)
    img_x = int(train_img.shape[1] / dpi)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(3 * img_x, img_y))

    plt.subplots_adjust(wspace=0.05)

    axs[0].imshow(train_img, cmap='gray')
    axs[0].set_title('Train image')
    axs[0].axis('off')

    axs[1].imshow(train_lab, cmap='jet', vmin=0, vmax=6)
    axs[1].set_title('label')
    axs[1].axis('off')

    axs[2].imshow(train_w, cmap='magma')
    axs[2].set_title('error weight')
    axs[2].axis('off')

    ''' Eval data images '''

    z_eval = z if z else 4
    eval_img = tif.imread(os.path.join(data_dir, 'eval-images.tif'))[z_eval]
    eval_lab = tif.imread(os.path.join(data_dir, 'eval-labels.tif'))[z_eval]

    img_y = int(eval_img.shape[0] / dpi)
    img_x = int(eval_img.shape[1] / dpi)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(3 * img_x, img_y))

    plt.subplots_adjust(wspace=0.05)

    axs[0].imshow(eval_img, cmap='gray')
    axs[0].set_title('Eval image')
    axs[0].axis('off')

    axs[1].imshow(eval_lab, cmap='jet', vmin=0, vmax=6)
    axs[1].set_title('label')
    axs[1].axis('off')

    axs[2].imshow(np.ones_like(eval_img, dtype=np.uint8), cmap='gray', vmin=0)
    axs[2].axis('off')

    ''' Test data images '''

    z_test = z if z else 75
    test_img = tif.imread(os.path.join(data_dir, 'test-images.tif'))[z_test]
    test_lab = tif.imread(os.path.join(data_dir, 'test-labels.tif'))[z_test]

    img_y = int(test_img.shape[0] / dpi)
    img_x = int(test_img.shape[1] / dpi)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(3 * img_x, img_y))

    plt.subplots_adjust(wspace=0.05)

    axs[0].imshow(test_img, cmap='gray')
    axs[0].set_title('Test image')
    axs[0].axis('off')

    axs[1].imshow(test_lab, cmap='jet', vmin=0, vmax=6)
    axs[1].set_title('label')
    axs[1].axis('off')

    axs[2].imshow(np.ones_like(test_img, dtype=np.uint8), cmap='gray', vmin=0)
    axs[2].axis('off')

    pass


def demo_segmentation(
        net_sources: Union[gn.GeneNet, str, List[str]],
        data_dir: str,
        z: Optional[int] = None,
        dpi: float = 110,
        show_prob_maps: bool = True):
    """Demo of segmentation on the training, eval, and test datasets

    Args:
        net_sources (Union[gn.GeneNet, str, List[str]]):
        data_dir (str):
        z (Optional[int]):
        dpi (float):
        show_prob_maps (bool):

    Returns: None

    """

    ''' Compute segmentation on train data example '''

    class_names = {
        0: 'Background',
        1: 'Cell',
        2: 'Mitochondrion',
        3: 'Alpha granule',
        4: 'Canalicular vessel',
        5: 'Dense granule',
        6: 'Dense core'}

    train_seg, train_probs = segment(
        net_sources=net_sources,
        image_source=os.path.join(data_dir, 'train-images.tif'),
        output_dir=None,
        label_source=os.path.join(data_dir, 'train-labels.tif'))

    z_train = z if z else 15
    seg = train_seg[z_train]
    probs = train_probs[:, z_train, ...]
    label = tif.imread(os.path.join(data_dir, 'train-labels.tif'))[z_train]

    img_y = int(seg.shape[0] / dpi)
    img_x = int(seg.shape[1] / dpi)

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(2 * img_x, img_y))

    plt.subplots_adjust(wspace=0.05)

    axs[0].imshow(seg, cmap='jet', vmin=0, vmax=6)
    axs[0].set_title('Train prediction')
    axs[0].axis('off')

    axs[1].imshow(label, cmap='jet', vmin=0, vmax=6)
    axs[1].set_title('label')
    axs[1].axis('off')

    if show_prob_maps:
        fig, axs = plt.subplots(
            nrows=2,
            ncols=4,
            figsize=(4 * img_x, 2 * img_y))
        plt.subplots_adjust(wspace=0.05)

        for y in range(2):
            for x in range(4):
                i = 4 * y + x
                if i < 7:
                    axs[y, x].imshow(probs[i], cmap='pink', vmin=0, vmax=1)
                    axs[y, x].set_title(class_names[i])
                else:
                    blank = np.ones_like(probs[0])
                    axs[y, x].imshow(blank, cmap='gray', vmin=0)
                axs[y, x].axis('off')

    ''' Compute segmentation on eval data example '''

    eval_seg, eval_probs = segment(
        net_sources=net_sources,
        image_source=os.path.join(data_dir, 'eval-images.tif'),
        output_dir=None,
        label_source=os.path.join(data_dir, 'eval-labels.tif'))

    z_train = z if z else 4
    seg = eval_seg[z_train]
    probs = eval_probs[:, z_train, ...]
    label = tif.imread(os.path.join(data_dir, 'eval-labels.tif'))[z_train]

    img_y = int(seg.shape[0] / dpi)
    img_x = int(seg.shape[1] / dpi)

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(2 * img_x, img_y))

    plt.subplots_adjust(wspace=0.05)

    axs[0].imshow(seg, cmap='jet', vmin=0, vmax=6)
    axs[0].set_title('Eval prediction')
    axs[0].axis('off')

    axs[1].imshow(label, cmap='jet', vmin=0, vmax=6)
    axs[1].set_title('label')
    axs[1].axis('off')

    if show_prob_maps:
        fig, axs = plt.subplots(
            nrows=2,
            ncols=4,
            figsize=(4 * img_x, 2 * img_y))
        plt.subplots_adjust(wspace=0.05)

        for y in range(2):
            for x in range(4):
                i = 4 * y + x
                if i < 7:
                    axs[y, x].imshow(probs[i], cmap='pink', vmin=0, vmax=1)
                    axs[y, x].set_title(class_names[i])
                else:
                    blank = np.ones_like(probs[0])
                    axs[y, x].imshow(blank, cmap='gray', vmin=0)
                axs[y, x].axis('off')


