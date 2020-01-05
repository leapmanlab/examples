"""Load and prepare datasets saved on disk for training, evaluation,
or prediction with a GeneNet.

"""
import logging
import math
import os

import metatree as mt
import numpy as np
import tensorflow as tf
import tifffile as tif

from scipy.ndimage.interpolation import zoom, map_coordinates
from scipy.ndimage.filters import gaussian_filter

from .GeneNet import GeneNet

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, \
    TypeVar, Union

# DataRanges specify which portion of a 3D volume to use for training or
# validation. Each DataRange is either a tuple of 2 slice objects for x and y
# ranges, or a tuple of 3 slice objects for z, x, and y ranges.
DataRange = TypeVar(Union[Tuple[slice, slice], Tuple[slice, slice, slice]])
# A source for a data volume can either be a string, indicating a path to a
# file saved to disk, or it can be a NumPy array.
VolSource = TypeVar(Union[str, np.ndarray])

logger = logging.getLogger('genenet.DataHandler')


class DataHandler:
    """Loads and processes data for segmentation and network training.

    TODO
    Attributes:

    """
    def __init__(self, *args, **kwargs):
        """Constructor.

        The user may pass arguments to load data into the DataHandler using
        DataHandler.load(), which will be called at the end of construction
        if arguments are supplied.

        """
        # Training/prediction data volume
        self.train_volume: np.ndarray = None
        # Training label volume
        self.train_label_volume: Optional[np.ndarray] = None
        # Per-voxel training error weighting volume
        self.train_weight_volume: Optional[np.ndarray] = None
        # Validation data volume
        self.eval_volume: Optional[np.ndarray] = None
        # Validation label volume
        self.eval_label_volume: Optional[np.ndarray] = None

        # Number of label classes
        self.n_classes: int = None

        # RandomState used to control random number generation
        self.random_state = np.random.RandomState()

        # Image augmentation instance_settings for use during training
        self.augmentation_settings: Dict[str, Any] = None

        # Load data if arguments are supplied
        if len(args) > 0 or len(kwargs) > 0:
            self.load(*args, **kwargs)

        pass

    def n_samples_per_epoch(self,
                            graph_source: Union[GeneNet, mt.GeneGraph],
                            train_window_spacing: Optional[Sequence[int]]=None,
                            mode:
                            tf.estimator.ModeKeys=tf.estimator.ModeKeys.TRAIN,
                            forward_window_overlap:
                                Optional[Sequence[int]]=None,
                            prediction_volume: Optional[np.ndarray]=None) \
            -> int:
        """Compute the number of training samples in an epoch, given a
        network's GeneGraph (for input and output data shapes) and a choice
        of training window spacing.

        Args:
            graph_source (Union[GeneNet, mt.GeneGraph]): A GeneGraph or
                GeneNet that this `input_fn` is being used with. Needed for
                getting computation graph input and output shapes.
            train_window_spacing (Sequence[int]): Spacing between the corners of
                consecutive windows along each spatial axis in training mode.
                Should be in (dx, dy) format for 2D windows and
                (dz, dx, dy) format for 3D windows. Example: If
                `train_window_spacing=[1, 80, 80]`, the first window corner
                will be at [0, 0, 0], and the second will be at [0, 0, 80].
            mode (tf.estimator.Modekeys): TensorFlow Estimator mode.
            forward_window_overlap (Sequence[int]): Overlap between
                successive windows during forward (inference) passes through
                a network. Used to mitigate edge effects caused by
                partitioning a large volume into independent windows for
                segmentation. Default is no overlap.
            prediction_volume (Optional[np.ndarray]): In PREDICT mode, one may
                wish to use data other than this network's training data as
                input for a GeneNet forward pass. Supply that data here.

        Returns:

        """
        if isinstance(graph_source, mt.GeneGraph):
            gene_graph = graph_source
        elif isinstance(graph_source, GeneNet):
            gene_graph = graph_source.gene_graph
        else:
            raise TypeError('Input graph_source must be a GeneNet or '
                            'gn.GeneGraph')

        if forward_window_overlap is None:
            forward_window_overlap = [0, 0, 0]

        if prediction_volume is None:
            prediction_volume = self.train_volume

        # Get network output window spatial shapes
        window_shape_out = gene_graph.output_shape()
        # Make 2D stuff 3D
        if len(window_shape_out) == 2:
            window_shape_out = [1] + list(window_shape_out)
        if train_window_spacing is not None and len(train_window_spacing) == 2:
            train_window_spacing = [1] + list(train_window_spacing)

        # Get the volumes and window spacing data needed for the specified mode
        if mode == tf.estimator.ModeKeys.EVAL:
            data_volume = self.eval_volume
            window_spacing = [s - o for s, o in zip(window_shape_out,
                                                    forward_window_overlap)]
        elif mode == tf.estimator.ModeKeys.PREDICT:
            data_volume = prediction_volume
            window_spacing = [s - o for s, o in zip(window_shape_out,
                                                    forward_window_overlap)]
        elif mode == tf.estimator.ModeKeys.TRAIN:
            data_volume = self.train_volume
            if train_window_spacing is None:
                # Default window spacing: Half the window size along each axis
                train_window_spacing = [max(1, int(s / 2))
                                        for s in window_shape_out]
            window_spacing = train_window_spacing[:]
        else:
            raise ValueError(f'Mode {mode} not recognized')

        # Shape of the volumes, and number of dimensions
        # Currently assumes that data will be 2D single channel (2d shape),
        # 3D single channel (3d shape), or 3D multichannel (4d shape). 2D
        # multichannel data is not supported
        # Is the spatial shape 3D?
        if data_volume.ndim == 3:
            # Add a single channel dimension
            data_volume = np.expand_dims(data_volume, 0)
        elif data_volume.ndim == 2:
            # Add a singleton z spatial dimension
            data_volume = np.expand_dims(data_volume, 0)
            # Add a singleton channel dimension
            data_volume = np.expand_dims(data_volume, 0)
        # Volumes' spatial shape
        spatial_shape = data_volume.shape[1:]

        # Generate window corner points
        corner_points = self._gen_corner_points(mode,
                                                spatial_shape,
                                                window_shape_out,
                                                window_spacing)

        # Calculate the number of windows
        n_windows = 1
        for p in corner_points:
            n_windows *= len(p)

        return n_windows

    def input_fn(self,
                 mode: tf.estimator.ModeKeys,
                 graph_source: Union[GeneNet, mt.GeneGraph],
                 train_window_spacing: Optional[Sequence[int]] = None,
                 forward_window_overlap: Optional[Sequence[int]] = None,
                 prediction_volume: Optional[np.ndarray] = None,
                 num_epochs: Optional[int] = None,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 random_seed: int = None,
                 augmentation_settings: Optional[Dict[str, Any]] = None) -> \
            Callable[[], None]:
        """

        Args:
            mode (tf.estimator.Modekeys): TensorFlow Estimator mode.
            graph_source (Union[GeneNet, mt.GeneGraph]): A GeneGraph or
                GeneNet that this `input_fn` is being used with. Needed for
                getting computation graph input and output shapes.
            train_window_spacing (Sequence[int]): Spacing between the corners of
                consecutive windows along each spatial axis in training mode.
                Should be in (dx, dy) format for 2D windows and
                (dz, dx, dy) format for 3D windows. Example: If
                `train_window_spacing=[1, 80, 80]`, the first window corner
                will be at [0, 0, 0], and the second will be at [0, 0, 80].
            forward_window_overlap (Sequence[int]): Overlap between
                successive windows during forward (inference) passes through
                a network. Used to mitigate edge effects caused by
                partitioning a large volume into independent windows for
                segmentation. Default is no overlap.
            prediction_volume (Optional[np.ndarray]): In PREDICT mode, one may
                wish to use data other than this network's training data as
                input for a GeneNet forward pass. Supply that data here.
            num_epochs (Optional[int]): Number of epochs to iterate through
                the data. If None, specify iterations in the Estimator training
                function instead.
            batch_size (int): Size of each training minibatch.
            shuffle (bool): If True, shuffle data batches.
            random_seed (int): Seed to control the Numpy random number
                generator.
            augmentation_settings (Optional[Dict[str, Any]]): Image
                augmentation instance_settings used during training.

        Returns:
            (Callable[[], None]): A tf.estimator.Estimator training input_fn.

        """
        if isinstance(graph_source, mt.GeneGraph):
            gene_graph = graph_source
        else:
            gene_graph = graph_source.gene_graph

        if forward_window_overlap is None:
            forward_window_overlap = [0, 0, 0]

        # Get network input and output window spatial shapes
        window_shape_in = gene_graph.input_shape
        window_shape_out = gene_graph.output_shape()
        # Note whether the window is 2D or 3D for later
        window_is_3d = len(window_shape_in) == 3
        # Make 2D stuff 3D
        if len(window_shape_in) == 2:
            window_shape_in = [1] + list(window_shape_in)
        if len(window_shape_out) == 2:
            window_shape_out = [1] + list(window_shape_out)
        if train_window_spacing is not None and len(train_window_spacing) == 2:
            train_window_spacing = [1] + list(train_window_spacing)

        # Difference in shape between input and output windows
        d_shape = [int((i - o) / 2) for i, o in zip(window_shape_in,
                                                    window_shape_out)]

        if prediction_volume is None:
            prediction_volume = self.train_volume

        # Set the random seed
        if random_seed is None:
            # Create a new seed
            random_seed = np.random.randint(np.iinfo(np.int32).max)

        self.random_state.seed(random_seed)
        logger.info(f'Seeded DataHandler.random_state with {random_seed}')

        # Image augmentation default instance_settings
        if augmentation_settings is None:
            augmentation_settings = {}
        if 'deform_scale' not in augmentation_settings:
            augmentation_settings['deform_scale'] = 40
        if 'deform_alpha' not in augmentation_settings:
            augmentation_settings['deform_alpha'] = 20
        if 'deform_sigma' not in augmentation_settings:
            augmentation_settings['deform_sigma'] = 0.6
        if 'do_augmentation' not in augmentation_settings:
            augmentation_settings['do_augmentation'] = True
        if 'do_deformation' not in augmentation_settings:
            augmentation_settings['do_deformation'] = True
        if 'do_reflect' not in augmentation_settings:
            augmentation_settings['do_reflect'] = True
        if 'do_brightness' not in augmentation_settings:
            augmentation_settings['do_brightness'] = True
        if 'do_contrast' not in augmentation_settings:
            augmentation_settings['do_contrast'] = True
        if 'eps_brightness' not in augmentation_settings:
            augmentation_settings['eps_brightness'] = 0.12
        if 'eps_contrast' not in augmentation_settings:
            augmentation_settings['eps_contrast'] = 0.2

        self.augmentation_settings = augmentation_settings

        # Get the volumes and window spacing data needed for the specified mode
        if mode == tf.estimator.ModeKeys.EVAL:
            data_volume = self.eval_volume
            label_volume = self.eval_label_volume
            volumes = [data_volume, label_volume]
            window_spacing = [s - o for s, o in zip(window_shape_out,
                                                    forward_window_overlap)]
        elif mode == tf.estimator.ModeKeys.PREDICT:
            data_volume = prediction_volume
            volumes = [data_volume]
            window_spacing = [s - o for s, o in zip(window_shape_out,
                                                    forward_window_overlap)]
        elif mode == tf.estimator.ModeKeys.TRAIN:
            data_volume = self.train_volume
            label_volume = self.train_label_volume
            weight_volume = self.train_weight_volume
            if augmentation_settings['do_augmentation'] and \
               augmentation_settings['do_deformation']:
                volumes = \
                    self._deform([data_volume, label_volume, weight_volume])
            else:
                volumes = [data_volume, label_volume, weight_volume]
            data_volume = volumes[0]
            if train_window_spacing is None:
                # Default window spacing: Half the window size along each axis
                train_window_spacing = [max(1, int(s / 2))
                                        for s in window_shape_out]
            window_spacing = train_window_spacing[:]
        else:
            raise ValueError(f'ModeKey {mode} not recognized.')

        # Shape of the volumes, and number of dimensions
        # Currently assumes that data will be 2D single channel (2d shape),
        # 3D single channel (3d shape), or 3D multichannel (4d shape). 2D
        # multichannel data is not supported
        # Is the spatial shape 3D?
        vol_is_3d = data_volume.ndim > 2
        if data_volume.ndim == 4:
            n_channels = data_volume.shape[0]
        elif data_volume.ndim == 3:
            # Add a single channel dimension
            data_volume = np.expand_dims(data_volume, 0)
            n_channels = 1
        elif data_volume.ndim == 2:
            # Add a singleton z spatial dimension
            data_volume = np.expand_dims(data_volume, 0)
            # Add a singleton channel dimension
            data_volume = np.expand_dims(data_volume, 0)
            n_channels = 1
        else:
            raise ValueError(f'Data volume ndim of {data_volume.ndim} not '
                             f'supported')
        volumes[0] = data_volume
        # Volumes' spatial shape
        spatial_shape = data_volume.shape[1:]
        # Number of spatial dimensions
        nsdim = len(spatial_shape)

        # Generate window corner points
        corner_points = self._gen_corner_points(mode,
                                                spatial_shape,
                                                window_shape_out,
                                                window_spacing)

        # Create windows

        # Calculate the number of windows
        n_windows = 1
        for p in corner_points:
            n_windows *= len(p)

        # Shape of each batch source array. Add in the channel axis:
        # Note the format is either NCXY or NCZXY
        array_shape_in = [n_windows] + [n_channels] + list(window_shape_in)
        array_shape_out = [n_windows] + list(window_shape_out)
        # Remove the z axis if the window is not 3D
        if not window_is_3d or not vol_is_3d:
            array_shape_in.pop(2)
            array_shape_out.pop(1)
        # Create source arrays
        source_arrays = []
        # Batch source arrays have shape array_shape_in for data volumes,
        # array_shape_out for other volumes
        for v in volumes:
            if v is data_volume:
                source_arrays.append(np.zeros(array_shape_in, dtype=v.dtype))
            else:
                source_arrays.append(np.zeros(array_shape_out, dtype=v.dtype))

        # For convenience
        dzi = window_shape_in[0]
        dxi = window_shape_in[1]
        dyi = window_shape_in[2]
        dzo = window_shape_out[0]
        dxo = window_shape_out[1]
        dyo = window_shape_out[2]
        zs = spatial_shape[0]
        xs = spatial_shape[1]
        ys = spatial_shape[2]

        def get_range(n0: int, n1: int, ns: int) -> List[int]:
            """Get a window range along axis n, accounting for reflecting
            boundary conditions when the range is out-of-bounds within the
            source volume.

            Args:
                n0 (int): Window starting point.
                n1 (int): Window ending point.
                ns (int): Source volume size along axis n.

            Returns:
                (List[int]): Window range.

            """
            # Return a range as a list
            def lrange(a, b, n=1) -> List[int]:
                return list(range(a, b, n))
            # Get the in-bounds part of the range
            n_range = lrange(max(0, n0), min(ns, n1))
            # Handle out-of-bounds indices by reflection across boundaries
            if n0 < 0:
                # Underflow
                n_range = lrange(-n0, 0, -1) + n_range
            if n1 > ns:
                # Overflow
                n_range = n_range + lrange(ns - 1, 2 * ns - n1 - 1, -1)

            return n_range

        # Keep a list of coordinates for each corner, to be returned during
        # PREDICT mode for easy assembly of network pass outputs into a
        # larger image
        all_corners = []

        # Add windows to the batch source arrays
        window_idx = 0
        # Window augmentation parameters
        reflect_axes = []
        brightness_delta = None
        contrast_scale = None
        for z in corner_points[0]:
            for x in corner_points[1]:
                for y in corner_points[2]:
                    all_corners.append([z, x, y])
                    # Generate window augmentation parameters in TRAIN mode
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        if self.augmentation_settings['do_reflect']:
                            # Reflections: 50% chance of a reflection across
                            # each coordinate axis
                            reflect_axes = []
                            for i in range(nsdim):
                                # noinspection PyArgumentList
                                if self.random_state.rand() < 0.5:
                                    reflect_axes.append(i)
                        if self.augmentation_settings['do_brightness']:
                            # Brightness shift: Alter the image brightness by
                            # +-(eps_b * data range)
                            eps_b = self.augmentation_settings['eps_brightness']
                            data_range = data_volume.max() - data_volume.min()
                            delta_max = eps_b * data_range
                            # noinspection PyArgumentList
                            brightness_delta = \
                                (2 * self.random_state.rand() - 1) * delta_max
                        if self.augmentation_settings['do_contrast']:
                            # Contrast shift: Multiply image by a factor
                            # between (1-eps_c) and (1+eps_c)
                            eps_c = self.augmentation_settings['eps_contrast']
                            # noinspection PyArgumentList
                            contrast_scale = \
                                1 + eps_c * (2 * self.random_state.rand() - 1)

                    for i, v in enumerate(volumes):
                        if v is data_volume:
                            # Use window_shape_in-sized windows for the data
                            # volumes
                            z0 = z - d_shape[0]
                            z1 = z0 + dzi
                            x0 = x - d_shape[1]
                            x1 = x0 + dxi
                            y0 = y - d_shape[2]
                            y1 = y0 + dyi
                        else:
                            # Use window_shape_out-sized windows for other
                            # volumes
                            z0 = z
                            z1 = z0 + dzo
                            x0 = x
                            x1 = x0 + dxo
                            y0 = y
                            y1 = y0 + dyo

                        # Compute window ranges
                        z_range = get_range(z0, z1, zs)
                        x_range = get_range(x0, x1, xs)
                        y_range = get_range(y0, y1, ys)

                        # Get window extent from the calculated ranges
                        if v is data_volume:
                            # Take all of channel axis 0
                            window = v.take(z_range, axis=1) \
                                      .take(x_range, axis=2) \
                                      .take(y_range, axis=3)
                        else:
                            # No channel axis
                            window = v.take(z_range, axis=0) \
                                      .take(x_range, axis=1) \
                                      .take(y_range, axis=2)
                        # Perform image augmentation in training mode,
                        # if specified
                        if mode == tf.estimator.ModeKeys.TRAIN and \
                                augmentation_settings['do_augmentation']:
                            # Only a single-channel data volume gets contrast
                            # and brightness shifts
                            if v is data_volume and n_channels == 1:
                                # Shift reflection axes to skip the first
                                # channel axis
                                shifted_axes = [a + 1 for a in reflect_axes]
                                window = self._augment(window,
                                                       shifted_axes,
                                                       brightness_delta,
                                                       contrast_scale)
                            else:
                                window = self._augment(window,
                                                       reflect_axes)
                        if not window_is_3d or not vol_is_3d:
                            squeeze_axis = 1 if v is data_volume else 0
                            # Remove singleton z dimension for 2D windows
                            window = np.squeeze(window, axis=squeeze_axis)

                        # Add window to source array
                        source_arrays[i][window_idx, ...] = window

                    window_idx += 1

        # Return an appropriate input_fn for each mode
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.inputs.numpy_input_fn(
                x={'input': source_arrays[0]},
                y={'label': source_arrays[1]},
                batch_size=batch_size,
                num_epochs=1,
                shuffle=False)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.inputs.numpy_input_fn(
                x={'input': source_arrays[0],
                   'corners': np.array(all_corners)},
                batch_size=batch_size,
                num_epochs=1,
                shuffle=False)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.inputs.numpy_input_fn(
                x={'input': source_arrays[0],
                   'weight_floor': np.array([
                       gene_graph.hyperparameter_configs['optim'][
                           'weight_floor']] * n_windows, dtype='f4')},
                y={'label': source_arrays[1],
                   'weight': source_arrays[2]},
                batch_size=batch_size,
                num_epochs=num_epochs,
                shuffle=shuffle)
        else:
            raise ValueError(f'ModeKey {mode} not recognized.')

    def load(self,
             train_data_dir: Optional[str] = None,
             train_file: Optional[Union[str, np.ndarray]] = 'train-volume.tif',
             train_label_file: Optional[VolSource] = 'label-volume.tif',
             train_weight_file: Optional[VolSource] = 'weight-volume.tif',
             train_range: Optional[DataRange] = None,
             eval_data_dir: Optional[str] = None,
             eval_file: Optional[VolSource] = 'eval-volume.tif',
             eval_label_file: Optional[VolSource] = 'eval-label-volume.tif',
             eval_range: Optional[DataRange] = None):
        """Load data from the file system.

        Args:
            train_data_dir (str): String specifying the location of the source
                training data, training labels, and training error weight files.
            train_file (Optional[VolSource]): Name of the training data image
                volume within the train_data_dir, or a numpy array.
            train_label_file (Optional[VolSource]): Name of the training label
                image volume within the train_data_dir, or a numpy array.
            train_weight_file (Optional[VolSource]): Name of the error
                weighting image volume within the train_data_dir, or a numpy
                array.
            train_range (Optional[DataRange]): Range of the loaded training
                volume to use during the training process, represented as a
                tuple of 2 or 3 slice objects.
            eval_data_dir (Optional[str]): String specifying the location of
                the evaluation data. If None, use `train_data_dir`.
            eval_file (Optional[VolSource]): Name of the evaluation data image
                volume within the train_data_dir, or a numpy array
            eval_label_file (Optional[VolSource]): Name of the evaluation label
                image volume within the train_data_dir, or a numpy array.
            eval_range (Optional[DataRange]): Range of the loaded eval volume to
                use during the eval process, represented as a tuple of 2 or 3
                slice objects.

        Returns: None

        """
        # Use same dir for training and eval data by default
        if eval_data_dir is None:
            eval_data_dir = train_data_dir
        # Prepare the training volume range slicing
        if train_range is None:
            train_range = (slice(None), slice(None), slice(None))
        elif len(train_range) == 2:
            train_range = (slice(None), train_range[0], train_range[1])
        # Prepare the eval volume range slicing
        if eval_range is None:
            eval_range = (slice(None), slice(None), slice(None))
        elif len(eval_range) == 2:
            eval_range = (slice(None), eval_range[0], eval_range[1])

        # Load training, label, and error weight volumes
        if train_data_dir is not None:
            if isinstance(train_file, str):
                self.train_volume = \
                    tif.imread(os.path.join(train_data_dir,
                                            train_file)).astype(np.float32)
            elif isinstance(train_file, np.ndarray):
                self.train_volume = train_file.astype(np.float32)
            else:
                raise TypeError('`train_file` type not recognized')

            # Make a 2D volume 3D
            if self.train_volume.ndim == 2:
                self.train_volume = np.expand_dims(self.train_volume, axis=0)
            # Normalize
            self.train_volume -= np.mean(self.train_volume)
            self.train_volume /= np.std(self.train_volume)
            # Slice
            self.train_volume = self.train_volume[train_range]

        if train_data_dir is not None and train_label_file is not None:
            if isinstance(train_label_file, str):
                self.train_label_volume = \
                    tif.imread(os.path.join(train_data_dir,
                                            train_label_file)).astype(np.int32)
            else:
                self.train_label_volume = train_label_file.astype(np.int32)
            # Make a 2D volume 3D
            if self.train_label_volume.ndim == 2:
                self.train_label_volume = \
                    np.expand_dims(self.train_label_volume, axis=0)
            # Compute number of classes in the label volume. Assumes labels
            # are ints from 0 to n_classes-1
            self.n_classes = self.train_label_volume.max() + 1
            # Slice
            self.train_label_volume = self.train_label_volume[train_range]

        if train_data_dir is not None and train_weight_file is not None:

            if isinstance(train_weight_file, str):
                self.train_weight_volume = \
                    tif.imread(os.path.join(train_data_dir,
                                            train_weight_file)).astype(
                                                np.float32)
            else:
                self.train_weight_volume = train_weight_file.astype(np.float32)

            # Make a 2D volume 3D
            if self.train_weight_volume.ndim == 2:
                self.train_weight_volume = \
                    np.expand_dims(self.train_weight_volume, axis=0)
            # Slice
            self.train_weight_volume = self.train_weight_volume[train_range]

            logger.info(
                f'Loaded training data from '
                f'{os.path.abspath(train_data_dir)}')

        if eval_file is not None:
            if isinstance(eval_file, str):
                self.eval_volume = \
                    tif.imread(os.path.join(eval_data_dir,
                                            eval_file)).astype(np.float32)
            else:
                self.eval_volume = eval_file.astype(np.float32)
            # Make a 2D volume 3D
            if self.eval_volume.ndim == 2:
                self.eval_volume = np.expand_dims(self.eval_volume, axis=0)
            self.eval_volume -= np.mean(self.eval_volume)
            self.eval_volume /= np.std(self.eval_volume)
            # Slice
            self.eval_volume = self.eval_volume[eval_range]

        if eval_label_file is not None:
            if isinstance(eval_label_file, str):
                self.eval_label_volume = \
                    tif.imread(os.path.join(
                        eval_data_dir,
                        eval_label_file)).astype(np.int32)
            else:
                self.eval_label_volume = eval_label_file.astype(np.int32)
            # Make a 2D volume 3D
            if self.eval_label_volume.ndim == 2:
                self.eval_label_volume = np.expand_dims(
                    self.eval_label_volume,
                    axis=0)
            self.n_classes = self.eval_label_volume.max() + 1
            # Slice
            self.eval_label_volume = self.eval_label_volume[eval_range]

        if eval_data_dir is not None:
            logger.info(
                f'Loaded eval data from '
                f'{os.path.abspath(eval_data_dir)}')

        pass

    @staticmethod
    def _augment(window: np.ndarray,
                 reflect_axes: Sequence[int],
                 brightness_delta: Optional[float]=None,
                 contrast_scale: Optional[float]=None) -> np.ndarray:
        """Apply augmentation operations (besides the elastic deformation) to
        an image window.

        Args:
            window (np.ndarray): Image window to augment.

        Returns:
            (np.ndarray): Augmented window.

        """
        # Do reflections
        for a in reflect_axes:
            np.flip(window, a)

        if brightness_delta is not None:
            window += brightness_delta

        if contrast_scale is not None:
            window *= contrast_scale

        return window

    def _deform(self, volumes: Union[np.ndarray, Sequence[np.ndarray]]) -> \
            List[np.ndarray]:
        """Apply an elastic deformation to a collection of image volumes.

        Args:
            volumes (Union[np.ndarray, Sequence[np.ndarray]]): One or more
                numpy arrays to deform.

        Returns:
            (List[np.ndarray]): List of deformed volumes.

        """
        # Make sure xy shape (last two axes) are the same for all volumes
        xy_shapes = [v.shape[-2:] for v in volumes]
        if xy_shapes.count(xy_shapes[0]) != len(xy_shapes):
            # True when xy shapes don't all match
            raise ValueError('Volumes passed to DataHandler._deform() must '
                             'all have the same shape.')

        # Build a new pixel index deformation map
        # Assumed to be the same
        xy_shape = xy_shapes[0]
        deform_map = self._deformation_map(xy_shape)

        deformed_volumes = []

        for volume in volumes:
            shape = volume.shape
            ndim = volume.ndim
            new_vol = np.zeros_like(volume)
            if ndim == 4:
                # 3D multichannel data. Apply 2D deformations to each z slice
                # in each channel of the volume
                for c in range(shape[0]):
                    for z in range(shape[1]):
                        new_vol[c, z, ...] = \
                            map_coordinates(volume[c, z, ...],
                                            deform_map,
                                            order=0).reshape(xy_shape)
            elif ndim == 3:
                # 3D single channel data. Apply 2D deformations to each z slice
                # of the volume
                for z in range(shape[0]):
                    new_vol[z, ...] = map_coordinates(volume[z, ...],
                                                      deform_map,
                                                      order=0).reshape(xy_shape)
            elif ndim == 2:
                # Volume is 2D, deform the whole thing at once
                new_vol = map_coordinates(volume,
                                          deform_map,
                                          order=0).reshape(xy_shape)
            else:
                raise ValueError(f'Cannot deform volume with ndim {ndim}')

            deformed_volumes.append(new_vol)

        return deformed_volumes

    def _deformation_map(self, shape: Sequence[int]) -> \
            Tuple[np.ndarray, np.ndarray]:
        """Create an elastic deformation map,

        Deformation map may be applied to, e.g., image, label,
        and error weight data.
        Adapted from:
        https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
        (October 30, 2016).

        Args:
            shape (Sequence[int]): Shape of the dataset the deformation map
                will be applied to. Should be 2D (for now).

        Return:
            (Tuple[np.ndarray, np.ndarray]): A deformation map, represented as
                a pair of lists of x and y indices.

        """
        if len(shape) != 2:
            raise TypeError(f'Input shape should be length 2, but is {shape}')

        # Controls the spatial scale of the distortions. A large value
        # creates larger-scale distortions. The image's spatial shape should
        # be evenly divisible by the scale value # TODO: Fix this
        scale = self.augmentation_settings['deform_scale']
        # Distortion average magnitude
        alpha = self.augmentation_settings['deform_alpha']
        # Distortion magnitude standard deviation
        sigma = self.augmentation_settings['deform_sigma']

        # Sample Gaussian distribution on a more coarse grid, then upsample
        # and interpolate
        shape_small = [int(s / float(scale)) for s in shape]

        # Calculate x index translations
        dx_small \
            = gaussian_filter((self.random_state.rand(*shape_small) * 2 - 1),
                              sigma,
                              mode='reflect') * alpha
        # Calculate y index translations
        dy_small \
            = gaussian_filter((self.random_state.rand(*shape_small) * 2 - 1),
                              sigma,
                              mode='reflect') * alpha

        # Calculate zoom scale
        scale_up = [i / float(j) for i, j in zip(shape, shape_small)]

        # Upsample and interpolate for smooth index translations
        dx = zoom(dx_small, scale_up)
        dy = zoom(dy_small, scale_up)

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                           indexing='ij')
        indices = np.reshape(np.clip(x + dx, 0, shape[0] - 1), (-1, 1)), \
            np.reshape(np.clip(y + dy, 0, shape[1] - 1), (-1, 1))

        return indices

    def _gen_corner_points(self,
                           mode: tf.estimator.ModeKeys,
                           spatial_shape: Sequence[int],
                           window_shape_out: Sequence[int],
                           window_spacing: Sequence[int]) -> List[List[int]]:
        """Generate lists of Z, X, and Y coordinates for the window
        corner points.

        Returns:
            (List[List[int], List[int], List[int]]): Corner point Z, X,
                and Y coordinate lists, respectively.

        """
        corners = []

        # Number of spatial dimensions
        nsdim = len(spatial_shape)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Corner point calculations for training mode windows

            # Get the bins from which corner points are chosen along each
            # axis
            for k in range(nsdim):
                vs = spatial_shape[k]
                w = window_shape_out[k]
                d = window_spacing[k]

                usable_length = vs - w + 1

                if usable_length == 1:
                    corners_k = [0]
                else:
                    n_bins = int(math.ceil(usable_length / d))

                    bins = [min(d * i, usable_length)
                            for i in range(n_bins + 1)]

                    # Output window corner point coordinates along axis k
                    corners_k = \
                        [self.random_state.randint(bins[i], bins[i + 1])
                         for i in range(n_bins)]

                corners.append(corners_k)

            return corners

        elif mode in [tf.estimator.ModeKeys.EVAL,
                      tf.estimator.ModeKeys.PREDICT]:
            for k in range(nsdim):
                vs = spatial_shape[k]
                w = window_shape_out[k]
                d = window_spacing[k]

                usable_length = vs - w + 1

                if usable_length == 1:
                    # Singleton dimension along axis k
                    corners_k = [0]
                else:
                    n_bins = int(math.ceil(usable_length / d))

                    # Output window corner point coordinates along axis k
                    corners_k = [d * i for i in range(n_bins)]
                    # Additional one to make sure we get full coverage
                    if corners_k[-1] != usable_length - 1:
                        corners_k.append(usable_length - 1)

                corners.append(corners_k)

            return corners

        else:
            raise ValueError(f'ModeKey {mode} not recognized.')
