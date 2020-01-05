"""Create a training weight volume for a dataset.

Status:
    (November 15 2018) Created.

"""
import numpy as np

from typing import Sequence


def class_balance_weights(label_volume: np.ndarray,
                          max_weight: float=1.) -> np.ndarray:
    """Create class frequency-balancing weights for a given label volume.

    Args:
        label_volume (np.ndarray): A volume of class labels. Compute class
            frequencies by counting the number of each label.
        max_weight (float): Scale the weight volume to have this maximum value.

    Returns:
        (np.ndarray): A weight volume with the same shape as `label_volume`,
            with class frequency-balancing weights.

    """
    # Get the labels in the label volume
    labels = np.unique(label_volume)
    vol_size = label_volume.size

    # Number of instances of each class in the training data
    class_counts = []
    for n in labels:
        class_instances = np.where(label_volume == n)[0]
        class_counts.append(len(class_instances))
    # Get the inverse of the class count proportions
    inv_proportions = [vol_size / float(c) for c in class_counts]
    # Scale the inverse proportions to create weights so that the maximum
    # value is `max_weight`
    ip_max = max(inv_proportions)
    class_weights = [ip * max_weight / ip_max for ip in inv_proportions]

    # Use the weights to create a weight volume
    weight_volume = np.zeros_like(label_volume).astype(np.float64)
    for i, v in np.ndenumerate(label_volume):
        weight_volume[i] = class_weights[v]

    return weight_volume


def diffusive_edge_weights(label_volume: np.ndarray,
                           source_labels: Sequence[int],
                           target_labels: Sequence[int],
                           max_weight: float=25.,
                           c_diffuse: float=0.2,
                           c_decay: float=0.98,
                           threshold: float=0.2,
                           n_steps: int=50,
                           symmetric: bool=True) -> np.ndarray:
    """Simulate a few steps of a diffusion equation to create weights around
    the boundaries between different labels in `label_volume`.

    Using the voxels with labels in `source_labels` as diffusion sources with
    with value 1, perform `n_steps` steps of a 2D FDM heat equation within
    the region of voxels with labels in `target_labels`.

    Args:
        label_volume (np.ndarray): A volume of class labels.
        source_labels (Sequence[int]): Voxels with labels in this list are
            sources for the diffusion equation updates.
        target_labels: Voxels with labels in this list are the only ones
            updated during the diffusion equation updates.
        max_weight (float): Scale weights so that this is the maximum value.
        c_diffuse (float): Diffusion coefficient. Larger value -> faster
            diffusion.
        c_decay (float): Decay coefficient. The diffusion map is multiplied
            by this value at each step of the diffusion equation update.
        threshold (float): After diffusion equation update finishes, replace
            the resulting `weight_volume` with
            `max(weight_volume - threshold, 0)`.
        n_steps (int): Number of diffusion update steps to run.
        symmetric (bool): If True, also run the diffusion update with the
            source and target labels swapped, and add the two resulting
            volumes together. This has the effect of putting weights on both
            sides of a source-target label boundary. If False, this is not
            done, and weights are just on the side of a label boundary
            containing labels in `target_labels`.

    Returns:
        (np.ndarray): A weight volume with the same shape as `label_volume`,
            with weights around label boundaries.

    """
    if symmetric:
        return _diffuse(label_volume, source_labels, target_labels, max_weight,
                        c_diffuse, c_decay, threshold, n_steps) + \
               _diffuse(label_volume, target_labels, source_labels, max_weight,
                        c_diffuse, c_decay, threshold, n_steps)
    else:
        return _diffuse(label_volume, source_labels, target_labels, max_weight,
                        c_diffuse, c_decay, threshold, n_steps)


def _diffuse(label_volume: np.ndarray,
             source_labels: Sequence[int],
             target_labels: Sequence[int],
             max_weight: float,
             c_diffuse: float,
             c_decay: float,
             threshold: float,
             n_steps: int) -> np.ndarray:
    """The actual diffusion equation update for `diffusive_edge_weights`.

    Args:
        label_volume (np.ndarray): A volume of class labels.
        source_labels (Sequence[int]): Voxels with labels in this list are
            sources for the diffusion equation updates.
        target_labels: Voxels with labels in this list are the only ones
            updated during the diffusion equation updates.
        max_weight (float): Scale weights so that this is the maximum value.
        c_diffuse (float): Diffusion coefficient. Larger value -> faster
            diffusion.
        c_decay (float): Decay coefficient. The diffusion map is multiplied
            by this value at each step of the diffusion equation update.
        threshold (float): After diffusion equation update finishes, replace
            the resulting `diffusion_volume` with
            `max(diffusion_volume - threshold, 0)`.
        n_steps (int): Number of diffusion update steps to run.

    Returns:
        (np.ndarray): A weight volume with the same shape as `label_volume`,
            with weights around label boundaries.)
    """
    # Convert the label volume to float
    label_volume = label_volume.astype(np.float64)
    # Make 3D if necessary
    if label_volume.ndim == 2:
        label_volume = np.expand_dims(label_volume, 0)

    # Map `label_volume` values to diffusion volume values - `source_label`
    # values are 1, all others are 0
    diffusion_volume = np.isin(label_volume, source_labels).astype(np.float64)

    # Compute the diffusion kernel and its Fourier transform
    k = np.zeros_like(diffusion_volume[0, ...])
    k[0, 0] = -1
    k[0, 1] = 1 / 6
    k[1, 0] = 1 / 6
    k[0, -1] = 1 / 6
    k[-1, 0] = 1 / 6
    k[1, 1] = 0.5 / 6
    k[1, -1] = 0.5 / 6
    k[-1, 1] = 0.5 / 6
    k[-1, -1] = 0.5 / 6
    # 2D FFT of the diffusion kernel
    f_k = np.fft.fft2(k)

    # Diffusion is performed in 2D, per z-slice
    for z in range(label_volume.shape[0]):
        diffusion_z = diffusion_volume[z, ...]

        # Track heat source indices, so that their value can be reset to 1
        # after each iteration
        source_idxs = np.where(diffusion_z == 1)

        # Perform diffusion
        for n in range(n_steps):
            f_diffusion = np.fft.fft2(diffusion_z)
            f_update = f_k * f_diffusion
            diffusion_z = c_decay * np.real(diffusion_z +
                                            c_diffuse * np.fft.ifft2(f_update))
            diffusion_z = np.maximum(diffusion_z, 0)
            diffusion_z[source_idxs] = 1

        # Threshold
        diffusion_z = np.maximum(diffusion_z - threshold, 0)

        # Update diffusion volume
        diffusion_volume[z, ...] = diffusion_z[...]

    # Mask `diffusion_volume` values so that only the target index weight
    # values are nonzero
    target_mask = np.isin(label_volume, target_labels).astype(np.float64)
    diffusion_volume = np.multiply(diffusion_volume, target_mask)

    # Scale the weight values and return
    return np.squeeze(diffusion_volume * max_weight / diffusion_volume.max())
