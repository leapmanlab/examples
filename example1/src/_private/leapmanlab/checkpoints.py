"""Functions for saving and loading named GeneNet checkpoints

"""
import os

import genenet as gn

from typing import Optional


def restore_from_checkpoint(
        net_dir: str,
        ckpt_dir: Optional[str] = None,
        *args, **kwargs) -> gn.GeneNet:
    """Wrapper around genenet.GeneNet.restore to handle loading named
    checkpoints.

    Args:
        net_dir (str): Directory containing the save info of the GeneNet to be
            restored.
        ckpt_dir (Optional[str]): Directory containing the checkpoint data (
        saved
            trainable weight values) to be loaded into the network.
        *args: Args passed on to genenet.GeneNet.restore.
        **kwargs: Kwargs passed on to genenet.GeneNet.restore.

    Returns:
        (gn.GeneNet): A restored GeneNet.

    """
    if ckpt_dir:
        load_checkpoint(ckpt_dir, net_dir)
    net = gn.GeneNet.restore(net_dir, *args, **kwargs)
    return net


def save_checkpoint(name: str, output_dir: str):
    """

    Args:
        name:
        output_dir:

    Returns:

    """
    ckpt_dir = os.path.join(output_dir, 'model', 'checkpoints', name)
    if os.path.exists(ckpt_dir):
        os.system(f'rm -f {ckpt_dir}/*')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_files = \
        [f for f in os.listdir(os.path.join(output_dir, 'model'))
         if 'ckpt' in f] + ['graph.pbtxt']
    for f in ckpt_files:
        src = os.path.join(output_dir, 'model', f)
        dst = os.path.join(ckpt_dir)
        os.system(f'cp {src} {dst}')

    pass


def load_checkpoint(
        ckpt_dir: str,
        output_dir: str,
        clear_output_dir: bool = True):
    """

    Args:
        ckpt_dir:
        output_dir:
        clear_output_dir:

    Returns:

    """
    model_dir = os.path.join(output_dir, 'model')
    if clear_output_dir:
        os.system(f'find {model_dir} -maxdepth 1 -type f '
                  f'-exec rm -fv {{}} \;')
        os.system(f'rm -rf {model_dir}/eval')
    os.system(f'cp {ckpt_dir}/* {model_dir}')
    _regen_checkpoint_file(f'{model_dir}')
    pass


def _regen_checkpoint_file(model_dir):
    # Figure out the name of the checkpoint file with some
    # string parsing
    checkpoint_file = [f for f in os.listdir(model_dir)
                       if 'model.ckpt' in f][0]
    name = '.'.join(checkpoint_file.split('.')[:2])
    checkpoint_text = f'model_checkpoint_path: "{name}"\n' \
        f'all_model_checkpoint_paths: "{name}"\n'
    with open(os.path.join(model_dir, 'checkpoint'), 'w') as fd:
        fd.write(checkpoint_text)
    pass
