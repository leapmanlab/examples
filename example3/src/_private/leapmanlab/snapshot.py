"""Contains utility functions for generating web content from project activity.

Currently Linux only, requires that the ImageMagick CLI be installed.

"""
import os
import shutil
import time

import genenet as gn
import matplotlib.pyplot as plt

from typing import Dict, List, Optional

HISTORY = 'file_history.json'
DATA_FILE = 'train-data.tif'
LABEL_FILE = 'train-label.tif'


def snapshot(docs_dir: str,
             net: gn.GeneNet,
             montage_geometry: Optional[str]=None):
    """Creates a snapshot of a net's performance based on the most recent
    training and evaluation data.

    Args:
        docs_dir (str): Directory containing a GitHub Pages source.
        net (gn.GeneNet): GeneNet to create a web summary for.
        montage_geometry (Optional[str]): Image montage geometry. A str fed
            directly into a bash `montage -geometry` command. See
            <http://www.imagemagick.org/Usage/montage/> for more information.
            Default is '240x240\>+2+2'

    Returns: None

    """
    if montage_geometry is None:
        montage_geometry = '240x240\>+2+2'
    net_name = net.name
    net_rel_path = os.path.relpath(net.save_dir, '..').replace('output/', '')
    media_dir = os.path.join(net.save_dir, 'media')
    eval_history_file = os.path.join(net.save_dir, HISTORY)
    # Create a new directory for the webpage
    page_dir = os.path.join(docs_dir, 'snapshots', net_rel_path)
    # Remove any preexisting page
    if os.path.exists(page_dir):
        shutil.rmtree(page_dir)
    os.makedirs(page_dir)
    # Populate a media directory for the page
    images = {}
    page_media_dir = os.path.join(page_dir, 'media')
    os.makedirs(page_media_dir)
    # Divide up the files in the net media dir by type
    media_files = os.listdir(media_dir)

    # Copy eval history to the snapshot
    if os.path.isfile(eval_history_file):
        shutil.copyfile(eval_history_file,
                        os.path.join(page_dir, 'eval_history.json'))

    # Add a Gene graph image path to the images dict
    if 'gene_graph.svg' in media_files:
        images['gene_graph'] = 'gene_graph.svg'
        shutil.copyfile(os.path.join(media_dir, 'gene_graph.svg'),
                        os.path.join(page_media_dir, 'gene_graph.svg'))

    # Sort a list of files in the media dir most-recently modified to
    # least-recently modified
    def mtimesort(li: List) -> List:
        return sorted(li,
                      reverse=True,
                      key=lambda n:
                      os.path.getmtime(os.path.join(media_dir, n)))

    # Get all segmentation files, sorted newest to oldest
    seg_files = mtimesort([f for f in media_files if 'segmentation_' in f])
    # Add a path to the most recent data file to the images dict
    if len(seg_files) > 0:
        # Note order: segmentation, data, label. Will be used implicitly in
        # ~12 lines
        files_to_copy = [seg_files[0], DATA_FILE, LABEL_FILE]
        old_paths = [os.path.join(media_dir, f) for f in files_to_copy]
        new_paths = [os.path.join(page_media_dir, f) for f in files_to_copy]
        # Data and label paths are always the same
        for pold, pnew in zip(old_paths, new_paths):
            shutil.copyfile(pold, pnew)
        pngs = [_to_png(p, True) for p in new_paths]
        # Implicit use of the ordering of `files_to_copy` elements
        images['segmentation'] = pngs[0]
        images['data'] = pngs[1]
        images['label'] = pngs[2]

    # Create training statistic images
    stat_images = []
    ignore_keys = ['global_step', 'name', 'param_seed']
    keys = [k for k in net.eval_history[0] if k not in ignore_keys]
    global_step = net.eval_history[-1]['global_step']
    for key in keys:
        vals = [h[key] for h in net.eval_history]
        plt.figure(frameon=False)
        plt.plot(vals, 'b-.')
        plt.xlabel('Validation instance')
        plt.title(key)
        if key in ['adj_rand_idx', 'mean_iou']:
            plt.ylim(0, 1)
        plt.tight_layout()
        save_name = f'{key}.svg'
        plt.savefig(os.path.join(page_media_dir, save_name),
                    format='svg',
                    frameon=False)
        stat_images.append(save_name)
    if len(stat_images) > 0:
        images['stat_images'] = stat_images
        plt.close('all')

    # Create two summaries of the image data - one with probability maps,
    # one without for a thumbnail
    _summary_montage(page_media_dir,
                     images,
                     geometry=montage_geometry,
                     montage_name='thumbnail')

    _summary_montage(page_media_dir,
                     images,
                     geometry='\<+2+2',
                     montage_name='summary')
    # Summary image gets appended to images, so we can link to it
    images['summary'] = 'summary.png'

    date = time.strftime('%B %d %Y, %H:%M:%S')

    def gen_link(src, idx=None, mini=True):
        if src in images:
            if idx is None:
                img = images[src]
            else:
                img = images[src][idx]
            full_src = os.path.join('media', img)
            mini_txt = 'class="mini"' if mini else ''
            return f'<div class="images">' \
                   f'<a href="{full_src}"><img {mini_txt} src="{full_src}" ' \
                   f'align="center"></a>' \
                   f'<p>{img}. <i>Click to enlarge</i></p>' \
                   f'</div>'
        else:
            return ''

    # Assemble a markdown file
    readme_md_parts = [
        f'# {net_name}',
        f'### Step {global_step} ({date})',
        f'[_Back_](..)',
        f'---',
        f'## Summary',
        f'{gen_link("summary", mini=False)}',
        f'## Gene graph',
        f'{gen_link("gene_graph", mini=False)}',
        f'---']
    if 'stat_images' in images:
        readme_md_parts += [f'## Performance statistics, step {global_step}'] \
            + ['\n'.join([f'{gen_link("stat_images", i)}'
                         for i in range(len(images['stat_images']))])] + ['---']
    readme_md_parts += [
        f'## Data, ground truth label, segmentation',
        f'{gen_link("data")}\n'
        f'{gen_link("label")}\n'
        f'{gen_link("segmentation")}',
        f'---']
    readme_md_parts += ['\n']

    # Generate the readme file
    readme_md = '\n\n'.join(readme_md_parts)
    # Write the readme file
    readme_file = os.path.join(page_dir, 'README.md')
    with open(readme_file, 'w') as fl:
        fl.write(readme_md)


def _to_png(fname: str, del_input: bool=False) -> str:
    """Convert an image with path `fname` to a PNG.

    Args:
        fname (str): Path to the image to be converted.
        del_input (bool): If True, delete the image saved at `fname`.

    Returns:
         (str): Name of the file, with the PNG extension, without its path.

    """
    # Strip the file extension from the path name, add a '.png'
    fname_parts = fname.split('.')
    name = '.'.join(fname_parts[:-1])
    png_name = f'{name}.png'
    # Save only one slice of a multi-slice TIFF
    os.system(f'convert {fname}[0] {png_name}')
    # Delete original file if specified
    if del_input:
        os.remove(fname)
    return png_name.split('/')[-1]


def _summary_montage(image_dir: str,
                     images: Dict[str, str],
                     geometry: str,
                     montage_name: str='montage'):
    """Create an image summary montage.

    Args:
        image_dir (str): Path from the current working directory to the
            directory containing the images to montage-ify.
        images (Dict[str, str]): Dictionary of names of images to summarize.
        geometry (str): Image montage geometry. A str fed
            directly into a bash `montage -geometry` command. See
            <http://www.imagemagick.org/Usage/montage/> for more information.
            Default is '120x120>+2+2'
        montage_name (str): Name of the created montage.

    Returns: None

    """
    # Simplest way to create a montage is to use ImageMagick from the linux
    # command line. Navigate to the directory containing the images
    pwd = os.getcwd()
    os.chdir(image_dir)
    # Assemble images like this: Data - Label - Segmentation - P Maps (if using)
    image_strs = []
    if 'data' in images:
        image_strs.append(images['data'])
    if 'label' in images:
        image_strs.append(images['label'])
    if 'segmentation' in images:
        image_strs.append(images['segmentation'])
    # Join parts
    image_str = ' '.join(image_strs)

    # Create an ImageMagick montage
    cmd = f'montage {image_str} ' \
          f'-geometry {geometry} ' \
          f'-background "rgba(0,0,0,0)" ' \
          f'{montage_name}.png'
    os.system(cmd)

    # Go back to original directory
    os.chdir(pwd)

    pass
