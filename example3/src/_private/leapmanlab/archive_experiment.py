"""Archive a copy of an experiment's source code files.

Useful for versioning experiments across multiple runs. By default, saves all
.py files contained in an experiment's directory and subdirectories,
excluding the 'output' subdirectory.

"""
import os
import shutil
import time
import uuid

from typing import Optional, Sequence,  Union


def archive_experiment(experiment_dir: str,
                       dst_dir: str,
                       save_extensions: Union[str, Sequence[str]]='py',
                       exclude_dirs: Union[str, Sequence[str]]='output',
                       archive_format: str='zip',
                       base_name: Optional[str]=None):
    """Archive a copy of an experiment's source code files.

    Useful for versioning experiments across multiple runs. By default, saves
    all .py files contained in an experiment's directory and subdirectories,
    excluding the 'output' subdirectory.

    Args:
        experiment_dir (str): Experiment base directory.
        dst_dir (str): Destination directory for the archived experiment.
        save_extensions (Union[str, Sequence[str]]): One or more file
            extensions. This function will save files in `experiment_dir` iff
            their file extension is contained in `save_extensions`. This
            function will save subdirectories of `experiment_dir` iff they
            contain a saved file.
        exclude_dirs (Union[str, Sequence[str]]): Subdirectories of the
            experiment_dir to exclude. By default, just excludes the 'output'
            directory. It is strongly recommended that you exclude the
            'output' subdirectory, even if you elect to exclude others as well.
        archive_format (str): Compression format for the archived experiment
            files. This function uses `shutil.make_archive` for compression,
            so `archive_format` should be a format available to shutil.
        base_name (Optional[str]): Base name for the experiment's archive (a
            timestamp is also appended). If None is supplied, use the last
            part of os.path.abspath(experiment_dir).

    Returns: None

    """
    # Format save_extensions for consistency
    # Make into a sequence
    if isinstance(save_extensions, str):
        save_extensions = [save_extensions]
    # Drop any .'s
    save_extensions = [s.strip('.') for s in save_extensions]
    # Format exclude_dirs for consistency
    if isinstance(exclude_dirs, str):
        exclude_dirs = [exclude_dirs]
    # Get default base name
    if base_name is None:
        experiment_path = os.path.abspath(experiment_dir)
        base_name = [p for p in experiment_path.split('/') if p][-1]

    # Full name of the archive name uses a time stamp
    timestamp = time.strftime('%b%d%Y_%H%M%S')
    archive_name = f'{base_name}_{timestamp}'

    # Use a temporary folder to create the archive
    tmp_folder = f'/tmp/{str(uuid.uuid4())}'
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    tmp_experiment = os.path.join(tmp_folder, archive_name)
    os.makedirs(tmp_experiment)

    # Recurse through the experiment directory and non-'output' subdirectories,
    # saving files to the temporary folder
    dirs_to_check = [experiment_dir]
    while len(dirs_to_check) > 0:
        # A directory to check (DTC), relative to the experiment_dir
        dtc = dirs_to_check.pop(0)
        # Full path to the DTC
        full_dtc = dtc if dtc == experiment_dir \
            else os.path.join(experiment_dir, dtc)
        # List of all files and folders in the DTC
        dlist = os.listdir(full_dtc)
        # List of all files in the DTC
        files = [d for d in dlist
                 if os.path.isfile(os.path.join(full_dtc, d))]
        # Check each file to see if it should be archived.
        for f in files:
            if f.split('.')[-1] in save_extensions:
                # Recreate the file structure inside experiment_dir, up to
                # the folder containing f
                tmp_save_dir = tmp_experiment if dtc == experiment_dir \
                    else os.path.join(tmp_experiment, dtc)
                os.makedirs(tmp_save_dir, exist_ok=True)
                # Save a copy of f
                shutil.copy2(os.path.join(full_dtc, f), tmp_save_dir)

        # Get non-excluded subdirectories
        subdirs = [d for d in dlist
                   if os.path.isdir(os.path.join(full_dtc, d))
                   and d not in exclude_dirs]
        # Track subdirectories as paths relative to the experiment dir
        if dtc != experiment_dir and len(subdirs) > 0:
            subdirs = [os.path.join(dtc, d) for d in subdirs]

        dirs_to_check += subdirs

    # At this point, all archivable files and folders are saved in tmp_folder.
    # Create an archive, coincidentally the same name as tmp_experiment's path
    tmp_archive = tmp_experiment[:]
    shutil.make_archive(tmp_archive, archive_format, tmp_folder, archive_name)
    # Get the full name of the archive. There should only be one file in
    # tmp_experiment
    tmp_archive_full = [f for f in os.listdir(tmp_folder)
                        if os.path.isfile(os.path.join(tmp_folder, f))][0]
    # Copy the archive to its destination
    os.makedirs(dst_dir, exist_ok=True)
    shutil.move(os.path.join(tmp_folder, tmp_archive_full),
                os.path.join(dst_dir, tmp_archive_full),
                copy_function=shutil.copyfile)
    # Remove the temporary folder
    shutil.rmtree(tmp_folder)

    pass
