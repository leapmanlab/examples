"""Tools for downloading the (Guay et al., 2019) dataset if needed.

"""
import os
import os.path
import hashlib
import gzip
import errno
import tarfile
from tqdm import tqdm
import zipfile

# Last updated 12:01 02 January 2020
dataset_info = {
    'url': "https://www.dropbox.com/s/68yclbraqq1diza/platelet_data_1219.zip?dl=1",
    'filename': 'platelet_data_1219.zip',
    'md5': '253310093ee2c00304fbabdb0c7e3a4a',
    'filedirs': ['platelet_data']}

download_md5s = {
    'train-images.tif': '940ffa25038e9e9ca0ae86bc7e75553c',
    'eval-labels.tif': 'abec7b896c2b57ebfc6bccac77cdcd2f',
    'test-images.tif': 'd3f296cbedb7c3240e6be0e9e7217f3a',
    'train-labels.tif': 'dc72be0033f8ce409cac0202928873ba',
    'eval-images.tif': '0fd6642cd7cf0a1e758d7d48b3ac5b57',
    'test-labels.tif': 'dd21ca5031d74b6f8be9f954e8fe7eac',
    'train-error-weights.tif': 'dd30e90f77b2141dd01ab7b76e238163'}


def download_data_if_missing(download_dir: str):
    """Download the data used in (Guay et al., 2019) to a specified
    download directory.

    Args:
        download_dir (str): Directory in which data will be downloaded
            and extracted.

    Returns: None

    """
    if verify_download(download_dir):
        print(f'Folder platelet_data in {download_dir} matches online copy.')
        print(f'Skipping download.')
    else:
        download_and_extract(
            dataset_info['url'],
            download_dir,
            filename=dataset_info['filename'],
            md5=dataset_info['md5'])

    pass


def verify_download(download_dir: str) -> bool:
    """Verify an existing download matches the data used in
    (Guay et al., 2019).

    Args:
        download_dir (str): Directory containing the downloaded
            'platelet_data' folder.

    Returns:
        (bool): True if download matches the online data, else False. 

    """
    if '~' in download_dir:
        download_dir = os.path.expanduser(download_dir)

    # Check if download dir exists
    if not os.path.isdir(download_dir):
        # Download directory doesn't exist, so it can't contain the
        # downloaded data
        return False

    # Check if platelet data dir exists
    platelet_dir = os.path.join(download_dir, 'platelet_data')
    if not os.path.isdir(platelet_dir):
        # Platelet data directory doesn't exist, so it can't contain the
        # downloaded data
        return False

    # Check if each data file exists
    data_files = [os.path.join(platelet_dir, k) 
                  for k in download_md5s.keys()]
    all_files_exist = all([os.path.exists(f) for f in data_files])
    if not all_files_exist:
        return False

    # Check if each data file's MD5 hash matches the recorded hash
    for k in download_md5s.keys():
        data_file = os.path.join(platelet_dir, k)
        if calculate_md5(data_file) != download_md5s[k]:
            return False

    # If we made it this far, all files exist and have matching MD5 hashes, so
    # the existing data matches the online data
    return True


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=True):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def download_and_extract(url, 
                         download_root,
                         filename=None,
                         md5=None,
                         remove_finished=True):

    download_root = os.path.expanduser(download_root)
    os.makedirs(download_root, exist_ok=True)
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, download_root))
    extract_archive(archive, download_root, remove_finished)
    print("Finished")
