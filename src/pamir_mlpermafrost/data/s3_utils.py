import pathlib

import dotenv

_dotenv_path = dotenv.find_dotenv()
_base_path = pathlib.Path(_dotenv_path).resolve().parent
assert dotenv.load_dotenv(_dotenv_path), (
    "Make sure you have a .env file with your S3 credentials."
)


def open_s3_file(url, opener):
    import warnings

    kwargs = get_fsspec_kwargs()
    
    if not url.startswith('simplecache'):
        kwargs.pop('simplecache')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opened = opener(url, storage_options=kwargs)
    return opened


def download_file_local(url)->str:
    import fsspec
    kwargs = get_fsspec_kwargs()
    
    if not url.startswith('simplecache'):
        url = "simplecache::" + url

    fname = fsspec.open_local(url, **kwargs)
    return fname

def open_file_local(url, opener):
    fname = download_file_local(url)
    return opener(fname)


def open_s3zarr(url):
    import xarray as xr
    ds = open_s3_file(url, xr.open_zarr)
    return ds


def get_fsspec_kwargs(endpoint_url_secret_name: str = "S3_ENDPOINT_URL"):
    return {
        **get_fsspec_s3_kwargs(endpoint_url_secret_name),
        **get_fsspec_simplecache_kwargs(),
    }


def get_fsspec_s3_kwargs(endpoint_url_secret_name: str = "S3_ENDPOINT_URL"):
    """
    Get the fsspec kwargs for S3 access.

    Returns
    -------
    dict
        A dictionary containing the S3 endpoint URL.
    """
    return {
        "s3": {"endpoint_url": dotenv.get_key(_dotenv_path, endpoint_url_secret_name)}
    }


def get_fsspec_simplecache_kwargs():
    """
    Get the fsspec kwargs for simplecache.

    Returns
    -------
    dict
        A dictionary containing the cache storage path and asynchronous flag.
    """
    return {
        "simplecache": {
            "cache_storage": str(_base_path / "data/cache"),
            "asynchronous": True,
            "cache_check": True,  # Check if the cache is valid before using it
        }
    }
