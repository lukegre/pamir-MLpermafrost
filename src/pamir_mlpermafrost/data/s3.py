import pathlib

import dotenv
import xarray as xr

_dotenv_path = dotenv.find_dotenv()
_base_path = pathlib.Path(_dotenv_path).resolve().parent
assert dotenv.load_dotenv(_dotenv_path), (
    "Make sure you have a .env file with your S3 credentials."
)


fsspec_kwargs = {
    "s3": {"endpoint_url": dotenv.get_key(_dotenv_path, "S3_ENDPOINT_URL")},
    "simplecache": {
        "cache_storage": str(_base_path / "data/cache"),
        "asynchronous": True,
    },
}
