import pathlib
import sys

import dotenv
from loguru import logger

from . import data, models, preprocessing, utils, viz

_dotenv_path = dotenv.find_dotenv()
if _dotenv_path:
    ROOT = pathlib.Path(_dotenv_path).parent
else:
    raise FileNotFoundError(
        "Could not find .env file. Please ensure it exists in the project root directory."
        "This is used for secrets (e.g., S3 credentials) and also to set the ROOT directory."
    )

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}:{function}:{line}</cyan> - <level>{message}</level>"
)

logger.remove()
logger.add(sys.stderr, level="DEBUG", format=logger_format)
logger.level("DEBUG", color="<blue><dim>")
