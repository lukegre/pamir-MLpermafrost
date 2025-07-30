import sys

from loguru import logger

from . import data, models, viz

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}:{function}:{line}</cyan> - <level>{message}</level>"
)

logger.remove()
logger.add(sys.stderr, level="DEBUG", format=logger_format)
logger.level("DEBUG", color="<blue><dim>")
