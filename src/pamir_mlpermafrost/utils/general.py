import sys
from loguru import logger


def DummyContextManager(*args, **kwargs):
    """A dummy context manager that does nothing."""

    class DummyContext:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    return DummyContext()



def set_logger_level(level):
    
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}:{function}:{line}</cyan> - <level>{message}</level>"
    )
    
    logger.remove()
    logger.add(sys.stderr, level=level, format=logger_format)
    logger.level("DEBUG", color="<blue><dim>")
