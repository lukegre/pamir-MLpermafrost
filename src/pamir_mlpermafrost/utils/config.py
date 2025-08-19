from pathlib import Path
import hydra
import munch
from omegaconf import OmegaConf
from loguru import logger


class MunchRich(munch.Munch):
    """
    A Munch subclass that provides a rich representation of the object.
    This is useful for debugging and logging purposes.
    """

    def __rich_repr__(self):
        # Yield key-value pairs for a class-style representation
        munch_dict = self.toDict()
        for key, value in munch_dict.items():
            yield key, value


def load_hydra_config(config_dir: str|Path, config_name: str, overrides=[]) -> munch.Munch:
    config_dir = str(Path(config_dir).absolute())

    with hydra.initialize_config_dir(
        config_dir=config_dir, version_base=None, job_name="notebook"
    ):
        params = hydra.compose(config_name=config_name, overrides=overrides)

    if in_jupyter_notebook():
        logger.warning('Will pop `run_dir` since detected in Notebook and cant be resolved')
        params.__dict__['_content'].pop('run_dir', None)
        
    params = OmegaConf.to_container(params, resolve=True)
    params = hydra.utils.instantiate(params)
    params = munch.munchify(params, factory=MunchRich)

    return params


def in_jupyter_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            return False
        else:
            # Other type (maybe in Spyder, etc.)
            return False
    except (NameError, ImportError):
        # Not running in IPython at all
        return False