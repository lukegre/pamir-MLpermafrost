import hydra
import munch
from omegaconf import OmegaConf


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


def load_hydra_config(config_dir: str, config_name: str, overrides=[]) -> munch.Munch:
    with hydra.initialize_config_dir(
        config_dir=config_dir, version_base=None, job_name="notebook"
    ):
        params = hydra.compose(config_name=config_name, overrides=overrides)

    params = OmegaConf.to_container(params, resolve=True)
    params = hydra.utils.instantiate(params)
    params = munch.munchify(params, factory=MunchRich)

    return params
