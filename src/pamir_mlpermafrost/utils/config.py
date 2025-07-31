import importlib
import pathlib
from functools import partial

import munch


def load_params(fname: str, prefix: str = "./experiments/") -> munch.Munch:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    fname : str
        The filename of the YAML file to load.

    Returns
    -------
    dict
        The loaded configuration as a dictionary.
    """
    import yaml

    with open(fname, "r") as f:
        params = yaml.safe_load(f)

    params["hash"] = create_hash(params)
    params["name"] = process_placeholders(params, template=params["name"])

    params = munch.munchify(params)
    params["directory"] = str(pathlib.Path(prefix) / params.name)

    save_params_output(params, prefix)

    params["model"] = instantiate_config_classes(params["model"])
    params["scalers"]["features"] = instantiate_config_classes(
        params["scalers"]["features"]
    )
    params["scalers"]["target"] = instantiate_config_classes(
        params["scalers"]["target"]
    )

    params["data"]["training"] = instantiate_config_functions(
        params["data"]["training"]
    )
    params["data"]["inference"] = instantiate_config_functions(
        params["data"]["inference"]
    )

    params["directory"] = pathlib.Path(params.directory)

    return params


def create_hash(params: munch.Munch) -> str:
    """
    Create a hash from the configuration.

    Parameters
    ----------
    config : munch.Munch
        The configuration loaded from the YAML file.

    Returns
    -------
    str
        A hash string based on the configuration.
    """
    import hashlib

    params_str = str(params)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


def process_placeholders(params: munch.Munch, template: str) -> str:
    """
    Replace placeholders in a string with corresponding values from the configuration.

    Parameters
    ----------
    config : munch.Munch
        The configuration loaded from the YAML file.
    template : str
        The string template with placeholders (e.g., '{model.name}').

    Returns
    -------
    str
        The processed string with placeholders replaced by their values.
    """
    import re

    def replace_placeholder(match):
        # Extract the key path (e.g., 'model.name')
        key_path = match.group(1)
        # Traverse the params to get the value
        value = params
        for key in key_path.split("."):
            value = value[key]
        return str(value).split(".")[-1]

    # Use regex to find placeholders in the format {key.path}
    return re.sub(r"\{([\w\.]+)\}", replace_placeholder, template)


def instantiate_config_classes(config: dict):
    """
    Recursively instantiate classes from a configuration dictionary.

    Parameters
    ----------
    config : dict
        A dictionary containing a 'class' key and other kwargs for instantiation.

    Returns
    -------
    object
        An instance of the specified class, with nested classes instantiated as well.
    """

    if isinstance(config, dict) and "class" in config:
        # Import the class
        class_obj = import_from_string(config["class"])

        # Process nested items (e.g., subkeys with 'class')
        kwargs = {
            key: instantiate_config_classes(value) if isinstance(value, dict) else value
            for key, value in config.items()
            if key != "class"  # Exclude the 'class' key from kwargs
        }

        # Instantiate the class with the processed kwargs
        return partial(class_obj, **kwargs)

    # If it's not a dictionary with a 'class' key, return as is
    return config


def process_classes(params: munch.Munch) -> None:
    """
    Process class paths in the configuration.

    This function replaces class paths with their actual class objects in the
    configuration.

    Parameters
    ----------
    config : munch.Munch
        The configuration loaded from the YAML file.
    """
    for key, value in params.items():
        if isinstance(value, munch.Munch):
            process_classes(value)
        elif key == "class":
            params[key] = import_from_string(value)

    return params


def instantiate_config_functions(config: dict):
    """
    Recursively instantiate functions from a configuration dictionary.

    Parameters
    ----------
    config : dict
        A dictionary containing a 'func' key and other kwargs for instantiation.

    Returns
    -------
    object
        An instance of the specified function, with nested functions instantiated as well.
    """
    if isinstance(config, dict) and "func" in config:
        # Import the function
        func_obj = import_from_string(config["func"])

        # Process nested items (e.g., subkeys with 'func')
        kwargs = {
            key: instantiate_config_functions(value)
            if isinstance(value, dict)
            else value
            for key, value in config.items()
            if key != "func"  # Exclude the 'func' key from kwargs
        }

        # Return the function with the processed kwargs
        return partial(func_obj, **kwargs)

    # If it's not a dictionary with a 'func' key, return as is
    return config


def process_functions(params: munch.Munch) -> None:
    """
    Process function paths in the configuration.

    This function replaces function paths with their actual function objects in the
    configuration.

    Parameters
    ----------
    config : munch.Munch
        The configuration loaded from the YAML file.
    """
    for key, value in params.items():
        if isinstance(value, munch.Munch):
            process_functions(value)
        elif key == "func":
            params[key] = import_from_string(value)

    return params


def save_params_output(params: munch.Munch, prefix: str) -> None:
    """
    Create a configuration output file in yaml format.

    Parameters
    ----------
    params : munch.Munch
        The configuration loaded from the YAML file.
    prefix : str
        The file path will be {prefix}/{params.name}/params.yaml
    """
    import yaml

    output_path = pathlib.Path(prefix) / params.name
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "params.yaml"

    with open(output_file, "w") as f:
        config_dict = munch.Munch.toDict(params)
        yaml.dump(config_dict, f, sort_keys=False, line_break="\n\n")


def import_from_string(class_path: str):
    """
    Dynamically import a class from a string path.

    Parameters
    ----------
    class_path : str
        The full path to the class (e.g., 'module.submodule.ClassName').

    Returns
    -------
    type
        The imported class.
    """
    module_path, class_name = class_path.rsplit(".", 1)  # Split into module and class
    module = importlib.import_module(module_path)  # Import the module
    return getattr(module, class_name)  # Get the class from the module
