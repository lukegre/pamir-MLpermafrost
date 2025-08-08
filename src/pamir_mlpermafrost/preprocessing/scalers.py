import json
import pathlib

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class StandardScaler_toTensor(StandardScaler):
    """
    transform (pd.DataFrame | pd.Series | np.ndarray --> torch.Tensor)
    inverse_transform (torch.Tensor --> pd.DataFrame | pd.Series | np.ndarray)
    """

    def __init__(
        self,
        skip_columns: list[int] = [],
        save_path: str | pathlib.Path = None,
        **kwargs,
    ):
        """
        Initialize the StandardScaler_toTensor.
        transform (pd.DataFrame | pd.Series | np.ndarray --> torch.Tensor)
        inverse_transform (torch.Tensor --> pd.DataFrame | pd.Series | np.ndarray)

        Parameters
        ----------
        skip_columns : list, optional
            List of columns to skip during scaling (default is an empty list).
            Keeps these columns unchanged during transformation.
        **kwargs : additional keyword arguments
            Additional parameters to pass to the StandardScaler constructor.
            This allows for customization of the scaler, such as `with_mean` and `with_std` parameters.
        """
        super(StandardScaler_toTensor, self).__init__(**kwargs)
        self.skip_columns = skip_columns
        self.save_path = save_path

    def fit(self, X, y=None):
        super(StandardScaler_toTensor, self).fit(X, y)

        if isinstance(X, pd.DataFrame):
            # Store feature names for inverse transformation
            self.feature_names_in_ = X.columns.tolist()
        elif isinstance(X, pd.Series):
            self.feature_names_in_ = X.name
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = None

        # return the mean and scale for the skipped columns
        if self.skip_columns:
            self.mean_[self.skip_columns] = 0
            self.scale_[self.skip_columns] = 1

        return self

    def transform(self, X):
        X_scaled = super(StandardScaler_toTensor, self).transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32, device="cpu")

    def inverse_transform(self, X_scaled_tensor):
        X_scaled = X_scaled_tensor.cpu().numpy()
        X_original = super(StandardScaler_toTensor, self).inverse_transform(X_scaled)

        if isinstance(self.feature_names_in_, list):
            X_original = pd.DataFrame(data=X_original, columns=self.feature_names_in_)
        elif isinstance(self.feature_names_in_, str):
            X_original = pd.Series(data=X_original, name=self.feature_names_in_)
        return X_original

    def save_params(self, fname: str | pathlib.Path = None):
        """
        Save the scaler parameters to a file.

        Parameters
        ----------
        fname : str
            The filename to save the scaler to.
        """

        if fname is None:
            if self.save_path is None:
                raise ValueError("No filename provided and no save_path set.")
            fname = self.save_path

        fname = str(fname)

        if not fname.endswith(".json"):
            raise ValueError("Filename must end with .json")

        params = {
            "mean": [float(x) for x in self.mean_],
            "scale": [float(x) for x in self.scale_],
            "skip_columns": list(self.skip_columns),
            "names": list(self.feature_names_in_),
        }

        with open(fname, "w") as f:
            json.dump(
                params,
                f,
            )

    @classmethod
    def from_params_file(cls, fname: str):
        """
        Load the scaler from a file.

        Parameters
        ----------
        fname : str
            The filename to load the scaler from.

        Returns
        -------
        StandardScaler_toTensor
            The loaded scaler.
        """

        with open(fname, "r") as f:
            params = json.load(f)

        scaler = cls(skip_columns=params["skip_columns"])
        scaler.mean_ = np.array(params["mean"])
        scaler.scale_ = np.array(params["scale"])
        scaler.feature_names_in_ = params["names"]

        return scaler


class NegativeLogScaler_toTensor(StandardScaler_toTensor):
    """
    Same as StandardScaler_toTensor, but applies a log transformation
    before scaling the data. Designed for data that is strictly negative.
    """

    def fit(self, X, y=None):
        super(NegativeLogScaler_toTensor, self).fit(X, y)
        return self

    def transform(self, X, y=None):
        X_log = -np.log10(-X)
        X_scaled = super(NegativeLogScaler_toTensor, self).transform(X_log)
        return torch.tensor(X_scaled, dtype=torch.float32, device="cpu")

    def inverse_transform(self, X_scaled_tensor):
        X_scaled_log = X_scaled_tensor.cpu().numpy()
        X_log = super(NegativeLogScaler_toTensor, self).inverse_transform(X_scaled_log)
        X_original = -(10 ** (-X_log))

        if isinstance(self.feature_names_in_, list):
            X_original = pd.DataFrame(data=X_original, columns=self.feature_names_in_)
        elif isinstance(self.feature_names_in_, str):
            X_original = pd.Series(data=X_original, name=self.feature_names_in_)
        return X_original
