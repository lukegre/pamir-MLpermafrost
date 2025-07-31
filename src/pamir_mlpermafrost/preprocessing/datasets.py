import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset

from pamir_mlpermafrost.preprocessing.scalers import StandardScaler_toTensor


class DatasetXarraySpatial(Dataset):
    def __init__(
        self, xarray_dataset, chunk_processor: callable, scaler: StandardScaler_toTensor
    ):
        assert len(xarray_dataset.dims) == 2, "Dataset must be 2D (time, features)"
        assert "x" in xarray_dataset.dims and "y" in xarray_dataset.dims, (
            "Dataset must have 'x' and 'y' dimensions"
        )

        self.xarray_dataset = xarray_dataset
        self.chunk_processor = chunk_processor
        self.scaler = scaler
        if scaler is not None:
            assert isinstance(scaler, StandardScaler_toTensor), (
                "Scaler must be an instance of StandardScaler_toTensor"
            )
            self.scaler = scaler
        self.chunk_mapper = None
        self._make_chunk_index()

    def __len__(self):
        return len(self.chunk_mapper)

    def __getitem__(self, idx):
        ds = self._get_chunk(idx)
        df = ds.to_dataframe()
        X = self.chunk_processor(df)
        X_tensor_scaled = self.scaler.transform(X)
        return X_tensor_scaled

    def _get_chunk(self, idx):
        """
        If the dataset is chunked, this method returns the
        chunk at the specified index for the entire dataset.
        """
        if self.chunk_mapper is None:
            self._make_chunk_index()

        if idx not in self.chunk_mapper:
            raise IndexError(
                f"Chunk index {idx} out of range. Available chunks: {list(self.chunk_mapper.keys())}"
            )

        chunk_selector = self.chunk_mapper[idx]
        ds_chunk = self.xarray_dataset.isel(**chunk_selector)
        return ds_chunk

    def _make_chunk_index(self):
        from itertools import product

        ds = self.xarray_dataset
        chunk_mapper = {}

        xchunks = list(ds.chunks["x"])
        ychunks = list(ds.chunks["y"])

        xchunks_cum0 = np.cumsum([0] + xchunks[:-1]).tolist()
        xchunks_cum1 = np.cumsum(xchunks).tolist()
        ychunks_cum0 = np.cumsum([0] + ychunks[:-1]).tolist()
        ychunks_cum1 = np.cumsum(ychunks).tolist()

        xslices = [slice(i0, i1) for i0, i1 in zip(xchunks_cum0, xchunks_cum1)]
        yslices = [slice(i0, i1) for i0, i1 in zip(ychunks_cum0, ychunks_cum1)]

        xy_slices = list(product(xslices, yslices))

        for i, (xs, ys) in enumerate(xy_slices):
            chunk_mapper[i] = dict(x=xs, y=ys)

        self.chunk_mapper = chunk_mapper

    def reconstruct_output(self, output: list) -> xr.DataArray:
        """
        Reconstruct the output from the dataset.
        This method is a placeholder and should be implemented
        based on the specific requirements of the output format.
        """
        # For now, just return the output as is
        assert len(output) == len(self.chunk_mapper), (
            "Output length must match the number of chunks"
        )

        key0 = list(self.xarray_dataset.data_vars)[0]
        dummy = self.xarray_dataset[key0].compute().rename("reconstructed") * np.nan

        for i, chunk in enumerate(output):
            chunk_iselector = self.chunk_mapper[i]
            # assign chunk to the corresponding slice in the dummy DataArray
            y_size = chunk_iselector["y"].stop - chunk_iselector["y"].start
            x_size = chunk_iselector["x"].stop - chunk_iselector["x"].start

            x_sel = dummy.x.isel(x=chunk_iselector["x"])
            y_sel = dummy.y.isel(y=chunk_iselector["y"])
            dummy.loc[y_sel, x_sel] = chunk.reshape(y_size, x_size)

        return dummy
