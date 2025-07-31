import gpytorch
import numpy as np
import torch
from loguru import logger

from ...preprocessing.scalers import StandardScaler_toTensor


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Better kernel for mixed continuous/discrete data
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                # ARD for different length scales per dimension
                ard_num_dims=train_x.shape[1]
            ),
        )

        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPMixedMean(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, linear_mean_idx=[]):
        super().__init__(train_x, train_y, likelihood)

        n_cols = train_x.shape[-1]
        self.linear_idx = linear_mean_idx
        self.constant_idx = [i for i in range(n_cols) if i not in linear_mean_idx]

        self.linear_mean = gpytorch.means.LinearMean(input_size=len(self.linear_idx))
        self.constant_mean = gpytorch.means.ConstantMean()

        rbf_kernel = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(rbf_kernel)

    def forward(self, x):
        x_linear = x[..., self.linear_idx]
        x_constant = x[..., self.constant_idx]

        mean_x = self.linear_mean(x_linear) + self.constant_mean(x_constant)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPMixedMeanCatIndex(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, linear_mean_idx=[], cat_idx=[]):
        print(train_x)
        super().__init__(train_x, train_y, likelihood)

        n_cols = train_x.shape[-1]
        n_cols_linear = len(linear_mean_idx)
        n_cols_numeric = n_cols - len(cat_idx)

        categories = torch.unique(train_x[:, cat_idx])
        n_categories = categories.numel() + 1  # +1 accounts for the zero category
        all_idx = set(range(n_cols))

        logger.info(f"{categories=}")

        self.idx_linear = linear_mean_idx
        self.idx_constant = list(all_idx - set(linear_mean_idx + cat_idx))
        self.idx_categorical = cat_idx

        # constructing the mean modules
        self.linear_mean = gpytorch.means.LinearMean(input_size=n_cols_linear)
        self.constant_mean = gpytorch.means.ConstantMean()

        # constructing the covariance modules
        rbf_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=n_cols_numeric,
                active_dims=self.idx_linear + self.idx_constant,
            )
        )
        # we use an IndexKernel to handle categorical variables
        idx_kernel = gpytorch.kernels.IndexKernel(
            num_tasks=n_categories, active_dims=self.idx_categorical
        )

        self.covar_module = rbf_kernel * idx_kernel

    def forward(self, x):
        x_linear = x[..., self.idx_linear]
        x_constant = x[..., self.idx_constant]

        mean_x = self.linear_mean(x_linear) + self.constant_mean(x_constant)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
