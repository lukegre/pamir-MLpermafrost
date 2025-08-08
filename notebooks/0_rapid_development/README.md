# Rapid Development

Here I just wanted to see what was going on with the data. How well can we model with a "0-shot" approach? Minimal training etc.

## Notebook descriptions
0. [make_train_data.ipynb](./0_make_train_data.ipynb): creates the basic training data - this is useful for all other steps
1. [data_explore.ipynb](./1_data_explore.ipynb): explores the data, plots it, and tries to understand the relationships (to help with feature selection)
2. [gbm_exploration.ipynb](./2_gbm_exploration.ipynb): try out of the box gradient boosting (sklearn) as a baseline
3. [gp_simple.ipynb](./3_gp_simple.ipynb): try out a simple Gaussian Process with RBF kernel
4. [gp_mixed_covar.ipynb](./4_gp_mixed_covar.ipynb): try out a Gaussian Process with mixed covariance functions
