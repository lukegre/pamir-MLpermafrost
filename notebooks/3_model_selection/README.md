# Model Selection

In this section, I try to implement what I learnt in [1_learning_about_GPs](../1_learning_about_GPs/README.md) to the data I have.

- [x] Address issue of permafrost temperature having positive values (or often near 0). *Solution: focus on temperature at different depths - issue: may not represent ALT accurately since mean temperature < 0 can still have positive values at some points.*
- [x] Select an appropriate target transformation if applicable (-log, etc.). *Solution: predict a contiuous variable (ground temperature)*
- [ ] Kernel selection using the following criteria
  - [ ] Assess product vs additive kernels for the categorical features (which uses IndexKernel)
  - [x] Explore adding PolynomialKernel for features where there is likely a linear-like relationship (e.g., temperature, elevation). Solution: only elevation as linear.
  - [ ] The RBF kernel for the rest


## Notebooks

### [0_choosing_target_variable.ipynb](./0_choosing_target_variable.ipynb)

The first thing we have to do is re-assess the target variable (permafrost temperature), since it seems that there are many points that are shown as being permafrost, but are actually not (mean ground temperatures at all levels are > 0).

<div style="background-color:rgb(245, 210, 152); padding: 10px; border-radius: 10px;">
<b>UPDATE:</b> Looking at the data again, there doesn't seem much sense in predicting permafrost temperature, since it is not a continuous variable, which adds complexity. Thus predicting ground temperature at 2, 5, 10 m may be a better option for now.
</div>

<div style='padding:6px 15px; margin:10px 0px; background: #a2e8b5; border-radius: 10px;'>
<i>FUTURE IDEA:</i> Predicting seasonal ground temperatures instead of annual mean. Choose coldest period and warmest period.</div>


### [1_modelling_ground_temp.ipynb](./1_modelling_ground_temp.ipynb)
Implement a separate GPs for each of the ground temperature levels (2, 5, 10 m).
Be sure to scale the input temperature to the `precip_scaling` parameters set for each of the experiments.

#### GP setup
- [ ] `ConstantMean`: for all features
- [ ] Covariance functions:
  - [ ] `Linear`: elevation - there's nothing to suggest that any other features have a linear relationship with ground temperatures
  - [ ] `IndexKernel`: categorical features - test if product (independent models) or additive (shared models) kernel works better
  - [ ] `LinearAutoEncoder`: 2m temperature / precip: use a linear layer `nn.Linear(n_temp_features, n_temp_projected_features)` to project the features to a lower dimension. Then pass these to an RBF kernel.
  - [ ] `RBF_ARD`: for the rest of the features, use an RBF kernel with automatic relevance determination (ARD) to learn the lengthscales for each feature independently.
- [ ] `GaussianLikelihood`: Since we are predicting ground temperatures (continuous), we can use a noise prior of $\mathcal{N}(0, 1)$, but this is something I should look into a lot more - what are the uncertainties we understand.
- [ ] `ExactGP`: This is if we have less than 10k training points, otherwise we can use


<div style='padding:6px 15px; margin:10px 0px; background: #a2e8b5; border-radius: 10px;'>
<i>FUTURE IDEA:</i> At the moment, we're predicting a single temperature level, but it could also be possible to predict the entire temperature profile (e.g., at <code>0:0.25:5 + 5:1:20</code> resolution) simultaneously using a multi-output model. We could then have different models for winter / summer periods to help get to ALT predictions. </div>
