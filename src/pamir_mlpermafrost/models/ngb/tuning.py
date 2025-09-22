import xarray as xr

N_FOLDS_DEFAULT = 4
PARAMS_DEFAULT = {
        'max_depth': [6, 12], 
        'min_samples_leaf': [12, 24], 
        'ccp_alpha': [0.005, 0.01, 0.1]}


def tune_with_cv(train_X, train_y, n_folds=N_FOLDS_DEFAULT):

    # data for cross validation
    cv_datasets = make_cv_datasets(train_X, train_y, n_folds=n_folds)

    # models with parameters to select best model
    param_list = make_param_list()
    models = make_models(param_list)

    # search the gridspace
    models = hyper_param_search(models, cv_datasets)

    # used for indexing 
    index = make_cv_index(param_list, n_folds)
    ds = xr.Dataset()
    ds['scores'] = get_model_attr(models, param_list, 'score', n_folds)
    ds['niters'] = get_model_attr(models, param_list, 'best_val_loss_itr', n_folds)

    best_params = get_best_params(ds.scores)
    best_model = make_models([best_params])[0]
    
    best_model.feature_names_in_ = train_X.columns.tolist()

    return best_model, ds

    
def make_models(param_list):
    import ngboost
    
    models = []
    for params in param_list:
        models += ngboost.NGBRegressor(
            Base=ngboost.learners.DecisionTreeRegressor(
                criterion='friedman_mse',
                splitter='random',
                ccp_alpha=params['ccp_alpha'],
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf']), 
            n_estimators=5_000, 
            verbose_eval=False, 
            early_stopping_rounds=5,
            learning_rate=params.get('learning_rate', 0.01)),
        
    return models


def make_cv_index(param_list, n_folds=N_FOLDS_DEFAULT):
    from copy import deepcopy
    import pandas as pd
    
    cv_list = []
    for params in param_list:
        for i in range(n_folds):
            cv_list += (deepcopy(params) | dict(cv_member=i)),
    
    index = pd.MultiIndex.from_tuples(pd.DataFrame(cv_list).values.tolist(), names=cv_list[0].keys())

    return index


def get_model_attr_list(model_list, attr):
    import numpy as np
    import pandas as pd
    values = np.array([getattr(m, attr) for m in model_list])
    df = pd.DataFrame(values, columns=[attr])[attr]
    return df


def get_model_losses(model_list, index, subset:['train', 'val']='val'):
    import pandas as pd
    
    losses = [results.evals_result[subset]['LOGSCORE'] for results in model_list]
    losses = (
        pd.DataFrame(losses)
        .T
        .set_axis(index, axis=1)
        .unstack()
        .to_xarray()
        .squeeze())
    unnamed = losses.dims[-1]
    losses = losses.rename({unnamed:'iter'})
    return losses
    

def make_param_list(params=PARAMS_DEFAULT):
    
    from itertools import product
    
    param_list = product(*params.values())
    param_list = [dict(zip(params.keys(), values)) for values in param_list]

    return param_list


def make_cv_datasets(train_X, train_y, n_folds=5):
    from sklearn import model_selection
    
    splitter = model_selection.KFold(n_folds)
    return [
        dict(
            X=train_X.iloc[t], 
            Y=train_y.iloc[t],
            X_val=train_X.iloc[v],
            Y_val=train_y.iloc[v],
        )
        for t, v in splitter.split(train_X)
    ]


def fit_model_and_score(model, **kwargs):
    model = model.fit(**kwargs, early_stopping_rounds=10)
    model.score = model.score(kwargs['X_val'], kwargs['Y_val'])
    return model


def hyper_param_search(models, cv_datasets, n_jobs=24):
    import joblib
    
    def fit_score(model, **kwargs):
        model = model.fit(**kwargs, early_stopping_rounds=10)
        model.score = model.score(kwargs['X_val'], kwargs['Y_val'])
        return model

    tasks = []
    for model in models:
        for i, cv_subset in enumerate(cv_datasets):
            tasks += joblib.delayed(fit_score)(model, **cv_subset),
    
    workers = joblib.Parallel(n_jobs=n_jobs, verbose=True)
    output = workers(tasks)
    return output
    

def try_int(v):
    if not v % 1:
        return int(v)
    else:
        return float(v)


def get_model_attr(model_list, param_list, attr, n_folds=N_FOLDS_DEFAULT):

    index = make_cv_index(param_list, n_folds=n_folds)
    values = get_model_attr_list(model_list, attr).set_axis(index, axis=0).to_xarray().rename(attr)

    return values
    

def get_best_params(scores):

    vals = scores.median('cv_member')
    cols = vals.dims
    best_vals = vals.to_series().idxmin()
    best_params = {k: try_int(v) for k, v in zip(cols, best_vals)}

    return best_params