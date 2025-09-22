import pandas as pd


METRICS_DEFAULT = [
    'r2_score', 
    'mean_absolute_error', 
    'root_mean_squared_error']


def predict(model, X, n_jobs=24):
    import joblib
    
    n_batches = n_jobs * 10
    step = X.index.size // n_batches
    steps = range(0, X.index.size, step)
    
    func = joblib.delayed(_predict_single_thread)
    tasks = [func(model, X.iloc[i: i + step]) for i in steps]
    worker = joblib.Parallel(n_jobs=-1, verbose=True)
    
    out = worker(tasks)
    yhat = pd.concat(out, axis=0)
    yhat = yhat.to_xarray()
    return yhat

def _predict_single_thread(model, x_subset):
    yhat, ystd = model.pred_dist(x_subset).params.values()
    yhat = pd.Series(yhat, index=x_subset.index).rename('yhat')
    ystd = pd.Series(ystd, index=x_subset.index).rename('ystd')
    yhat = pd.concat([yhat, ystd], axis=1)
    return yhat


def scoring(model, metric_funcs=METRICS_DEFAULT, **xy_pairs):
    from sklearn import metrics
    
    results = pd.DataFrame()
    
    for func_name in metric_funcs:
        func = getattr(metrics, func_name)
        
        for key in xy_pairs:
            x, y = xy_pairs[key]
            yhat = model.predict(x)
            results.loc[func_name, key] = func(y, yhat)

    return results