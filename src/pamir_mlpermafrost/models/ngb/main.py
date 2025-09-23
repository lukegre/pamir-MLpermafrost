

def main(cfg):
    from .viz import plot_train_test_split, plot_feature_importance
    from .tuning import tune_with_cv
    from .inference import scoring

    data = load_training_data(cfg)
    fig_split = plot_train_test_split(data)

    best_model, scores = tune_with_cv(data.train.x, data.train.y, n_folds=5)
    best_model.fit(
        data.train.x, data.train.y, 
        data.valid.x, data.valid.y, 
        early_stopping_rounds=10)
    
    scores = scoring(
        best_model, 
        train=[data.train.x, data.train.y], 
        valid=[data.valid.x, data.valid.y],
        test=[data.test.x, data.test.y])

    fig_feature_importance = plot_feature_importance(best_model)[0]

    figs = fig_split, fig_feature_importance

    return data, best_model, scores, figs
    


def load_training_data(cfg):
    import munch
    
    data = cfg.data.training.load()
    data = data.reset_index().set_index(["y", "x"])
    
    # Preprocess the data
    data_X, data_y = cfg.preprocessing.training(data)
    if data_y.shape[1] == 1:
        data_y = data_y.iloc[:, 0]
    
    # Split the data into training and testing sets
    train_X, test_X, train_y, test_y = cfg.preprocessing.train_test_split(
        data_X, data_y, test_size=0.15,
    )
    
    train_X, valid_X, train_y, valid_y = cfg.preprocessing.train_test_split(
        train_X, train_y, test_size=0.15, random_state=cfg.seed + 6
    )

    return munch.munchify({
        'train': {'x': train_X, 'y': train_y},
        'valid': {'x': valid_X, 'y': valid_y},
        'test':  {'x': test_X, 'y': test_y},
    })

def load_inference_data(cfg):
    predictors = cfg.data.inference.load().compute().set_index(['y', 'x'])
    X = cfg.preprocessing.inference(predictors)
    return X
    