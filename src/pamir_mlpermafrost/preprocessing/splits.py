import pandas as pd


def train_test_split_stratified(
    data_X: pd.DataFrame,
    data_y: pd.Series,
    stratified_columns="surface_index",
    test_size=0.2,
    random_state=42,
    shuffle=True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets with stratification.
    A wrapper around `sklearn.model_selection.train_test_split` that
    allows for stratified sampling based on a specified column.

    Parameters
    ----------
    data_X : pd.DataFrame
        DataFrame containing the features.
    data_y : pd.Series
        Series containing the target variable.
    stratified_columns : str, optional
        Column name to use for stratification, by default "surface_index".

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        A tuple containing
        - train_x (pd.DataFrame)
        - test_x (pd.DataFrame)
        - train_y (pd.Series)
        - test_y (pd.Series)
    """
    from sklearn.model_selection import train_test_split

    props = dict(
        test_size=test_size,
        random_state=random_state,
        stratify=data_X[stratified_columns],
        shuffle=shuffle,
    )
    # train test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, **props)

    return X_train, X_test, y_train, y_test
