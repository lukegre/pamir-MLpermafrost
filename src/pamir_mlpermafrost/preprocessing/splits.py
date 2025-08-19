import numpy as np
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


def corner_split(
    data_X: pd.DataFrame,
    data_y: pd.Series,
    loc: str = 'random',
    random_state: int = 42,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing set by removing 
    a corner at a random location in the domain. 

    Parameters
    ----------
    data_X : pd.DataFrame
        DataFrame containing the features.
    data_y : pd.Series
        Series containing the target variable.
    loc : str, optional
        Location of the corner to remove, by default 'random'.
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    **kwargs : dict, optional
        Additional keyword arguments (not used in this function).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        A tuple containing
        - train_x (pd.DataFrame)
        - test_x (pd.DataFrame)
        - train_y (pd.Series)
        - test_y (pd.Series)
    """
    # Implementation of corner split logic goes here
    # This is a placeholder implementation; actual logic will depend on requirements

    coords = data_X.index.to_frame()
    y_mid = _get_coord_mid(coords['y'])
    x_mid = _get_coord_mid(coords['x'])

    left = coords['x'] < x_mid
    right = coords['x'] >= x_mid
    upper = coords['y'] > y_mid
    lower = coords['y'] <= y_mid

    if loc == "random":
        rand_state = np.random.RandomState(random_state)
        vert = rand_state.choice(['upper', 'lower'], size=1)
        horz = rand_state.choice(['left', 'right'], size=1)
        loc = f"{vert[0]} {horz[0]}"
        
    test_corners = {
        'upper left': upper & left,
        'upper right': upper & right,
        'lower left': lower & left,
        'lower right': lower & right,
    }

    mask_train = ~test_corners[loc]
    train_X, test_X, train_y, test_y = _masking(data_X, data_y, mask_train)
    return train_X, test_X, train_y, test_y


def horizontal_band(
    data_X: pd.DataFrame,
    data_y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing set by removing 
    a horizontal band at a random location in the domain. 

    Parameters
    ----------
    data_X : pd.DataFrame
        DataFrame containing the features.
    data_y : pd.Series
        Series containing the target variable.
    test_size : float, optional
        Fraction of the data to use for the test set, by default 0.2.
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    **kwargs : dict, optional
        Additional keyword arguments (not used in this function).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        A tuple containing
        - train_x (pd.DataFrame)
        - test_x (pd.DataFrame)
        - train_y (pd.Series)
        - test_y (pd.Series)
    """

    y = data_X.index.to_frame()['y']
    mask_train = ~_get_band(y, band_frac=test_size, seed=random_state)
    train_X, test_X, train_y, test_y = _masking(data_X, data_y, mask_train)

    return train_X, test_X, train_y, test_y


def vertical_band(
    data_X: pd.DataFrame,
    data_y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing set by removing 
    a vertical band at a random location in the domain. 

    Parameters
    ----------
    data_X : pd.DataFrame
        DataFrame containing the features.
    data_y : pd.Series
        Series containing the target variable.
    test_size : float, optional
        Fraction of the data to use for the test set, by default 0.2.
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    **kwargs : dict, optional
        Additional keyword arguments (not used in this function).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        A tuple containing
        - train_x (pd.DataFrame)
        - test_x (pd.DataFrame)
        - train_y (pd.Series)
        - test_y (pd.Series)
    """

    x = data_X.index.to_frame()['x']
    mask_train = ~_get_band(x, band_frac=test_size, seed=random_state)
    train_X, test_X, train_y, test_y = _masking(data_X, data_y, mask_train)

    return train_X, test_X, train_y, test_y


def vert_horz_cross(
    data_X: pd.DataFrame,
    data_y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing set by removing 
    a vertical and horizontal band at a random location in the domain. 

    Parameters
    ----------
    data_X : pd.DataFrame
        DataFrame containing the features.
    data_y : pd.Series
        Series containing the target variable.
    test_size : float, optional
        Fraction of the data to use for the test set, by default 0.2.
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    **kwargs : dict, optional
        Additional keyword arguments (not used in this function).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        A tuple containing
        - train_x (pd.DataFrame)
        - test_x (pd.DataFrame)
        - train_y (pd.Series)
        - test_y (pd.Series)
    """
    coords = data_X.index.to_frame()
    mask_y = _get_band(coords['y'], band_frac=test_size/2, seed=random_state)
    mask_x = _get_band(coords['x'], band_frac=test_size/2, seed=random_state)
    train_mask = ~(mask_y | mask_x)
    train_X, test_X, train_y, test_y = _masking(data_X, data_y, train_mask)

    return train_X, test_X, train_y, test_y


def _get_coord_mid(coords: pd.Series) -> float:
    """
    Returns the midpoint of the coordinates.

    Parameters
    ----------
    coords : pd.Series
        Series containing the coordinates (e.g., 'y' values).

    Returns
    -------
    float
        The midpoint of the coordinates.
    """
    c0, c1 = coords.min(), coords.max()
    return (c0 + c1) / 2.0


def _get_band(coords: pd.Series, band_frac: float = 0.2, seed:int=42) -> pd.Series:
    """
    Returns a boolean mask for a horizontal band in the data.

    Parameters
    ----------
    coords : pd.Series
        Series containing the coordinates (e.g., 'y' values).
    band_frac : float, optional
        Fraction of the data to use for the band, by default 0.2.

    Returns
    -------
    pd.Series
        Boolean Series indicating which rows are in the band.
    """
    c0, c1 = coords.min(), coords.max()
    domain_size = c1 - c0
    band_size = domain_size * band_frac

    # Select a random location for the band
    random_state = np.random.RandomState(seed)
    band = random_state.uniform(c0, c1 - band_size)

    return (coords >= band) & (coords <= band + band_size)


def _masking(
    data_X: pd.DataFrame,
    data_y: pd.Series,
    train_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Masks the data based on a boolean mask.

    Parameters
    ----------
    data_X : pd.DataFrame
        DataFrame containing the features.
    data_y : pd.Series
        Series containing the target variable.
    mask : pd.Series
        Boolean Series indicating which rows to keep.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        A tuple containing the masked features and target variable.
    """
    train_X = data_X[train_mask]
    test_X = data_X[~train_mask]
    train_y = data_y[train_mask]
    test_y = data_y[~train_mask]

    return train_X, test_X, train_y, test_y