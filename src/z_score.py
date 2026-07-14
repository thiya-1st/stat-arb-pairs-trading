import pandas as pd

def compute_rolling_z(spread: pd.Series, window: int) -> pd.Series:
    """
    Compute the rolling z-score of a spread series.

    Args:
        spread: Series of spread values indexed by date.
        window: number of periods to use for the rolling mean/std.

    Returns:
        Series of rolling z-scores, same index as spread. First 
        (window - 1) values will be NaN due to insufficient data.
    """
    rolling_object = spread.rolling(window)
    rolling_mean = rolling_object.mean()
    rolling_std = rolling_object.std()
    rolling_z = (spread - rolling_mean) / rolling_std
    return rolling_z