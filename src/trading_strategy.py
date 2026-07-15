import statsmodels.api as sm
import pandas as pd

def compute_spread(
        adj_close_df: pd.DataFrame, 
        cointegrated_pair: tuple
    ) -> tuple[float, float, pd.Series]:
    """
    Compute the spread between two cointegrated tickers using OLS regression.

    Args:
        adj_close_df: DataFrame of adjusted close prices (columns = tickers).
        cointegrated_pair: tuple where index 0 and 1 are the two ticker symbols.

    Returns:
        Tuple of (alpha, beta, spread), where spread is a Series indexed by date.
    """
    t1 = cointegrated_pair[0]
    t2 = cointegrated_pair[1]
    x = adj_close_df[t1]
    y = adj_close_df[t2]
    x_with_const = sm.add_constant(x)

    model = sm.OLS(y, x_with_const).fit()
    alpha = model.params["const"]
    beta = model.params[t1]

    spread = y - (alpha + beta*x)

    return alpha, beta, spread

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

def generate_signals(rolling_z: pd.Series) -> pd.Series:
    """
    Generate trading signals from a rolling z-score using mean-reversion 
    entry/exit rules.

    Position is held until the exit condition is met, so the resulting 
    signal represents an ongoing position, not just instantaneous 
    threshold crossings.

    Rules:
        - z > 2: enter/hold short (-1)
        - z < -2: enter/hold long (1)
        - |z| < 0.5: exit to flat (0)
        - 0.5 <= |z| <= 2: hold current position
        - NaN (insufficient data): hold current position (or 0 if none yet)

    Args:
        rolling_z: Series of rolling z-scores indexed by date.

    Returns:
        Series of signals (-1, 0, 1) indexed by date.
    """
    
    signal_list = []

    for z in rolling_z:
        if pd.isna(z):
            if len(signal_list) == 0:
                signal_list.append(0)
            else: 
                signal_list.append(signal_list[-1])
        elif z > 2:
            signal_list.append(-1)
        elif z < -2:
            signal_list.append(1)
        elif 0.5 <= abs(z) <= 2:
            signal_list.append(signal_list[-1])
        else:
            signal_list.append(0)

    signals = pd.Series(signal_list, index = rolling_z.index)
    return signals