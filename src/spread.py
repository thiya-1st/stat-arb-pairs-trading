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