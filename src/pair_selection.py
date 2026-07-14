import itertools
import pandas as pd
from statsmodels.tsa.stattools import coint

def find_correlated_pairs(
        adj_close_df: pd.DataFrame, 
        min_correlation: float
    ) -> list[tuple[str, str, float]]:
    """
    Find all pairs of tickers whose price correlation exceeds a threshold.

    Args:
        adj_close_df: DataFrame of adjusted close prices (columns = tickers).
        min_correlation: minimum correlation required to keep a pair.

    Returns:
        List of tuples (ticker1, ticker2, correlation).
    """
    correlated_pairs = []
    for t1, t2 in itertools.combinations(adj_close_df.columns, 2):
        correlation = adj_close_df[t1].corr(adj_close_df[t2])
        if correlation > min_correlation:
            correlated_pairs.append((t1, t2, correlation))
    return correlated_pairs

def find_cointegrated_pairs(
        adj_close_df: pd.DataFrame, 
        correlated_pairs: list[tuple[str, str, float]], 
        max_pvalue: float
    ) -> list[tuple[str, str, float, float]]:
    """
    Test a list of correlated pairs for cointegration and keep the significant ones.

    Args:
        adj_close_df: DataFrame of adjusted close prices (columns = tickers).
        correlated_pairs: list of (ticker1, ticker2, correlation) tuples.
        max_pvalue: maximum p-value for a pair to be considered cointegrated.

    Returns:
        List of tuples (ticker1, ticker2, correlation, pvalue).
    """
    cointegrated_pairs = []
    for t1, t2, correlation in correlated_pairs:
        pvalue = coint(adj_close_df[t1], adj_close_df[t2])[1]
        if pvalue < max_pvalue:
            cointegrated_pairs.append((t1, t2, correlation, pvalue))
    return cointegrated_pairs