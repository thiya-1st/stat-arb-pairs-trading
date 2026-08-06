import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_position_size(
        capital: float, 
        beta: float, 
        price_y: float, 
        price_x: float
    ) -> tuple[float, float]:
    """
    Calculate beta-neutral share counts for a pairs trade.

    Args:
        capital: dollar amount allocated to the trade.
        beta: hedge ratio from the pair's regression (Y on X).
        price_y: current price of ticker Y.
        price_x: current price of ticker X.

    Returns:
        Tuple of (shares_x, shares_y).
    """
    dollars_in_y = capital / (1 + beta)
    dollars_in_x = capital - dollars_in_y
    shares_y = dollars_in_y / price_y
    shares_x = dollars_in_x / price_x
    return shares_x, shares_y

def calculate_daily_pnl(
        adj_close_df: pd.DataFrame, 
        capital: float, 
        trade_cost_rate: float, 
        cointegrated_pair: tuple, 
        signals: pd.Series, 
        beta: float
    ) -> pd.Series:
    """
    Calculate daily PnL for a pairs trading strategy on one cointegrated pair.

    Args:
        adj_close_df: DataFrame of adjusted close prices (columns = tickers).
        capital: dollar amount allocated per trade.
        trade_cost_rate: transaction cost rate applied on entry/exit (e.g. 0.001 = 0.1%).
        cointegrated_pair: tuple where index 0 and 1 are the two ticker symbols.
        signals: Series of trading signals (-1, 0, 1) indexed by date.
        beta: hedge ratio from the pair's regression (Y on X).

    Returns:
        Series of daily PnL values indexed by date.
    """
    x = cointegrated_pair[0]
    y = cointegrated_pair[1]
    pnl_list = []
    for i in range(len(adj_close_df)):
        price_x = adj_close_df[x].iloc[i]
        price_y = adj_close_df[y].iloc[i]
        signal = signals.iloc[i]
        if i == 0: # First day in the list
            if signal != 0: # entering trade
                # entry shares
                shares_x, shares_y = calculate_position_size(capital, beta, price_y, price_x)
                trade_cost = capital * trade_cost_rate # entry transaction cost
                pnl_list.append(-trade_cost)
            else: # no trades
                pnl_list.append(0)
        else:
            prev_signal = signals.iloc[i-1]
            if prev_signal != 0: # ongoing trade
                profit_x = (-shares_x * (price_x - adj_close_df[x].iloc[i-1])) * signal
                profit_y = (shares_y * (price_y - adj_close_df[y].iloc[i-1])) * signal
                profit_sum = profit_x + profit_y
                if signal != prev_signal: # exit trade
                    trade_value = (shares_x * price_x) + (shares_y * price_y)
                    trade_cost = trade_value * trade_cost_rate # exit transaction cost
                    profit_sum -= trade_cost
                if signal == prev_signal * -1: # entering new trade
                    shares_x, shares_y = calculate_position_size(capital, beta, price_y, price_x)
                    trade_cost = capital * trade_cost_rate # entry transaction cost
                    profit_sum -= trade_cost
                pnl_list.append(profit_sum)
            else: # no ongoing trade
                if signal != 0: # entering new trade
                    shares_x, shares_y = calculate_position_size(capital, beta, price_y, price_x)
                    trade_cost = capital * trade_cost_rate # entry transaction cost
                    pnl_list.append(-trade_cost)
                else: # no new trade
                    pnl_list.append(0)
    pnl_series = pd.Series(pnl_list, index = adj_close_df.index)
    return pnl_series

def plot_equity_curve(
        capital: float, 
        cointegrated_pair: tuple, 
        daily_pnl_cumsum: pd.Series, 
        save: bool = False
    ) -> None:
    """
    Plot the strategy's equity curve (portfolio value over time) for one pair.

    Args:
        capital: starting capital.
        cointegrated_pair: tuple where index 0 and 1 are the two ticker symbols.
        daily_pnl_cumsum: Series of cumulative daily PnL indexed by date.
        save: if True, saves the plot to ../figures/.

    Returns:
        None. Displays (and optionally saves) the plot.
    """
    t1, t2 = cointegrated_pair[0], cointegrated_pair[1]
    plt.plot(daily_pnl_cumsum + capital)
    plt.title(f"{t1}-{t2} Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    if save:
        plt.savefig(f"../figures/{t1}_{t2}_equity_curve.png", dpi=150, bbox_inches="tight")
    plt.show()

def calculate_sharpe(capital: float, daily_pnl: pd.Series) -> float:
    """
    Calculate the annualized Sharpe ratio from daily PnL.

    Args:
        capital: starting capital, used to convert PnL to percentage returns.
        daily_pnl: Series of daily PnL values indexed by date.

    Returns:
        Annualized Sharpe ratio.
    """
    daily_percentage_returns = daily_pnl / capital
    sharpe = (daily_percentage_returns.mean() / daily_percentage_returns.std()) * np.sqrt(252)
    return sharpe 

def calculate_max_drawdown(daily_pnl_cumsum: pd.Series) -> float:
    """
    Calculate the maximum drawdown from a cumulative PnL series.

    Args:
        daily_pnl_cumsum: Series of cumulative daily PnL indexed by date.

    Returns:
        Maximum peak-to-trough decline (in dollars).
    """
    max_drawdown = 0
    current_peak = daily_pnl_cumsum.iloc[0]
    for equity in daily_pnl_cumsum:
        if equity > current_peak:
            current_peak = equity
        else:
            drawdown = current_peak - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    return max_drawdown

def calculate_cagr(
        adj_close_df: pd.DataFrame, 
        capital: float, 
        daily_pnl_cumsum: pd.Series
    ) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR) of the strategy.

    Args:
        adj_close_df: DataFrame of adjusted close prices, used to determine the number of years.
        capital: starting capital.
        daily_pnl_cumsum: Series of cumulative daily PnL indexed by date.

    Returns:
        Annualized growth rate as a decimal (e.g. 0.039 = 3.9%).
    """
    ending_value = capital + daily_pnl_cumsum.iloc[-1]
    num_years = len(adj_close_df) / 252
    cagr = (ending_value / capital) ** (1 / num_years) - 1
    return cagr

def benchmark_returns(benchmark_ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Download and compute daily percentage returns for a benchmark ticker.

    Args:
        benchmark_ticker: ticker symbol of the benchmark (e.g. "SPY").
        start_date: start date string (e.g. "2017-01-01").
        end_date: end date string (e.g. "2025-01-01").

    Returns:
        Series of daily percentage returns indexed by date.
    """
    benchmark_data = yf.download(benchmark_ticker, start = start_date, end = end_date)["Close"]
    benchmark_returns = benchmark_data.pct_change().dropna()
    return benchmark_returns

def plot_strategy_vs_benchmark(
        capital: float, 
        daily_pnl_cumsum: pd.Series, 
        cointegrated_pair: tuple, 
        benchmark_ticker: str, 
        benchmark_returns: pd.Series, 
        save: bool = False
    ) -> None:
    """
    Plot the strategy's equity curve against a benchmark's equity curve.

    Args:
        capital: starting capital.
        daily_pnl_cumsum: Series of cumulative daily PnL indexed by date.
        cointegrated_pair: tuple where index 0 and 1 are the two ticker symbols.
        benchmark_ticker: ticker symbol of the benchmark (e.g. "SPY").
        benchmark_returns: Series of daily percentage returns for the benchmark.
        save: if True, saves the plot to ../figures/.

    Returns:
        None. Displays (and optionally saves) the plot.
    """
    t1, t2 = cointegrated_pair[0], cointegrated_pair[1]
    benchmark_cumulative = (1 + benchmark_returns).cumprod() * capital
    plt.plot(daily_pnl_cumsum + capital, label="Strategy")
    plt.plot(benchmark_cumulative, label=f"{benchmark_ticker} Benchmark")
    plt.legend()
    plt.title(f"{t1}-{t2} Equity Curve vs {benchmark_ticker} Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    if save:
        plt.savefig(f"../figures/{t1}_{t2}_equity_curve_vs_{benchmark_ticker}_benchmark.png", dpi=150, bbox_inches="tight")
    plt.show()