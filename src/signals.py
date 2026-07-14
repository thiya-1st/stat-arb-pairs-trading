import pandas as pd

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