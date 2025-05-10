from .base import register_indicator
import ta

@register_indicator("RSI")
def calculate_rsi(df, column='close', window=14):
    df[f'RSI_{window}'] = ta.momentum.RSIIndicator(close=df[column], window=window).rsi()
    return df

@register_indicator("Stoch")
def calculate_stoch(df, column='close', window=14):
    stoch = ta.momentum.StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df[column],
        window=window
    )
    df[f'Stoch_{window}'] = stoch.stoch()
    return df

@register_indicator("CCI")
def calculate_cci(df, column='close', window=14):
    # 注意：CCI 现在在 ta.trend 模块中，并需要 high, low, close 数据
    df[f'CCI_{window}'] = ta.trend.CCIIndicator(
        high=df['high'],
        low=df['low'],
        close=df[column],
        window=window
    ).cci()
    return df

@register_indicator("WILLR")
def calculate_willr(df, column='close', window=14):
    df[f'WILLR_{window}'] = ta.momentum.WilliamsRIndicator(
        high=df['high'],
        low=df['low'],
        close=df[column],
        lbp=window  # 使用 lbp 作为窗口参数
    ).williams_r()
    return df


@register_indicator("ROC")
def calculate_roc(df, column='close', window=14):
    df[f'ROC_{window}'] = ta.momentum.ROCIndicator(close=df[column], window=window).roc()
    return df
