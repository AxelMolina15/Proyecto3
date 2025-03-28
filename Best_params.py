import numpy as np
import ta
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def run_strategy(data, params):
    data = data.copy()

    rsi_window = params["rsi_window"]
    rsi_upper = params["rsi_upper"]
    rsi_lower = params["rsi_lower"]
    stop_loss = params["stop_loss"]
    take_profit = params["take_profit"]
    bb_window = params["bb_window"]
    bb_std = params["bb_std"]
    macd_short = params["macd_short"]
    macd_long = params["macd_long"]
    macd_signal = params["macd_signal"]
    n_shares = params["n_shares"]

    # RSI
    rsi = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    data["RSI"] = rsi.rsi()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(data.Close, window=bb_window, window_dev=bb_std)
    data["BB"] = bb.bollinger_mavg()
    data["BB_BUY"] = bb.bollinger_lband_indicator().astype(bool)
    data["BB_SELL"] = bb.bollinger_hband_indicator().astype(bool)

    # MACD
    macd = ta.trend.MACD(data.Close, window_slow=macd_long, window_fast=macd_short, window_sign=macd_signal)
    data["MACD"] = macd.macd()
    data["MACD_SIGNAL"] = macd.macd_signal()

    dataset = data.dropna()

    capital = 1_000_000
    com = 0.125 / 100
    portfolio_value = [capital]
    active_long = None
    active_short = None
    win = 0
    loss = 0

    for i, row in dataset.iterrows():
        # Close long positions
        if active_long:
            if row.Close >= active_long["take_profit"] or row.Close <= active_long["stop_loss"]:
                pnl = row.Close * n_shares * (1 - com)
                capital += pnl
                win += 1 if row.Close >= active_long["take_profit"] else 0
                loss += 1 if row.Close <= active_long["stop_loss"] else 0
                active_long = None

        # Close short positions
        if active_short:
            if row.Close <= active_short["take_profit"] or row.Close >= active_short["stop_loss"]:
                pnl = (active_short["opened_at"] - row.Close) * n_shares * (1 - com)
                capital += pnl
                win += 1 if row.Close <= active_short["take_profit"] else 0
                loss += 1 if row.Close >= active_short["stop_loss"] else 0
                active_short = None

        # Open long position
        if sum([row.RSI < rsi_lower, row.BB_BUY, row.MACD > row.MACD_SIGNAL]) >= 2 and not active_long:
            cost = row.Close * n_shares * (1 + com)
            if capital >= cost:
                capital -= cost
                active_long = {"opened_at": row.Close, "take_profit": row.Close * (1 + take_profit),
                               "stop_loss": row.Close * (1 - stop_loss)}

        # Open short position
        if sum([row.RSI > rsi_upper, row.BB_SELL, row.MACD < row.MACD_SIGNAL]) >= 2 and not active_short:
            cost = row.Close * n_shares * com
            if capital >= cost:
                capital -= cost
                active_short = {"opened_at": row.Close, "take_profit": row.Close * (1 - take_profit),
                                "stop_loss": row.Close * (1 + stop_loss)}

        # Update portfolio value
        long_val = row.Close * n_shares if active_long else 0
        short_val = (active_short["opened_at"] - row.Close) * n_shares if active_short else 0
        portfolio_value.append(capital + long_val + short_val)

    # Cálculo de métricas de rendimiento
    rets = pd.Series(portfolio_value).pct_change().dropna()
    er = rets.mean()
    ev= rets.std()
    dt = (252)*(6.5)*(60/5)
    sharpe_ratio = (er*dt)/(ev*np.sqrt(dt))
    returns = np.diff(portfolio_value) / portfolio_value[:-1] # actual menos el anterior, entre el anterior
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns)
    sortino_ratio = (np.mean(returns) * dt) / (downside_std * np.sqrt(dt)) if downside_std != 0 else 0
    calmar_ratio = (np.mean(returns) * dt) / abs(min(returns)) if min(returns) != 0 else 0
    win_loss_ratio = win / (win + loss) if (win + loss) != 0 else 0

    # Plot portfolio value
    plt.figure(figsize=(12, 6))
    plt.title("Strategy")
    plt.plot(portfolio_value, label="Portfolio Value")
    plt.legend()
    plt.show()

    print(f"Final Portfolio Value: {portfolio_value[-1]:,.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Sortino Ratio: {sortino_ratio:.4f}")
    print(f"Calmar Ratio: {calmar_ratio:.4f}")
    print(f"Win/Loss Ratio: {win_loss_ratio:.4f}")

    return sharpe_ratio if not np.isnan(sharpe_ratio) else -np.inf

def main():
    print("Running strategy with best parameters...")

    data = pd.read_csv("aapl_5m_train.csv").dropna()

    # Cargar parámetros optimizados
    with open("best_params.pkl", "rb") as f:
        best_params = pickle.load(f)

    print("Loaded best parameters:", best_params)

    run_strategy(data, best_params)

if __name__ == "__main__":
    main()