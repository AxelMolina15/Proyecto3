import numpy as np
import ta
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import pickle


def objective(trial, data):
    data = data.copy()

    # Hiperparámetros de la estrategia
    rsi_window = trial.suggest_int("rsi_window", 10, 100)
    rsi_upper = trial.suggest_int("rsi_upper", 70, 95)
    rsi_lower = trial.suggest_int("rsi_lower", 5, 30)

    stop_loss = trial.suggest_float("stop_loss", 0.04, 0.12)
    take_profit = trial.suggest_float("take_profit", 0.04, 0.12)

    bb_window = trial.suggest_int("bb_window", 10, 100)
    bb_std = trial.suggest_int("bb_std", 1, 3)

    macd_short = trial.suggest_int("macd_short", 10, 50)
    macd_long = trial.suggest_int("macd_long", 50, 200)
    macd_signal = trial.suggest_int("macd_signal", 5, 20)

    # Nuevo: Optimización de número de acciones a operar
    n_shares = trial.suggest_int("n_shares", 100, 5000, step=100)

    # Cálculo de indicadores
    rsi = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    data["RSI"] = rsi.rsi()

    bb = ta.volatility.BollingerBands(data.Close, window=bb_window, window_dev=bb_std)
    data["BB"] = bb.bollinger_mavg()

    macd = ta.trend.MACD(data.Close, window_slow=macd_long, window_fast=macd_short, window_sign=macd_signal)
    data["MACD"] = macd.macd()
    data["MACD_SIGNAL"] = macd.macd_signal()

    dataset = data.dropna()

    # Parámetros de simulación
    capital = 1_000_000
    com = 0.125 / 100
    portfolio_value = [capital]
    active_long = None
    active_short = None
    win = 0
    loss = 0

    for i, row in dataset.iterrows():
        # Cerrar posiciones largas
        if active_long:
            if row.Close >= active_long["take_profit"] or row.Close <= active_long["stop_loss"]:
                pnl = row.Close * active_long["n_shares"] * (1 - com)
                capital += pnl
                win += 1 if row.Close >= active_long["take_profit"] else 0
                loss += 1 if row.Close <= active_long["stop_loss"] else 0
                active_long = None

        # Cerrar posiciones cortas
        if active_short:
            if row.Close <= active_short["take_profit"] or row.Close >= active_short["stop_loss"]:
                pnl = (active_short["opened_at"] - row.Close) * active_short["n_shares"] * (1 - com)
                capital += pnl
                win += 1 if row.Close <= active_short["take_profit"] else 0
                loss += 1 if row.Close >= active_short["stop_loss"] else 0
                active_short = None

        # Abrir posición larga
        if row.RSI < rsi_lower and row.Close < row.BB and row.MACD > row.MACD_SIGNAL and not active_long:
            cost = row.Close * n_shares * (1 + com)
            if capital >= cost:
                capital -= cost
                active_long = {
                    "opened_at": row.Close,
                    "take_profit": row.Close * (1 + take_profit),
                    "stop_loss": row.Close * (1 - stop_loss),
                    "n_shares": n_shares
                }

        # Abrir posición corta
        if row.RSI > rsi_upper and row.Close > row.BB and row.MACD < row.MACD_SIGNAL and not active_short:
            cost = row.Close * n_shares * (1 + com)
            if capital >= cost:
                capital -= cost
                active_short = {
                    "opened_at": row.Close,
                    "take_profit": row.Close * (1 - take_profit),
                    "stop_loss": row.Close * (1 + stop_loss),
                    "n_shares": n_shares
                }

        # Actualizar valor del portafolio
        long_val = active_long["opened_at"] * active_long["n_shares"] if active_long else 0
        short_val = (active_short["opened_at"] - row.Close) * active_short["n_shares"] if active_short else 0
        portfolio_value.append(capital + long_val + short_val)

    # Cálculo de métricas de rendimiento
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    downside_returns = returns[returns < 0]
    sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if np.std(downside_returns) != 0 else 0
    calmar_ratio = np.mean(returns) * 252 / abs(min(returns)) if min(returns) != 0 else 0
    win_loss_ratio = win / (win + loss) if (win + loss) != 0 else 0

    return portfolio_value[-1]


def main():
    print("Running Optuna optimization...")

    data = pd.read_csv("/Users/axelmolina/Desktop/Noveno Semestre/Trading/Proyecto3/aapl_5m_train.csv").dropna()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda x: objective(x, data), n_trials=50)

    # Guardar mejores parámetros
    with open("best_params.pkl", "wb") as f:
        pickle.dump(study.best_params, f)

    print("Best parameters saved:", study.best_params)


if __name__ == "__main__":
    main()


