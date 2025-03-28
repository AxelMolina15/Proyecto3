import pandas as pd
import Optuna
import Best_params

data = pd.read_csv("aapl_5m_train.csv").dropna()

if __name__ == "__main__":
    Optuna.main()
    Best_params.main()


