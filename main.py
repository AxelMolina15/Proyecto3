import pandas as pd
import Optuna
import Best_params

data = pd.read_csv("/Users/axelmolina/Desktop/Noveno Semestre/Trading/Proyecto3/aapl_5m_train.csv").dropna()

if __name__ == "__main__":
    Optuna.main()
    Best_params.main()


