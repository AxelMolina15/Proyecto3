# Proyecto3

# 📈 Optimizing a Trading Strategy Using Optuna: A Case Study on AAPL

### Autores: Axel Santiago Molina Ceja & Pablo Lemus  
### Proyecto de Trading – Noveno Semestre

## 🧠 Descripción del Proyecto

Este proyecto tiene como objetivo diseñar, optimizar y evaluar una estrategia de trading cuantitativa sobre acciones de Apple Inc. (AAPL), utilizando el framework de optimización bayesiana **Optuna**.

La estrategia combina tres indicadores técnicos:
- **RSI (Relative Strength Index)**
- **Bollinger Bands**
- **MACD (Moving Average Convergence Divergence)**

Además de las reglas de entrada y salida, se optimiza también el número de acciones (`n_shares`) por operación.

## ⚙️ ¿Qué hace el código?

1. Calcula indicadores técnicos sobre datos históricos de AAPL (intervalo de 5 minutos).
2. Define una estrategia de entrada/salida usando múltiples señales técnicas.
3. Ejecuta una simulación de backtesting sobre el portafolio.
4. Usa **Optuna** para encontrar los parámetros óptimos que maximizan el **Sharpe Ratio**.
5. Evalúa el desempeño con métricas como:
   - Sharpe Ratio
   - Sortino Ratio
   - Calmar Ratio
   - Win/Loss Ratio
6. Grafica la evolución del portafolio a lo largo del tiempo.

## 📊 Resultados

El mejor set de parámetros obtuvo:

- **Sharpe Ratio:** 2.08  
- **Sortino Ratio:** 1.56  
- **Calmar Ratio:** 11.60  
- **Win/Loss Ratio:** 0.80  
- **Valor final del portafolio:** \$1,283,198.13

## 📁 Estructura del proyecto


```bash
📦 proyecto_trading_optuna/
 ┣ 📄 notebook.ipynb        # Notebook con todo el desarrollo, explicación y resultados
 ┣ 📄 aapl_5m_train.csv     # Datos históricos de AAPL (intervalo 5 minutos)
 ┣ 📄 README.md             # Este archivo

```

## 📌 Requisitos

- Python 3.10+
- Pandas
- Numpy
- Matplotlib
- TA-Lib (`ta`)
- Optuna
- Jupyter Notebook

Puedes instalar las dependencias con:

```bash
pip install pandas numpy matplotlib ta optuna
