"""
Pipeline de dados para o modelo LSTM.
Coleta dados da Petrobras via yfinance, limpa, normaliza e cria sequências.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib


def coletar_dados(symbol, start_date, end_date, save_path="data"):
    """Baixa dados históricos do Yahoo Finance e salva em CSV."""
    print(f"Baixando dados de {symbol} ({start_date} até {end_date})...")
    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {symbol}")

    # corrigir colunas multi-nível do yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, f"{symbol}_historico.csv")
    df.to_csv(csv_path)
    print(f"Dados salvos: {csv_path}")
    print(f"Registros: {len(df)} | Período: {df.index[0].date()} a {df.index[-1].date()}")

    return df


def preprocessar_dados(df, coluna_alvo="Close"):
    """Limpa dados: trata nulos, duplicatas e ordena por data."""
    print("\nPré-processando...")

    # tratar nulos
    nulos = df[coluna_alvo].isnull().sum()
    if nulos > 0:
        print(f"  Nulos encontrados: {nulos} -> preenchendo com forward fill")
        df[coluna_alvo] = df[coluna_alvo].ffill()

    # remover duplicatas no índice
    duplicatas = df.index.duplicated().sum()
    if duplicatas > 0:
        df = df[~df.index.duplicated(keep='first')]

    df = df.sort_index()

    print(f"  Shape: {df.shape}")
    print(f"  Preço mín: R${df[coluna_alvo].min():.2f} | máx: R${df[coluna_alvo].max():.2f}")

    return df


def normalizar_dados(df, coluna_alvo="Close", save_path="models"):
    """Normaliza preços entre 0 e 1 com MinMaxScaler."""
    print("\nNormalizando dados (MinMaxScaler 0-1)...")

    valores = df[coluna_alvo].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_norm = scaler.fit_transform(valores)

    # salvar scaler pra usar na API depois
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))

    return dados_norm, scaler


def criar_sequencias(dados, janela=60):
    """
    Cria sequências de entrada para a LSTM.
    Cada X tem 'janela' dias e o Y é o preço do dia seguinte.
    """
    print(f"\nCriando sequências (janela={janela} dias)...")

    X, y = [], []
    for i in range(janela, len(dados)):
        X.append(dados[i - janela:i, 0])
        y.append(dados[i, 0])

    X = np.array(X)
    y = np.array(y)

    # reshape para formato da LSTM: [amostras, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    print(f"  X: {X.shape} | y: {y.shape}")
    return X, y


def dividir_dados(X, y, proporcao_treino=0.8):
    """Divide em treino e teste (sem shuffle, pois é série temporal)."""
    corte = int(len(X) * proporcao_treino)

    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    print(f"\nDivisão treino/teste:")
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Teste:  {X_test.shape[0]} amostras")

    return X_train, X_test, y_train, y_test


def pipeline_dados(symbol="PETR4.SA", start_date="2018-01-01",
                   end_date="2024-12-31", janela=60, proporcao_treino=0.8):
    """
    Executa a pipeline completa de dados.
    Retorna dicionário com dados prontos pro modelo.
    """
    df = coletar_dados(symbol, start_date, end_date)
    df = preprocessar_dados(df)
    dados_norm, scaler = normalizar_dados(df)
    X, y = criar_sequencias(dados_norm, janela)
    X_train, X_test, y_train, y_test = dividir_dados(X, y, proporcao_treino)

    return {
        "df": df,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "janela": janela,
        "symbol": symbol
    }


if __name__ == "__main__":
    resultado = pipeline_dados()
    print("\nPipeline de dados ok!")
