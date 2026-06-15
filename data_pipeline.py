"""
Módulo de coleta e pré-processamento de dados de ações.
Pipeline: coleta via yfinance -> limpeza -> normalização -> criação de sequências para LSTM.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib


def coletar_dados(symbol: str, start_date: str, end_date: str, save_path: str = "data") -> pd.DataFrame:
    """
    Coleta dados históricos de ações via Yahoo Finance.
    
    Args:
        symbol: Símbolo da ação (ex: 'DIS' para Disney)
        start_date: Data inicial no formato 'YYYY-MM-DD'
        end_date: Data final no formato 'YYYY-MM-DD'
        save_path: Diretório para salvar o CSV
    
    Returns:
        DataFrame com os dados históricos
    """
    print(f"Coletando dados de {symbol} de {start_date} até {end_date}...")
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para o símbolo {symbol}")
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, f"{symbol}_historico.csv")
    df.to_csv(csv_path)
    print(f"Dados salvos em {csv_path}")
    print(f"Shape dos dados: {df.shape}")
    print(f"Período: {df.index[0]} até {df.index[-1]}")
    
    return df


def preprocessar_dados(df: pd.DataFrame, coluna_alvo: str = "Close") -> pd.DataFrame:
    """
    Realiza limpeza e pré-processamento dos dados.
    
    Args:
        df: DataFrame com dados brutos
        coluna_alvo: Nome da coluna alvo para predição
    
    Returns:
        DataFrame limpo
    """
    print("\nPré-processando dados...")
    
    # Verificar valores nulos
    nulos = df[coluna_alvo].isnull().sum()
    if nulos > 0:
        print(f"Valores nulos encontrados na coluna {coluna_alvo}: {nulos}")
        df[coluna_alvo] = df[coluna_alvo].ffill()
        print("Valores nulos preenchidos com forward fill.")
    
    # Verificar e remover duplicatas no índice
    duplicatas = df.index.duplicated().sum()
    if duplicatas > 0:
        print(f"Índices duplicados encontrados: {duplicatas}")
        df = df[~df.index.duplicated(keep='first')]
    
    # Ordenar por data
    df = df.sort_index()
    
    print(f"Dados pré-processados. Shape final: {df.shape}")
    print(f"\nEstatísticas da coluna '{coluna_alvo}':")
    print(df[coluna_alvo].describe())
    
    return df


def normalizar_dados(df: pd.DataFrame, coluna_alvo: str = "Close", save_path: str = "models") -> tuple:
    """
    Normaliza os dados usando MinMaxScaler.
    
    Args:
        df: DataFrame com dados limpos
        coluna_alvo: Coluna a ser normalizada
        save_path: Diretório para salvar o scaler
    
    Returns:
        Tuple com (dados_normalizados, scaler)
    """
    print("\nNormalizando dados...")
    
    valores = df[coluna_alvo].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(valores)
    
    # Salvar o scaler para uso posterior na API
    os.makedirs(save_path, exist_ok=True)
    scaler_path = os.path.join(save_path, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler salvo em {scaler_path}")
    
    return dados_normalizados, scaler


def criar_sequencias(dados: np.ndarray, janela: int = 60) -> tuple:
    """
    Cria sequências temporais para alimentar a LSTM.
    
    Cada sequência X contém 'janela' dias de preços e o Y correspondente
    é o preço do dia seguinte.
    
    Args:
        dados: Array normalizado de preços
        janela: Número de dias anteriores usados como input (lookback)
    
    Returns:
        Tuple com (X, y) arrays numpy
    """
    print(f"\nCriando sequências com janela de {janela} dias...")
    
    X, y = [], []
    for i in range(janela, len(dados)):
        X.append(dados[i - janela:i, 0])
        y.append(dados[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X para [amostras, timesteps, features] - formato LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print(f"Shape X: {X.shape}")
    print(f"Shape y: {y.shape}")
    
    return X, y


def dividir_dados(X: np.ndarray, y: np.ndarray, proporcao_treino: float = 0.8) -> tuple:
    """
    Divide os dados em treino e teste.
    
    Args:
        X: Features
        y: Target
        proporcao_treino: Proporção dos dados para treino
    
    Returns:
        Tuple com (X_train, X_test, y_train, y_test)
    """
    tamanho_treino = int(len(X) * proporcao_treino)
    
    X_train = X[:tamanho_treino]
    X_test = X[tamanho_treino:]
    y_train = y[:tamanho_treino]
    y_test = y[tamanho_treino:]
    
    print(f"\nDivisão dos dados:")
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")
    
    return X_train, X_test, y_train, y_test


def pipeline_dados(symbol: str = "DIS", start_date: str = "2018-01-01", 
                   end_date: str = "2024-07-20", janela: int = 60,
                   proporcao_treino: float = 0.8) -> dict:
    """
    Executa toda a pipeline de dados.
    
    Returns:
        Dicionário com todos os dados processados
    """
    # 1. Coleta
    df = coletar_dados(symbol, start_date, end_date)
    
    # 2. Pré-processamento
    df = preprocessar_dados(df)
    
    # 3. Normalização
    dados_normalizados, scaler = normalizar_dados(df)
    
    # 4. Criar sequências
    X, y = criar_sequencias(dados_normalizados, janela)
    
    # 5. Dividir em treino/teste
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
    print("\n✅ Pipeline de dados concluída com sucesso!")
