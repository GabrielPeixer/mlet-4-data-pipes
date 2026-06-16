"""
Módulo de definição e treinamento do modelo LSTM.
Arquitetura: LSTM multicamadas para predição de séries temporais financeiras.
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def criar_modelo_lstm(input_shape: tuple, units_lstm: list = None) -> Sequential:
    """
    Cria a arquitetura do modelo LSTM.
    
    Arquitetura:
        - 2 camadas LSTM com Dropout para regularização
        - 1 camada Dense de saída
    
    Args:
        input_shape: Shape de entrada (timesteps, features)
        units_lstm: Lista com número de neurônios por camada LSTM
    
    Returns:
        Modelo Keras compilado
    """
    if units_lstm is None:
        units_lstm = [50, 50]
    
    print(f"\nCriando modelo LSTM...")
    print(f"Input shape: {input_shape}")
    print(f"Neurônios por camada: {units_lstm}")
    
    model = Sequential()
    
    # Primeira camada LSTM - return_sequences=True para empilhar LSTMs
    model.add(LSTM(
        units=units_lstm[0],
        return_sequences=True,
        input_shape=input_shape
    ))
    model.add(Dropout(0.2))
    
    # Segunda camada LSTM
    model.add(LSTM(
        units=units_lstm[1],
        return_sequences=False
    ))
    model.add(Dropout(0.2))
    
    # Camada de saída - um neurônio para predição do preço
    model.add(Dense(units=1))
    
    # Compilar modelo
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    model.summary()
    
    return model


def treinar_modelo(model: Sequential, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   epochs: int = 100, batch_size: int = 32,
                   save_path: str = "models") -> dict:
    """
    Treina o modelo LSTM com Early Stopping.
    
    Args:
        model: Modelo Keras compilado
        X_train: Dados de treino
        y_train: Labels de treino
        X_test: Dados de validação
        y_test: Labels de validação
        epochs: Número máximo de épocas
        batch_size: Tamanho do batch
        save_path: Diretório para salvar o modelo
    
    Returns:
        Dicionário com histórico de treinamento e modelo treinado
    """
    print(f"\nIniciando treinamento...")
    print(f"Épocas: {epochs} | Batch size: {batch_size}")
    
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "lstm_model.keras")
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Treinamento
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    print(f"\nModelo salvo em {model_path}")
    print(f"Melhor val_loss: {min(history.history['val_loss']):.6f}")
    
    return {
        "model": model,
        "history": history,
        "model_path": model_path
    }


def avaliar_modelo(model: Sequential, X_test: np.ndarray, y_test: np.ndarray,
                   scaler) -> dict:
    """
    Avalia o modelo nos dados de teste.
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste
        y_test: Labels de teste
        scaler: MinMaxScaler para inverter normalização
    
    Returns:
        Dicionário com métricas de avaliação
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    print("\nAvaliando modelo...")
    
    # Predições
    y_pred_normalizado = model.predict(X_test)
    
    # Inverter normalização para obter valores reais
    y_pred = scaler.inverse_transform(y_pred_normalizado)
    y_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calcular métricas
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    
    metricas = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape
    }
    
    print("\n📊 Métricas de Avaliação:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        "metricas": metricas,
        "y_pred": y_pred,
        "y_real": y_real
    }


def carregar_modelo(model_path: str = "models/lstm_model.keras") -> Sequential:
    """
    Carrega um modelo treinado do disco.
    
    Args:
        model_path: Caminho do modelo salvo
    
    Returns:
        Modelo Keras carregado
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
    
    model = load_model(model_path)
    print(f"Modelo carregado de {model_path}")
    return model
