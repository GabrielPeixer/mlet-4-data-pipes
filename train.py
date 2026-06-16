"""
Script principal para executar toda a pipeline de treinamento.
Desde a coleta de dados até a avaliação do modelo LSTM.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from data_pipeline import pipeline_dados
from modelo_lstm import criar_modelo_lstm, treinar_modelo, avaliar_modelo


def plotar_historico_treinamento(history, save_path: str = "data"):
    """Plota as curvas de loss do treinamento."""
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "treinamento_historico.png"), dpi=150)
    plt.close()
    print(f"Gráfico de treinamento salvo em {save_path}/treinamento_historico.png")


def plotar_predicoes(y_real, y_pred, symbol: str, save_path: str = "data"):
    """Plota comparação entre valores reais e preditos."""
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_real, label='Preço Real', color='blue', linewidth=1.5)
    plt.plot(y_pred, label='Preço Predito', color='red', linewidth=1.5, alpha=0.8)
    plt.title(f'Predição de Preço de Fechamento - {symbol}')
    plt.xlabel('Dias (conjunto de teste)')
    plt.ylabel('Preço (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "predicoes_vs_real.png"), dpi=150)
    plt.close()
    print(f"Gráfico de predições salvo em {save_path}/predicoes_vs_real.png")


def main():
    """Executa a pipeline completa de treinamento."""
    print("=" * 60)
    print("   PIPELINE DE PREDIÇÃO DE AÇÕES COM LSTM")
    print("=" * 60)
    
    # Configurações
    CONFIG = {
        "symbol": "DIS",
        "start_date": "2018-01-01",
        "end_date": "2024-07-20",
        "janela": 60,
        "proporcao_treino": 0.8,
        "epochs": 100,
        "batch_size": 32,
        "units_lstm": [50, 50]
    }
    
    print(f"\nConfiguração:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    # Salvar configuração
    os.makedirs("models", exist_ok=True)
    with open("models/config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    # 1. Pipeline de dados
    print("\n" + "=" * 60)
    print("   ETAPA 1: COLETA E PRÉ-PROCESSAMENTO")
    print("=" * 60)
    dados = pipeline_dados(
        symbol=CONFIG["symbol"],
        start_date=CONFIG["start_date"],
        end_date=CONFIG["end_date"],
        janela=CONFIG["janela"],
        proporcao_treino=CONFIG["proporcao_treino"]
    )
    
    # 2. Criar modelo
    print("\n" + "=" * 60)
    print("   ETAPA 2: CRIAÇÃO DO MODELO LSTM")
    print("=" * 60)
    input_shape = (dados["X_train"].shape[1], dados["X_train"].shape[2])
    modelo = criar_modelo_lstm(input_shape, CONFIG["units_lstm"])
    
    # 3. Treinar modelo
    print("\n" + "=" * 60)
    print("   ETAPA 3: TREINAMENTO")
    print("=" * 60)
    resultado_treino = treinar_modelo(
        model=modelo,
        X_train=dados["X_train"],
        y_train=dados["y_train"],
        X_test=dados["X_test"],
        y_test=dados["y_test"],
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"]
    )
    
    # 4. Avaliar modelo
    print("\n" + "=" * 60)
    print("   ETAPA 4: AVALIAÇÃO")
    print("=" * 60)
    resultado_avaliacao = avaliar_modelo(
        model=resultado_treino["model"],
        X_test=dados["X_test"],
        y_test=dados["y_test"],
        scaler=dados["scaler"]
    )
    
    # 5. Salvar métricas
    metricas_serializaveis = {k: float(v) for k, v in resultado_avaliacao["metricas"].items()}
    with open("models/metricas.json", "w") as f:
        json.dump(metricas_serializaveis, f, indent=2)
    print("\nMétricas salvas em models/metricas.json")
    
    # 6. Gerar gráficos
    print("\n" + "=" * 60)
    print("   ETAPA 5: VISUALIZAÇÕES")
    print("=" * 60)
    plotar_historico_treinamento(resultado_treino["history"])
    plotar_predicoes(
        resultado_avaliacao["y_real"],
        resultado_avaliacao["y_pred"],
        CONFIG["symbol"]
    )
    
    print("\n" + "=" * 60)
    print("   ✅ PIPELINE CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    print(f"\nArquivos gerados:")
    print(f"  - models/lstm_model.keras (modelo treinado)")
    print(f"  - models/scaler.pkl (normalizador)")
    print(f"  - models/config.json (configurações)")
    print(f"  - models/metricas.json (métricas de avaliação)")
    print(f"  - data/{CONFIG['symbol']}_historico.csv (dados brutos)")
    print(f"  - data/treinamento_historico.png (curvas de loss)")
    print(f"  - data/predicoes_vs_real.png (predições vs real)")
    print(f"\nPara iniciar a API de predição, execute:")
    print(f"  python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
