"""
Script de treinamento do modelo LSTM para prever preços da Petrobras (PETR4.SA).
Executa: coleta de dados -> pré-processamento -> treinamento -> avaliação.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_pipeline import pipeline_dados
from modelo_lstm import criar_modelo, treinar_modelo, avaliar_modelo


def plotar_historico(historico, save_path="data"):
    """Plota curvas de loss durante o treinamento."""
    os.makedirs(save_path, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(historico['loss'], label='Treino')
    ax1.plot(historico['val_loss'], label='Validação')
    ax1.set_title('Loss (MSE)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(historico['mae'], label='Treino')
    ax2.plot(historico['val_mae'], label='Validação')
    ax2.set_title('MAE')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "treinamento_historico.png"), dpi=150)
    plt.close()
    print(f"Gráfico salvo: {save_path}/treinamento_historico.png")


def plotar_predicoes(y_real, y_pred, symbol, save_path="data"):
    """Plota preço real vs predição do modelo."""
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(y_real, label='Preço Real', color='blue')
    plt.plot(y_pred, label='Preço Predito', color='red', alpha=0.7)
    plt.title(f'Predição LSTM - {symbol}')
    plt.xlabel('Dias (teste)')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "predicoes_vs_real.png"), dpi=150)
    plt.close()
    print(f"Gráfico salvo: {save_path}/predicoes_vs_real.png")


def main():
    print("=" * 50)
    print("  LSTM - Previsão de Preços Petrobras (PETR4)")
    print("=" * 50)

    # configurações do modelo
    config = {
        "symbol": "PETR4.SA",
        "start_date": "2018-01-01",
        "end_date": "2024-12-31",
        "janela": 60,
        "proporcao_treino": 0.8,
        "epochs": 100,
        "batch_size": 32,
        "hidden_size": 50,
        "num_layers": 2,
        "learning_rate": 0.001
    }

    print("\nConfigurações:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # salvar config
    os.makedirs("models", exist_ok=True)
    with open("models/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 1. coleta e preparação dos dados
    print("\n" + "-" * 50)
    print("  ETAPA 1: Coleta e pré-processamento")
    print("-" * 50)
    dados = pipeline_dados(
        symbol=config["symbol"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        janela=config["janela"],
        proporcao_treino=config["proporcao_treino"]
    )

    # 2. criar modelo
    print("\n" + "-" * 50)
    print("  ETAPA 2: Criação do modelo LSTM")
    print("-" * 50)
    input_shape = (dados["X_train"].shape[1], dados["X_train"].shape[2])
    modelo = criar_modelo(input_shape, hidden_size=config["hidden_size"],
                          num_layers=config["num_layers"])

    # 3. treinar
    print("\n" + "-" * 50)
    print("  ETAPA 3: Treinamento")
    print("-" * 50)
    resultado = treinar_modelo(
        modelo=modelo,
        X_train=dados["X_train"],
        y_train=dados["y_train"],
        X_val=dados["X_test"],
        y_val=dados["y_test"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["learning_rate"]
    )

    # 4. avaliar
    print("\n" + "-" * 50)
    print("  ETAPA 4: Avaliação")
    print("-" * 50)
    avaliacao = avaliar_modelo(
        modelo=resultado["model"],
        X_test=dados["X_test"],
        y_test=dados["y_test"],
        scaler=dados["scaler"]
    )

    # salvar métricas
    metricas = {k: float(v) for k, v in avaliacao["metricas"].items()}
    with open("models/metricas.json", "w") as f:
        json.dump(metricas, f, indent=2)
    print("\nMétricas salvas em models/metricas.json")

    # 5. gráficos
    print("\n" + "-" * 50)
    print("  ETAPA 5: Gráficos")
    print("-" * 50)
    plotar_historico(resultado["history"])
    plotar_predicoes(avaliacao["y_real"], avaliacao["y_pred"], config["symbol"])

    print("\n" + "=" * 50)
    print("  Pipeline finalizada!")
    print("=" * 50)
    print(f"\nArquivos gerados:")
    print(f"  models/lstm_model.pth")
    print(f"  models/scaler.pkl")
    print(f"  models/config.json")
    print(f"  models/metricas.json")
    print(f"  data/{config['symbol']}_historico.csv")
    print(f"  data/treinamento_historico.png")
    print(f"  data/predicoes_vs_real.png")


if __name__ == "__main__":
    main()
