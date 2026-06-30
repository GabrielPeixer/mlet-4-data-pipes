"""
Modelo LSTM para prever preço de ações da Petrobras.
Usa PyTorch para construir e treinar a rede neural.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ModeloLSTM(nn.Module):
    """Rede LSTM para previsão de séries temporais de preços."""

    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(ModeloLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # camadas LSTM empilhadas
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # camada de saída
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # passa pela LSTM
        out, _ = self.lstm(x)
        # pega só a saída do último timestep
        out = out[:, -1, :]
        # camada linear final: aprende apenas o AJUSTE sobre o último preço
        # conhecido (conexão residual), em vez do preço absoluto. Isso evita
        # que a rede gaste capacidade reaprendendo o nível da série (que já
        # é ~igual ao dia anterior) e foca no que realmente varia.
        ajuste = self.fc(out)
        ultimo_preco = x[:, -1, :]
        return ultimo_preco + ajuste


def criar_modelo(input_shape, hidden_size=50, num_layers=2):
    """
    Cria o modelo LSTM.
    
    Parâmetros:
        input_shape: formato (timesteps, features)
        hidden_size: neurônios na LSTM
        num_layers: quantidade de camadas LSTM
    """
    input_size = input_shape[1]
    modelo = ModeloLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    n_params = sum(p.numel() for p in modelo.parameters())
    print(f"\nModelo LSTM criado:")
    print(f"  Input: {input_shape}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Camadas LSTM: {num_layers}")
    print(f"  Parâmetros: {n_params:,}")
    print(modelo)

    return modelo


def treinar_modelo(modelo, X_train, y_train, X_val, y_val,
                   epochs=100, batch_size=32, lr=0.001, patience=10,
                   save_path="models"):
    """
    Treina o modelo com early stopping.
    Salva o melhor modelo baseado na loss de validação.
    """
    print(f"\n--- Treinamento ---")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Patience: {patience}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    modelo = modelo.to(device)

    # preparar dados
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # otimizador e loss
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # controle do early stopping
    melhor_loss = float('inf')
    contador_paciencia = 0
    melhor_pesos = None

    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "lstm_model.pth")

    # historico para plotar depois
    historico = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

    for epoch in range(epochs):
        # treino
        modelo.train()
        loss_acumulada = 0
        mae_acumulado = 0
        n_batches = 0

        for xb, yb in loader:
            optimizer.zero_grad()
            pred = modelo(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            loss_acumulada += loss.item()
            mae_acumulado += torch.mean(torch.abs(pred - yb)).item()
            n_batches += 1

        loss_treino = loss_acumulada / n_batches
        mae_treino = mae_acumulado / n_batches

        # validação
        modelo.eval()
        with torch.no_grad():
            pred_val = modelo(X_val_t)
            loss_val = loss_fn(pred_val, y_val_t).item()
            mae_val = torch.mean(torch.abs(pred_val - y_val_t)).item()

        historico['loss'].append(loss_treino)
        historico['val_loss'].append(loss_val)
        historico['mae'].append(mae_treino)
        historico['val_mae'].append(mae_val)

        # checa se melhorou
        if loss_val < melhor_loss:
            melhor_loss = loss_val
            contador_paciencia = 0
            melhor_pesos = modelo.state_dict().copy()
            torch.save(modelo.state_dict(), model_path)
        else:
            contador_paciencia += 1

        # print a cada 10 épocas
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | loss: {loss_treino:.6f} | val_loss: {loss_val:.6f} | val_mae: {mae_val:.6f}")

        # early stopping
        if contador_paciencia >= patience:
            print(f"\n  Early stopping! Parou na epoch {epoch+1}")
            break

    # restaurar melhor modelo
    if melhor_pesos:
        modelo.load_state_dict(melhor_pesos)

    print(f"  Melhor val_loss: {melhor_loss:.6f}")
    print(f"  Modelo salvo em: {model_path}")

    return {"model": modelo, "history": historico, "model_path": model_path}


def avaliar_modelo(modelo, X_test, y_test, scaler):
    """
    Avalia o modelo calculando MAE, RMSE, MAPE e R².
    Inverte a normalização para comparar em valores reais (R$).

    Também compara o modelo com um baseline ingênuo (prever que o preço de
    amanhã = preço de hoje) e calcula a acurácia direcional, pois em séries
    de preços (quase um random walk) o R² sozinho costuma enganar: um
    baseline "preguiçoso" já atinge R² alto só pela autocorrelação da série.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    device = next(modelo.parameters()).device
    modelo.eval()

    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        pred_norm = modelo(X_t).cpu().numpy()

    # voltar pra escala original
    y_pred = scaler.inverse_transform(pred_norm)
    y_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    # métricas do modelo
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100

    # baseline ingênuo: "preço de amanhã = último preço conhecido" (último
    # dia da janela de entrada, que é o dia anterior ao alvo)
    ultimo_preco_norm = X_test[:, -1, :1].reshape(-1, 1)
    y_naive = scaler.inverse_transform(ultimo_preco_norm)

    mae_naive = mean_absolute_error(y_real, y_naive)
    mse_naive = mean_squared_error(y_real, y_naive)
    rmse_naive = np.sqrt(mse_naive)
    r2_naive = r2_score(y_real, y_naive)
    mape_naive = np.mean(np.abs((y_real - y_naive) / y_real)) * 100

    # acurácia direcional: o modelo acerta se sobe/desce em relação ao
    # último preço conhecido, na mesma direção do movimento real
    direcao_real = np.sign(y_real - y_naive)
    direcao_prevista = np.sign(y_pred - y_naive)
    acuracia_direcional = np.mean(direcao_real == direcao_prevista) * 100

    print(f"\n--- Métricas de Avaliação (modelo LSTM) ---")
    print(f"  MAE:  R$ {mae:.4f}")
    print(f"  RMSE: R$ {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    print(f"\n--- Baseline ingênuo (preço de amanhã = preço de hoje) ---")
    print(f"  MAE:  R$ {mae_naive:.4f}")
    print(f"  RMSE: R$ {rmse_naive:.4f}")
    print(f"  MAPE: {mape_naive:.2f}%")
    print(f"  R²:   {r2_naive:.4f}")

    print(f"\n--- Acurácia direcional (sobe/desce) ---")
    print(f"  Modelo acerta a direção em {acuracia_direcional:.2f}% dos dias")
    if rmse >= rmse_naive:
        print(f"  ⚠ O modelo NÃO supera o baseline ingênuo (RMSE pior ou igual).")
    else:
        melhora = (1 - rmse / rmse_naive) * 100
        print(f"  O modelo supera o baseline ingênuo em RMSE ({melhora:.1f}% melhor).")

    return {
        "metricas": {
            "MAE": mae, "RMSE": rmse, "MAPE": mape, "MSE": mse, "R2": r2,
            "MAE_baseline": mae_naive, "RMSE_baseline": rmse_naive,
            "MAPE_baseline": mape_naive, "MSE_baseline": mse_naive, "R2_baseline": r2_naive,
            "Acuracia_Direcional": acuracia_direcional
        },
        "y_pred": y_pred,
        "y_real": y_real,
        "y_naive": y_naive
    }


def carregar_modelo(model_path="models/lstm_model.pth", input_size=1,
                    hidden_size=50, num_layers=2):
    """Carrega modelo salvo do disco."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    modelo = ModeloLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    modelo.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    modelo.eval()
    print(f"Modelo carregado: {model_path}")
    return modelo
