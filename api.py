"""
API para previsão de preços de ações usando modelo LSTM treinado.
Usa FastAPI para servir predições da Petrobras (PETR4.SA).
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from modelo_lstm import ModeloLSTM, carregar_modelo
from datetime import datetime, timedelta


# Inicializar FastAPI
app = FastAPI(
    title="API Predição Petrobras - LSTM",
    description="Previsão de preço de fechamento da PETR4 com LSTM",
    version="1.0.0"
)

# Variáveis globais para o modelo e scaler
model = None
scaler = None
config = None


def carregar_artefatos():
    """Carrega modelo, scaler e configurações ao iniciar a API."""
    global model, scaler, config
    
    model_path = "models/lstm_model.pth"
    scaler_path = "models/scaler.pkl"
    config_path = "models/config.json"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. Execute train.py primeiro."
        )
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    hidden_size = config.get("hidden_size", 50)
    num_layers = config.get("num_layers", 2)
    model = carregar_modelo(model_path, input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers)
    scaler = joblib.load(scaler_path)
    
    print(f"Modelo carregado: {model_path}")
    print(f"Ação: {config['symbol']} | Janela: {config['janela']} dias")


# Carregar artefatos na inicialização
@app.on_event("startup")
async def startup_event():
    carregar_artefatos()


# Schemas de request/response
class PredictionRequest(BaseModel):
    symbol: str = Field(
        default="PETR4.SA",
        description="Símbolo da ação (deve ser o mesmo usado no treinamento)"
    )
    dias_futuros: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Número de dias futuros para predizer (1 a 30)"
    )
    precos_historicos: list[float] | None = Field(
        default=None,
        description=(
            "Lista opcional de preços de fechamento históricos fornecidos pelo "
            "usuário (do mais antigo para o mais recente). Se informado, deve "
            "conter ao menos 'janela' valores e será usado no lugar dos dados "
            "obtidos automaticamente via Yahoo Finance."
        )
    )


class PredictionResponse(BaseModel):
    symbol: str
    data_referencia: str
    predicoes: list[dict]
    modelo_info: dict


class HealthResponse(BaseModel):
    status: str
    modelo_carregado: bool
    symbol_treinado: str
    janela_temporal: int


# Endpoints
@app.get("/", tags=["Info"])
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "api": "Predição de Ações com LSTM",
        "versao": "1.0.0",
        "endpoints": {
            "/health": "Status da API",
            "/predict": "Realizar predição (POST)",
            "/historico": "Últimos preços históricos (GET)",
            "/metricas": "Métricas do modelo (GET)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Verifica o status da API e se o modelo está carregado."""
    return HealthResponse(
        status="healthy",
        modelo_carregado=model is not None,
        symbol_treinado=config["symbol"] if config else "N/A",
        janela_temporal=config["janela"] if config else 0
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predição"])
async def predict(request: PredictionRequest):
    """
    Realiza predição de preço de fechamento.
    
    Utiliza os últimos N dias (janela temporal) de dados reais
    para predizer os próximos dias.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    if request.symbol != config["symbol"]:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo treinado para '{config['symbol']}'. "
                   f"Símbolo '{request.symbol}' não é suportado."
        )
    
    janela = config["janela"]
    
    # Opção 1: usuário forneceu os preços históricos manualmente
    if request.precos_historicos is not None:
        if len(request.precos_historicos) < janela:
            raise HTTPException(
                status_code=400,
                detail=f"Dados insuficientes. Necessário ao menos {janela} "
                       f"preços históricos, recebidos {len(request.precos_historicos)}."
            )
        precos_recentes = np.array(request.precos_historicos[-janela:], dtype=float)
        data_referencia = datetime.now().strftime("%Y-%m-%d")
        fonte_dados = "fornecido_pelo_usuario"
    else:
        # Opção 2: buscar dados recentes via yfinance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=janela * 3)  # Margem para dias não úteis

        try:
            df = yf.download(
                request.symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d")
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao buscar dados: {str(e)}")

        if df.empty or len(df) < janela:
            raise HTTPException(
                status_code=400,
                detail=f"Dados insuficientes. Necessário ao menos {janela} dias."
            )

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Pegar últimos 'janela' dias de fechamento
        precos_recentes = df["Close"].values[-janela:]
        data_referencia = df.index[-1].strftime("%Y-%m-%d")
        fonte_dados = "yahoo_finance"
    
    # Normalizar
    precos_normalizados = scaler.transform(precos_recentes.reshape(-1, 1))
    
    # Realizar predições iterativas para múltiplos dias
    predicoes = []
    sequencia_atual = precos_normalizados.copy()
    
    for dia in range(request.dias_futuros):
        # Preparar input para o modelo
        X_input = sequencia_atual[-janela:].reshape(1, janela, 1)
        X_tensor = torch.FloatTensor(X_input)
        
        # Predizer
        model.eval()
        with torch.no_grad():
            pred_normalizado = model(X_tensor).numpy()
        pred_valor = scaler.inverse_transform(pred_normalizado)[0][0]
        
        # Data estimada da predição (dias úteis)
        data_pred = pd.Timestamp(data_referencia) + pd.offsets.BDay(dia + 1)
        
        predicoes.append({
            "dia": dia + 1,
            "data_estimada": data_pred.strftime("%Y-%m-%d"),
            "preco_previsto": round(float(pred_valor), 2)
        })
        
        # Atualizar sequência com a nova predição
        sequencia_atual = np.append(sequencia_atual, pred_normalizado.reshape(-1, 1), axis=0)
    
    # Carregar métricas do modelo
    metricas_path = "models/metricas.json"
    modelo_info = {}
    if os.path.exists(metricas_path):
        with open(metricas_path, "r") as f:
            modelo_info = json.load(f)
    modelo_info["fonte_dados"] = fonte_dados
    modelo_info["dias_utilizados"] = janela

    return PredictionResponse(
        symbol=request.symbol,
        data_referencia=data_referencia,
        predicoes=predicoes,
        modelo_info=modelo_info
    )


@app.get("/historico", tags=["Dados"])
async def historico(dias: int = 30):
    """Retorna os últimos N dias de preços históricos."""
    if config is None:
        raise HTTPException(status_code=503, detail="Configuração não carregada")
    
    if dias < 1 or dias > 365:
        raise HTTPException(status_code=400, detail="Parâmetro 'dias' deve ser entre 1 e 365")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=dias * 2)
    
    try:
        df = yf.download(
            config["symbol"],
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar dados: {str(e)}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df_ultimos = df.tail(dias)
    
    historico_data = []
    for data, row in df_ultimos.iterrows():
        historico_data.append({
            "data": data.strftime("%Y-%m-%d"),
            "abertura": round(float(row["Open"]), 2),
            "maxima": round(float(row["High"]), 2),
            "minima": round(float(row["Low"]), 2),
            "fechamento": round(float(row["Close"]), 2),
            "volume": int(row["Volume"])
        })
    
    return {
        "symbol": config["symbol"],
        "dias_solicitados": dias,
        "registros_retornados": len(historico_data),
        "dados": historico_data
    }


@app.get("/metricas", tags=["Modelo"])
async def metricas():
    """Retorna as métricas de avaliação do modelo treinado."""
    metricas_path = "models/metricas.json"
    
    if not os.path.exists(metricas_path):
        raise HTTPException(
            status_code=404,
            detail="Métricas não encontradas. Execute train.py primeiro."
        )
    
    with open(metricas_path, "r") as f:
        metricas_data = json.load(f)
    
    return {
        "symbol": config["symbol"] if config else "N/A",
        "metricas": metricas_data,
        "descricao": {
            "MSE": "Mean Squared Error - erro quadrático médio",
            "RMSE": "Root Mean Squared Error - raiz do erro quadrático médio",
            "MAE": "Mean Absolute Error - erro absoluto médio",
            "R2": "Coeficiente de determinação (1.0 = perfeito)",
            "MAPE": "Mean Absolute Percentage Error (%)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
