# Tech Challenge - Predição de Ações com LSTM

## Descrição
Modelo de deep learning **LSTM (Long Short-Term Memory)** para prever o preço de fechamento das ações da **Petrobras (PETR4.SA)**, com deploy em API REST via FastAPI.

## Estrutura do Projeto

```
mlet_4/
├── data_pipeline.py     # Coleta e pré-processamento dos dados
├── modelo_lstm.py       # Arquitetura e treinamento do modelo LSTM
├── train.py             # Script principal de treinamento
├── api.py               # API FastAPI para predições
├── requirements.txt     # Dependências do projeto
├── .gitignore
├── README.md
├── models/              # (gerado) Artefatos do modelo
│   ├── lstm_model.pth
│   ├── scaler.pkl
│   ├── config.json
│   └── metricas.json
└── data/                # (gerado) Dados e gráficos
    ├── PETR4.SA_historico.csv
    ├── treinamento_historico.png
    └── predicoes_vs_real.png
```

## Requisitos

- Python 3.10+
- PyTorch
- Conexão com internet (para coleta de dados via Yahoo Finance)

## Instalação

```bash
# Criar e ativar ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
```

## Execução

### 1. Treinar o Modelo

```bash
python train.py
```

Este comando executa toda a pipeline:
1. Coleta dados da Petrobras (PETR4.SA) via Yahoo Finance (2018-2024)
2. Pré-processa e normaliza os dados com MinMaxScaler
3. Cria sequências temporais (janela de 60 dias)
4. Treina modelo LSTM com 2 camadas (50 neurônios) e early stopping
5. Avalia o modelo com métricas MAE, RMSE, MAPE e R²
6. Salva modelo treinado e gráficos

### 2. Iniciar a API

```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

A API estará disponível em: http://localhost:8000

Documentação interativa (Swagger): http://localhost:8000/docs

## Endpoints da API

| Método | Endpoint     | Descrição                              |
|--------|-------------|----------------------------------------|
| GET    | `/`         | Informações gerais da API              |
| GET    | `/health`   | Status e informações do modelo         |
| POST   | `/predict`  | Realizar predição de preço             |
| GET    | `/historico`| Últimos N dias de preços históricos    |
| GET    | `/metricas` | Métricas de avaliação do modelo        |

### Exemplo de Predição

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "PETR4.SA", "dias_futuros": 5}'
```

**Resposta:**
```json
{
  "symbol": "PETR4.SA",
  "data_referencia": "2024-12-30",
  "predicoes": [
    {"dia": 1, "data_estimada": "2024-12-31", "preco_previsto": 36.85},
    {"dia": 2, "data_estimada": "2025-01-02", "preco_previsto": 36.92}
  ],
  "modelo_info": {"MAE": 0.46, "RMSE": 0.60, "MAPE": 1.63, "R2": 0.95}
}
```

## Arquitetura do Modelo

```
ModeloLSTM(
  (lstm): LSTM(1, 50, num_layers=2, batch_first=True, dropout=0.2)
  (fc): Linear(in_features=50, out_features=1, bias=True)
)
Total params: 31,051
```

- **Input**: 60 dias de preços normalizados
- **Output**: preço de fechamento do dia seguinte
- **Otimizador**: Adam (lr=0.001)
- **Loss**: Mean Squared Error
- **Regularização**: Dropout 20% entre camadas LSTM
- **Early Stopping**: patience=10 épocas

## Tecnologias Utilizadas

- **Python** - Linguagem principal
- **PyTorch** - Framework de Deep Learning
- **yfinance** - Coleta de dados financeiros
- **scikit-learn** - Normalização e métricas
- **FastAPI** - Framework para API REST
- **Pandas/NumPy** - Manipulação de dados
- **Matplotlib** - Visualizações
