# Tech Challenge - Predição de Ações com LSTM

## Descrição
Modelo preditivo de redes neurais **Long Short-Term Memory (LSTM)** para predizer o valor de fechamento da ação da **Disney (DIS)**, com deploy em API REST via FastAPI.

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
│   ├── lstm_model.keras
│   ├── scaler.pkl
│   ├── config.json
│   └── metricas.json
└── data/                # (gerado) Dados e gráficos
    ├── DIS_historico.csv
    ├── treinamento_historico.png
    └── predicoes_vs_real.png
```

## Requisitos

- Python 3.10+
- TensorFlow 2.16+
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
1. Coleta dados da Disney (DIS) via Yahoo Finance (2018-2024)
2. Pré-processa e normaliza os dados
3. Cria sequências temporais (janela de 60 dias)
4. Treina modelo LSTM com 2 camadas (50 neurônios cada)
5. Avalia o modelo e gera métricas
6. Salva artefatos e gráficos

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
  -d '{"symbol": "DIS", "dias_futuros": 5}'
```

**Resposta:**
```json
{
  "symbol": "DIS",
  "data_referencia": "2024-07-19",
  "predicoes": [
    {"dia": 1, "data_estimada": "2024-07-22", "preco_previsto": 98.45},
    {"dia": 2, "data_estimada": "2024-07-23", "preco_previsto": 98.72},
    ...
  ],
  "modelo_info": {"RMSE": 3.21, "R2": 0.94, "MAPE": 2.8}
}
```

## Arquitetura do Modelo

```
Model: Sequential
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
LSTM (50 units)             (None, 60, 50)            10,400
Dropout (0.2)               (None, 60, 50)            0
LSTM (50 units)             (None, 50)                20,200
Dropout (0.2)               (None, 50)                0
Dense (1 unit)              (None, 1)                 51
=================================================================
Total params: 30,651
```

- **Input**: 60 dias de preços normalizados
- **Output**: preço de fechamento do dia seguinte
- **Otimizador**: Adam (lr=0.001)
- **Loss**: Mean Squared Error
- **Regularização**: Dropout 20% entre camadas
- **Early Stopping**: patience=10 épocas

## Tecnologias Utilizadas

- **Python** - Linguagem principal
- **TensorFlow/Keras** - Framework de Deep Learning
- **yfinance** - Coleta de dados financeiros
- **scikit-learn** - Normalização e métricas
- **FastAPI** - Framework para API REST
- **Pandas/NumPy** - Manipulação de dados
- **Matplotlib** - Visualizações
