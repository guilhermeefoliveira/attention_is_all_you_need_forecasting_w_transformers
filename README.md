# Attention is All You Need - Time Series Forecasting with Transformers

A from-scratch implementation of a Transformer-based model for univariate time series forecasting using PyTorch. The project validates the architecture on synthetic data and then applies it to real-world daily gold prices.

## Motivation

The Transformer architecture, introduced in [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762), revolutionized NLP through its self-attention mechanism. This project explores how the same core ideas — positional encoding, multi-head attention, and encoder stacking — can be applied to time series forecasting, where capturing long-range temporal dependencies is essential.

## Project structure

```
├── AttentionIsAllYouNeed_forecasting_w_transformers.ipynb   # Main notebook
├── Gold.csv                                                  # Daily gold prices dataset
├── README.md
├── pyproject.toml                                            # Project metadata and dependencies
├── uv.lock                                                   # Locked dependency versions
└── .python-version                                           # Python version pinned for uv
```

The notebook is organized into six chapters:

| Chapter | Description |
|---------|-------------|
| **I — Data preparation** | Synthetic time series generation (3 sinusoidal components + noise), train/test split, MinMaxScaler normalization, sliding window sequencing |
| **II — Model architecture** | `PositionalEncoding` and `TimeSeriesTransformer` classes built from scratch using `nn.Module` |
| **III — Training** | Training loop with MSE loss, Adam optimizer, and progress tracking |
| **IV — Evaluation** | Test set metrics (MSE, RMSE, MAE) and prediction visualization |
| **V — Deploy** | Autoregressive multi-step forecasting function |
| **VI — Real-world data** | Full pipeline applied to daily gold prices (1995–2016) |

## Model architecture

```
Input (batch, 70, 1)
  → Linear Encoder (1 → 32) × √d_model
  → Positional Encoding (sinusoidal)
  → TransformerEncoder (2 layers × 2 heads)
  → Mean pooling across time
  → Linear Decoder (32 → 1)
Output: predicted next value
```

Key design choices:

- **d_model = 32**: sufficient for the complexity of univariate series; keeps training fast
- **nhead = 2**: each head can specialize in different temporal patterns (short vs. long range)
- **num_layers = 2**: first layer captures direct point-to-point relationships; second layer captures patterns-of-patterns
- **Mean pooling** (vs. last token): more stable aggregation that uses information from the entire window
- **Sliding window = 70**: captures approximately 71% of the longest period in the synthetic data (period = 98)

## Key concepts demonstrated

- **Self-Attention**: each timestep computes Query, Key, and Value projections, then attends to all other timesteps via scaled dot-product attention
- **Positional Encoding**: deterministic sinusoidal encoding injected into embeddings so the model can distinguish temporal positions
- **Data leakage prevention**: `MinMaxScaler` fitted exclusively on training data
- **Autoregressive forecasting**: the model feeds its own predictions back as input to generate multi-step forecasts

## Results

### Synthetic data
The model successfully learns the composite sinusoidal pattern, achieving low MSE on the test set and producing coherent future forecasts that maintain the periodicity of the original signal.

### Gold prices
Applied to ~5,500 daily gold prices, the model captures the general trend and short-term dynamics. Forecast quality degrades over longer horizons due to error accumulation in the autoregressive loop — a known limitation of this approach.

## How to run

### Requirements

Managed via `uv` and `pyproject.toml`. Core dependencies:

```
python >= 3.10
torch
numpy
pandas
matplotlib
scikit-learn
tqdm
```

### Setup

```bash
# Clone the repository
git clone https://github.com/guilhermeefoliveira/<repo-name>.git
cd <repo-name>

# Install dependencies and create an environment with uv
uv sync

# Run the notebook
uv run jupyter notebook AttentionIsAllYouNeed_forecasting_w_transformers.ipynb
```

> **Note**: This project uses [uv](https://docs.astral.sh/uv/) for dependency management. All dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Training runs on CPU in ~3 minutes for the synthetic data and ~16 minutes for gold prices. GPU (CUDA/MPS) is automatically detected and used if available.

## Dataset

**Gold.csv** — Daily gold fixing price (10:30 AM London time) in USD per Troy ounce, from the London Bullion Market. Period: January 3, 1995 to November 10, 2016.

Source: [TTU Time Series Data Repository](https://www.math.ttu.edu/~atrindad/tsdata/)

## References

- Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *NeurIPS*.
- PyTorch documentation: [nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

## Author

**Gui Freire Oliveira**
