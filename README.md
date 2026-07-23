# SCNet

**SCNet: Explicit Same-Scale Cross-Channel Modeling for Multi-Scale Multivariate Time Series Forecasting**

This is the official implementation of SCNet, a multivariate time series forecasting framework that **explicitly** models same-scale cross-channel dependencies, in contrast to existing multi-scale decomposition–based approaches that capture cross-channel relationships only implicitly after scale fusion.

## Architecture

SCNet consists of four modules:

- **MAPE (Multi-Scale Adaptive Patch Embedding)** — performs multi-scale decomposition and constructs granularity-aligned representations: smaller patches at fine-grained scales, larger patches at coarse-grained scales.
- **CSAL (Cross-Channel Same-Scale Association Learner)** — explicitly captures inter-variable interactions *within each scale* via a constrained cross-channel mapping (Softplus + row-wise L1 normalization), providing attention-like channel aggregation without the quadratic cost of self-attention.
- **TDL (Temporal Dependency Learner)** — a lightweight feed-forward block (Linear → GELU → Linear with residual connection and LayerNorm) that captures intrinsic channel-wise temporal dynamics.
- **SGF (Scale Gate Fusion)** — adaptively integrates multi-scale features through a sigmoid gating mechanism, followed by a linear predictor.

Pipeline: `Input → RevIN → MAPE → (CSAL → TDL) × N → SGF → Linear Predictor → RevIN⁻¹ → Forecast`

## Getting Started

### 1. Environment

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

All eight benchmark datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, Exchange, Solar-Energy) are available from the well pre-processed archive provided by [Time-Series-Library](https://github.com/thuml/Time-Series-Library) ([Google Drive](https://drive.google.com/file/d/1C89grm0nkDBMZZem4pj6TI8UpKtz9o7p/view?usp=sharing)).

Place the downloaded data under `./dataset/`, e.g.:

```
dataset/
├── ETT-small/          # ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
├── electricity/
├── exchange_rate/
├── Solar/
├── traffic/
└── weather/
```

### 3. Train and Evaluate

Experiment scripts for all benchmarks are provided under `./scripts/SCNet/`:

```bash
bash scripts/SCNet/ETTh1.sh
bash scripts/SCNet/ETTm1.sh
bash scripts/SCNet/weather.sh
bash scripts/SCNet/traffic.sh
# ...
```

Across all datasets the look-back window is fixed to `L = 96` and prediction horizons are `H ∈ {96, 192, 336, 720}`.

Key model arguments:

| Argument | Description |
|---|---|
| `--model SCNet` | select SCNet |
| `--patch_sizes` | patch size p_s for each temporal scale (MAPE), e.g. `96 48 24 12 6` |
| `--d_model` | embedding dimension |
| `--d_ff` | hidden dimension of TDL / predictor |
| `--e_layers` | number of stacked (CSAL → TDL) blocks |

## Main Results

SCNet consistently achieves the best or second-best performance on eight real-world benchmarks against state-of-the-art baselines (AMD, xPatch, PatchMLP, PatchTST, SimpleTM), with MSE reductions of up to 18% on Traffic and 28% on Solar-Energy compared with AMD.

## Project Structure

```
├── run.py                        # entry point
├── models/SCNet.py               # SCNet model (MAPE / CSAL / TDL / SGF)
├── exp/                          # training & evaluation pipeline
├── layers/                       # embedding & RevIN layers
├── data_provider/                # dataset loaders
├── utils/                        # metrics, tools, etc.
└── scripts/SCNet/                # reproduction scripts for 8 benchmarks
```

## Acknowledgements

We appreciate the following repositories for their valuable code and datasets:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [iTransformer](https://github.com/thuml/iTransformer)
