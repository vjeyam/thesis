from dataclasses import dataclass

@dataclass
class Config:
    data_path: str = "src/data/US/sp500/baseline_data_sp500.npy"
    ticker_path: str = "src/data/US/sp500/sp500_ticker.csv"
    lookback: int = 20
    lookahead: int = 5
    top_k: int = 10
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    sector_embedding_dim: int = 8
    batch_size: int = 512
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 4
    seed: int = 52
