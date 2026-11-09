from .common.data import load_market_data
from .common.phases import build_paper_phases

def main():
    m = load_market_data(
        "src/data/US/sp500/baseline_data_sp500.npy",
        "src/data/US/sp500/sp500_ticker.csv",
    )
    first = m.frames[m.symbols[0]]
    print("stocks:", len(m.symbols))
    print("sectors:", len(m.sector_to_id))
    print("date range:", first.index.min(), "->", first.index.max())
    print("shape:", first.shape)
    print("columns:", list(first.columns))
    print("\nphases:")
    for p in build_paper_phases():
        print(p)

if __name__ == "__main__":
    main()
