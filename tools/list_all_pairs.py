"""
Listaa kaikki binäärikorpuksen parit kaikista CSV-tiedostoista.
Tukee sekä pilkku- (',') että puolipiste-erotinta (';').

Käyttö: python tools/list_all_pairs.py
"""
from pathlib import Path

import pandas as pd


def read_csv_smart(path: Path) -> pd.DataFrame:
    """Lue CSV automaattisesti tunnistaen erottimen (',', ';')."""
    # Lue ensimmäinen rivi paljastamaan erotin
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
    
    if ";" in first_line and first_line.count(";") > first_line.count(","):
        sep = ";"
    else:
        sep = ","
    
    return pd.read_csv(path, sep=sep, low_memory=False)


def main() -> int:
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")

    files = sorted(processed_dir.glob("*.csv")) + sorted(interim_dir.glob("*.csv"))

    if not files:
        print("VIRHE: Ei loytynyt CSV-tiedostoja kansioista data/processed/ tai data/interim/")
        return 1

    total_pairs = 0
    print(f"{'='*100}")
    print(f"BINAARIKORPUKSEN PARIT")
    print(f"{'='*100}")

    for f in files:
        try:
            df = read_csv_smart(f)
        except Exception as e:
            print(f"\n[VIRHE] {f.name}: {e}")
            continue
        
        status = "PROCESSED" if "processed" in str(f) else "INTERIM"
        print(f"\n[{status}] {f.name} ({len(df)} pairs, {len(df.columns)} columns)")
        print(f"{'-'*100}")

        cols = [
            "pair_id",
            "drug_A_name",
            "drug_B_name",
            "mole_fraction_A",
            "mole_fraction_B",
            "stability_week_bin",
            "stability_protocol_match",
        ]
        # Suodata vain saatavilla olevat sarakkeet
        available_cols = [c for c in cols if c in df.columns]
        print(df[available_cols].to_string(index=False))
        total_pairs += len(df)

    print(f"\n{'='*100}")
    print(f"YHTEENSA: {total_pairs} paria, {len(files)} lahdetta")
    print(f"{'='*100}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())