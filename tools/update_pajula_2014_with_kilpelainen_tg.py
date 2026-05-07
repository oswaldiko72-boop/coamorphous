"""
Päivittää Pajula 2014 -ekstraktion 3 riviä Kilpeläinen 2020:n
DSC 10 K/min Tg-arvoilla TMDSC-arvojen sijaan.

Lähde: Kilpeläinen et al. 2020, Eur. J. Pharm. Biopharm. 155, 49-54.
DOI: 10.1016/j.ejpb.2020.08.007. Taulukko 2 (DSC 10 K/min, n=3).

Päivitettävät parit:
  - terfenadine + paracetamol  (Tg = 53.0 ± 0.3 °C = 326.15 K)
  - terfenadine + indomethacine (Tg = 80.5 ± 1.0 °C = 353.65 K)
  - indomethacine + paracetamol (Tg = 33.4 ± 0.3 °C = 306.55 K)

Muutokset päivitetyille riveille:
  1. Tg_K (TMDSC) -> Tg_K (DSC 10 K/min)
  2. Tg_heating_rate_K_min: 2.0 -> 10.0
  3. Tg_uncertainty_K: null -> SD-arvo
  4. Poista TMDSC-syy review_reasons-listalta
  5. Lisää uusi review_reasons-merkintä Kilpeläinen 2020 -lähteestä

Käyttö: python tools/update_pajula_2014_with_kilpelainen_tg.py

Idempotentti: tarkistaa onko päivitys jo tehty.
"""
import json
import sys
from pathlib import Path

RAW_JSON_PATH = Path("data/interim/pajula_2014_raw.json")

# TMDSC-syy joka POISTETAAN päivitetyiltä riveiltä
TMDSC_REVIEW_REASON = (
    "Pair-Tg from Table 3 (TMDSC TOPEM, 2 K/min underlying), not standard "
    "DSC 10 K/min. Pure component Tg values in Table 2 are at 10 K/min but "
    "not used for pair-Tg field per schema. Pajula 2014 is the only paper "
    "in corpus with TMDSC-only pair-Tg measurement."
)

# UUSI syy joka LISÄTÄÄN päivitetyille riveille
KILPELAINEN_UPDATE_REASON = (
    "Tg updated from Kilpelainen et al. 2020 (DOI: 10.1016/j.ejpb.2020.08.007), "
    "Table 2, DSC 10 K/min, n=3, replacing Pajula 2014 TMDSC value. "
    "Three pairs in Pajula 2014 (terfenadine-paracetamol, terfenadine-indomethacin, "
    "indomethacin-paracetamol) were re-measured by Kilpelainen 2020 with standard "
    "DSC 10 K/min, enabling cross-corpus consistency."
)

# Päivitysarvot per pari (drug_A_name_raw, drug_B_name_raw) -> (Tg_K, Tg_uncertainty_K)
# Kelvin: T_K = T_C + 273.15
KILPELAINEN_TG_VALUES = {
    ("terfenadine", "paracetamol"): {
        "Tg_K": 326.15,           # 53.0 + 273.15
        "Tg_uncertainty_K": 0.3,
        "Tg_C_source": 53.0,
    },
    ("terfenadine", "indomethacine"): {
        "Tg_K": 353.65,           # 80.5 + 273.15
        "Tg_uncertainty_K": 1.0,
        "Tg_C_source": 80.5,
    },
    ("indomethacine", "paracetamol"): {
        "Tg_K": 306.55,           # 33.4 + 273.15
        "Tg_uncertainty_K": 0.3,
        "Tg_C_source": 33.4,
    },
}


def main() -> int:
    if not RAW_JSON_PATH.exists():
        print(f"VIRHE: Tiedostoa ei loydy: {RAW_JSON_PATH}")
        return 1

    with RAW_JSON_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        print(f"VIRHE: Odotettiin listaa, saatiin {type(rows).__name__}")
        return 1

    print(f"Luettu {len(rows)} rivia tiedostosta {RAW_JSON_PATH}")

    updated_count = 0
    skipped_count = 0

    for i, row in enumerate(rows):
        pair_key = (
            row.get("drug_A_name_raw", "").strip(),
            row.get("drug_B_name_raw", "").strip(),
        )

        if pair_key not in KILPELAINEN_TG_VALUES:
            continue  # Ei paivitettava pari

        update = KILPELAINEN_TG_VALUES[pair_key]
        pair_name = f"{pair_key[0]} + {pair_key[1]}"

        # Idempotency-tarkistus: onko jo paivitetty?
        if (
            row.get("Tg_heating_rate_K_min") == 10.0
            and abs((row.get("Tg_K") or 0) - update["Tg_K"]) < 0.01
        ):
            print(f"  [{i + 1}/{len(rows)}] {pair_name}: jo paivitetty (skip)")
            skipped_count += 1
            continue

        # 1) Paivita Tg-arvot
        old_tg = row.get("Tg_K")
        old_rate = row.get("Tg_heating_rate_K_min")
        row["Tg_K"] = update["Tg_K"]
        row["Tg_heating_rate_K_min"] = 10.0
        row["Tg_uncertainty_K"] = update["Tg_uncertainty_K"]

        # 2) Poista TMDSC-syy review_reasons-listalta
        existing_reasons = row.get("review_reasons", []) or []
        removed_tmdsc = False
        if TMDSC_REVIEW_REASON in existing_reasons:
            existing_reasons.remove(TMDSC_REVIEW_REASON)
            removed_tmdsc = True

        # 3) Lisaa Kilpelainen-syy (jos ei jo siella)
        added_kilpelainen = False
        if KILPELAINEN_UPDATE_REASON not in existing_reasons:
            existing_reasons.append(KILPELAINEN_UPDATE_REASON)
            added_kilpelainen = True

        row["review_reasons"] = existing_reasons

        # Lokitus
        actions = []
        actions.append(f"Tg_K {old_tg:.2f}->{update['Tg_K']:.2f}")
        actions.append(f"rate {old_rate}->10.0")
        actions.append(f"uncertainty=null->{update['Tg_uncertainty_K']}")
        if removed_tmdsc:
            actions.append("-TMDSC")
        if added_kilpelainen:
            actions.append("+Kilpelainen")

        n_reasons = len(row["review_reasons"])
        print(
            f"  [{i + 1}/{len(rows)}] {pair_name}: "
            f"{', '.join(actions)} (review_reasons={n_reasons})"
        )
        updated_count += 1

    if updated_count == 0:
        if skipped_count > 0:
            print(f"\nINFO: Kaikki {skipped_count} riviä on jo päivitetty.")
        else:
            print("\nVIRHE: Ei loytynyt yhtaan paivitettavaa paria!")
            print("Tarkista raw.json:in drug_A/drug_B nimet.")
            return 1
        return 0

    # Tallenna takaisin
    with RAW_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print()
    print(f"OK: Tallennettu {RAW_JSON_PATH}")
    print(f"  - Paivitetty: {updated_count} rivia")
    print(f"  - Skipattu (jo paivitetty): {skipped_count} rivia")
    print(f"  - Ei muutettu (ei Kilpelainen 2020:ssa): {len(rows) - updated_count - skipped_count} rivia")
    print()
    print("Seuraava vaihe: aja notebookissa Vaihe 1.5-5 uudelleen (Solut 5, 7, 9, 11, 13).")
    print("Odotus: 3 riviä DSC 10 K/min, 5 riviä TMDSC 2 K/min (sekamuotoinen).")
    return 0


if __name__ == "__main__":
    sys.exit(main())