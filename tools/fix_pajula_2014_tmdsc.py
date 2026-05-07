"""
Korjaa Pajula 2014 -ekstraktion tieteelliset epätarkkuudet:

1. Tg_heating_rate_K_min: 10.0 -> 2.0
   Syy: Pari-Tg on mitattu TMDSC TOPEM:lla (underlying rate 2 K/min) per
   osio 2.3.2, ei standardilla DSC 10 K/min.

2. crystallizes_on_dsc_cooling: false -> null
   crystallizes_on_dsc_reheating: false -> null
   Syy: Pajula 2014 ei tehnyt täyttä DSC H/C/H -sykliä Class-luokitukseen.
   Toinen lämmitys päättyy storage temperature:en (ei sulamispisteen yläpuolelle).
   "false" tarkoittaa "mittaus tehtiin, kiteytymistä ei havaittu" — Pajula 2014:lle
   oikea arvo on null = "mittausta ei tehty Class-luokitukseen".

3. needs_review = True kaikille 8 riville
   review_reasons: TMDSC + GFA-syyt lisätty

Käyttö: python tools/fix_pajula_2014_tmdsc.py

Idempotentti: tarkistaa onko korjaus jo tehty, exitoituu turvallisesti jos on.
"""
import json
import sys
from pathlib import Path

RAW_JSON_PATH = Path("data/interim/pajula_2014_raw.json")

TMDSC_REVIEW_REASON = (
    "Pair-Tg from Table 3 (TMDSC TOPEM, 2 K/min underlying), not standard "
    "DSC 10 K/min. Pure component Tg values in Table 2 are at 10 K/min but "
    "not used for pair-Tg field per schema. Pajula 2014 is the only paper "
    "in corpus with TMDSC-only pair-Tg measurement."
)

GFA_NULL_REVIEW_REASON = (
    "DSC H/C/H cycle for GFA classification was NOT performed. Second heating "
    "ended at storage temperature, not above Tm. Therefore "
    "crystallizes_on_dsc_cooling and crystallizes_on_dsc_reheating set to null "
    "(measurement not performed for Class 1/2/3 assessment), and gfa_class_dsc "
    "should remain null. Pajula 2014 used DSC only for sample preparation, "
    "not for Baird et al. 2010 GFA classification."
)


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

    # Idempotency-tarkistus: kaikki 3 korjausta tehty?
    all_fixed = all(
        row.get("Tg_heating_rate_K_min") == 2.0
        and row.get("crystallizes_on_dsc_cooling") is None
        and row.get("crystallizes_on_dsc_reheating") is None
        for row in rows
    )
    if all_fixed:
        print(f"INFO: Kaikki {len(rows)} rivia on jo korjattu (skip).")
        return 0

    tg_rate_fixed = 0
    dsc_cycle_fixed = 0
    review_added_tmdsc = 0
    review_added_gfa = 0

    for i, row in enumerate(rows):
        changed_fields = []

        # 1) Tg_heating_rate_K_min: 10.0 -> 2.0
        if row.get("Tg_heating_rate_K_min") == 10.0:
            row["Tg_heating_rate_K_min"] = 2.0
            tg_rate_fixed += 1
            changed_fields.append("Tg_rate=2.0")

        # 2a) crystallizes_on_dsc_cooling: false -> null
        if row.get("crystallizes_on_dsc_cooling") is False:
            row["crystallizes_on_dsc_cooling"] = None
            dsc_cycle_fixed += 1
            changed_fields.append("dsc_cooling=null")

        # 2b) crystallizes_on_dsc_reheating: false -> null
        if row.get("crystallizes_on_dsc_reheating") is False:
            row["crystallizes_on_dsc_reheating"] = None
            changed_fields.append("dsc_reheating=null")

        # 3) needs_review = True
        if not row.get("needs_review", False):
            row["needs_review"] = True
            changed_fields.append("needs_review=True")

        # 4a) Lisaa TMDSC-syy (jos ei jo)
        existing_reasons = row.get("review_reasons", []) or []
        if TMDSC_REVIEW_REASON not in existing_reasons:
            existing_reasons.append(TMDSC_REVIEW_REASON)
            review_added_tmdsc += 1
            changed_fields.append("+TMDSC_reason")

        # 4b) Lisaa GFA-null-syy (jos ei jo)
        if GFA_NULL_REVIEW_REASON not in existing_reasons:
            existing_reasons.append(GFA_NULL_REVIEW_REASON)
            review_added_gfa += 1
            changed_fields.append("+GFA_null_reason")

        row["review_reasons"] = existing_reasons

        if changed_fields:
            pair_name = (
                f"{row.get('drug_A_name_raw', '?')} + "
                f"{row.get('drug_B_name_raw', '?')}"
            )
            n_reasons = len(row["review_reasons"])
            changes_str = ", ".join(changed_fields)
            print(
                f"  [{i + 1}/{len(rows)}] {pair_name}: {changes_str} "
                f"(review_reasons={n_reasons})"
            )

    if (
        tg_rate_fixed == 0
        and dsc_cycle_fixed == 0
        and review_added_tmdsc == 0
        and review_added_gfa == 0
    ):
        print("INFO: Ei muutoksia tarvittu.")
        return 0

    with RAW_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print()
    print(f"OK: Tallennettu {RAW_JSON_PATH}")
    print(f"  - Tg_heating_rate_K_min korjattu 10.0 -> 2.0:    {tg_rate_fixed} rivia")
    print(f"  - crystallizes_on_dsc_cooling false -> null:     {dsc_cycle_fixed} rivia")
    print(f"  - crystallizes_on_dsc_reheating false -> null:   {dsc_cycle_fixed} rivia")
    print(f"  - TMDSC-review_reason lisatty:                    {review_added_tmdsc} rivia")
    print(f"  - GFA-null-review_reason lisatty:                 {review_added_gfa} rivia")
    print()
    print("Seuraava vaihe: aja notebookissa Vaihe 2-5 uudelleen (Solut 5, 7, 9, 11, 13).")
    print("Odotus: gfa_class_dsc=null x8, gfa_label_confidence=not_reported tai vastaava.")
    return 0


if __name__ == "__main__":
    sys.exit(main())