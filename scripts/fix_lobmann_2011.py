"""Korjaa data/interim/lobmann_2011.csv vastaamaan uutta skeemaa.

MIKSI tämä skripti on olemassa
------------------------------
Löbmann et al. 2011 (DOI: 10.1021/mp2002973) -lähteen alkuperäisessä
ekstraktiossa oli kaksi virhettä:

1. **Mooliosuudet käännetty**: Table 1:n otsikko sanoo
   "naproxen (molar ratio) | indomethacin (molar ratio)", mutta sivun
   1920 tekstin punnitukset osoittavat että lähteen "2:1" tarkoittaa
   IND:NAP = 2:1, ei NAP:IND = 2:1. Massat ovat autoritäätti.

2. **Sensurointi väärin**: Lähteen kohta 3.2:n mukaan vain 2:1-rivit
   (4 °C ja 25 °C) ja 1:2-rivi 25 °C:ssa kiteytyivät. Muut pysyivät
   amorfisina 21 vrk:n koejakson loppuun -> ``censored=True``.

Lisäksi skeema on päivittynyt: kolme uutta saraketta
(``experimental_protocol``, ``protocol_max_duration_days``,
``storage_T_K``) on täytettävä, ja ``gfa_class``/``label_confidence``
lasketaan uudelleen kutsumalla ``classify_baird_taylor`` uusilla
parametreilla — emme kovakoodaa luokituksia, jotta lopputulos pysyy
yhtenäisenä luokituslogiikan kanssa.

Käyttö:
    python scripts/fix_lobmann_2011.py

Skripti on idempotentti: voit ajaa sen uudelleen turvallisesti.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Skripti ajetaan repo-juuresta; varmistetaan src-polku.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coamorphous.corpus.classify import classify_baird_taylor  # noqa: E402
from coamorphous.corpus.schema import build_schema, column_order, load_schema_yaml  # noqa: E402

CSV_PATH = PROJECT_ROOT / "data" / "interim" / "lobmann_2011.csv"
SCHEMA_YAML = PROJECT_ROOT / "configs" / "corpus_schema.yaml"

# -----------------------------------------------------------------------------
# Korjaussäännöt pair_id:n mukaan.
#
# Avaimet vastaavat lähteen mooliosuuksien uutta tulkintaa:
#   - "2:1" (lobmann2011_01, _02): IND:NAP = 2:1 -> IND-painoinen
#   - "1:1" (lobmann2011_03, _04): yhtä paljon kumpaakin
#   - "1:2" (lobmann2011_05, _06): NAP:IND = 1:2 mutta käytännössä massat
#     antavat NAP-osuudeksi 0.547 (lähteen punnitukset eivät vastaa tarkkaa
#     1:2-mooliosuutta, mutta käytetään raportoituja massoja autoritäätinä)
#
# Sensurointi:
#   - 2:1-rivit: kiteytyivät kummassakin lämpötilassa (Fig 6) -> False
#   - 1:1-rivit: pysyivät amorfisina 21 vrk -> True
#   - 1:2 4 °C:ssa: pysyi amorfisena -> True
#   - 1:2 25 °C:ssa: gamma-IND kiteytyi -> False
# -----------------------------------------------------------------------------
CORRECTIONS: dict[str, dict] = {
    "lobmann2011_01": {
        "ratio_label": "2:1",
        "mole_fraction_A": 0.333, "mole_fraction_B": 0.667,
        "weight_fraction_A": 0.243, "weight_fraction_B": 0.757,
        "induction_time_censored": False,
        "storage_T_C": 4.0, "storage_T_K": 277.15,
        "result_text": "Excess NAP recrystallized at day 21 (Fig 6, peak ~19.1 deg)",
    },
    "lobmann2011_02": {
        "ratio_label": "2:1",
        "mole_fraction_A": 0.333, "mole_fraction_B": 0.667,
        "weight_fraction_A": 0.243, "weight_fraction_B": 0.757,
        "induction_time_censored": False,
        "storage_T_C": 25.0, "storage_T_K": 298.15,
        "result_text": "Excess NAP recrystallized at day 21 (Fig 6, peaks ~19.0, 22.5, 27.4 deg)",
    },
    "lobmann2011_03": {
        "ratio_label": "1:1",
        "mole_fraction_A": 0.500, "mole_fraction_B": 0.500,
        "weight_fraction_A": 0.392, "weight_fraction_B": 0.608,
        "induction_time_censored": True,
        "storage_T_C": 4.0, "storage_T_K": 277.15,
        "result_text": "Amorphous halo at day 21 (no crystallization observed)",
    },
    "lobmann2011_04": {
        "ratio_label": "1:1",
        "mole_fraction_A": 0.500, "mole_fraction_B": 0.500,
        "weight_fraction_A": 0.392, "weight_fraction_B": 0.608,
        "induction_time_censored": True,
        "storage_T_C": 25.0, "storage_T_K": 298.15,
        "result_text": "Amorphous halo at day 21 (no crystallization observed)",
    },
    "lobmann2011_05": {
        "ratio_label": "1:2",
        "mole_fraction_A": 0.547, "mole_fraction_B": 0.453,
        "weight_fraction_A": 0.437, "weight_fraction_B": 0.563,
        "induction_time_censored": True,
        "storage_T_C": 4.0, "storage_T_K": 277.15,
        "result_text": "Amorphous halo at day 21 (no crystallization observed)",
    },
    "lobmann2011_06": {
        "ratio_label": "1:2",
        "mole_fraction_A": 0.547, "mole_fraction_B": 0.453,
        "weight_fraction_A": 0.437, "weight_fraction_B": 0.563,
        "induction_time_censored": False,
        "storage_T_C": 25.0, "storage_T_K": 298.15,
        "result_text": "Excess gamma-IND recrystallized at day 21 (Fig 6, peaks ~11.8, 17.2, 22.0, 26.7 deg)",
    },
}

# Yhteiset uudet kentät (vakioarvoja kaikille riveille).
COMMON_FIELDS = {
    "experimental_protocol": "dry_short_term",
    "protocol_max_duration_days": 21.0,
    "induction_time_days": 21.0,
}

# Odotetut luokitukset — käytetään säännöllisenä testinä, ei kovakoodattuna.
EXPECTED_CLASSIFICATIONS: dict[str, tuple[int | None, str]] = {
    "lobmann2011_01": (2, "low"),  # 2:1, censored=False, t=21 -> Class II low
    "lobmann2011_02": (2, "low"),
    "lobmann2011_03": (3, "low"),  # 1:1, censored=True, t=21 >= 14 -> Class III low
    "lobmann2011_04": (3, "low"),
    "lobmann2011_05": (3, "low"),
    "lobmann2011_06": (2, "low"),  # 1:2 25 C, censored=False, t=21 -> Class II low
}


def build_notes(row: dict, ratio_label: str, T_C: float, result_text: str) -> str:
    """Rakenna yhdenmukainen notes-merkintä riville.

    Tämä korvaa edellisen ekstraktion pitemmät, hieman ristiriitaiset
    suomenkieliset notes-tekstit englanninkielisellä, lyhyellä, faktoihin
    perustuvalla muodolla. Sisältää:

    * prosessin lyhennelmä (quench cooling, 441.15 K, 5 min)
    * säilytys (P2O5, RH ≈ 0 %, lämpötila, 21 vrk)
    * tulos (amorfinen vai kiteytyminen)
    * lähteen alkuperäinen mooliosuusmerkintä, jotta lukija voi varmistaa
      mooliosuuksien tulkinnan
    """
    return (
        f"Quench cooling 441.15 K, 5 min. Stored over P2O5 (RH~0%) at "
        f"{T_C:g} C for 21 days. {result_text}. "
        f"Original mole ratio nomenclature in source: '{ratio_label}' "
        f"(NAP:IND in source's table header but IND-rich in mass-based "
        f"composition for 2:1 and 1:2)."
    )


def gfa_label_from_class(gfa_class: int | None) -> str | None:
    """Muunna numeerinen GFA-luokka ihmisluettavaksi merkinnäksi."""
    if gfa_class is None:
        return None
    return f"Class {'I' * gfa_class}"


def main() -> int:
    print(f"[fix_lobmann_2011] Reading {CSV_PATH}")
    df_before = pd.read_csv(CSV_PATH)
    print(f"  -> {len(df_before)} rows, {len(df_before.columns)} columns")

    # Tallenna luokitus ennen muutoksia raportointia varten.
    before_classification = {
        row["pair_id"]: (row.get("gfa_class"), row.get("label_confidence"))
        for _, row in df_before.iterrows()
    }

    # Indeksoidaan pair_id:n mukaan helppoa päivitystä varten.
    df = df_before.set_index("pair_id", drop=False).copy()

    # Sovelletaan korjaukset ja täytetään uudet kentät rivi kerrallaan.
    for pair_id, fix in CORRECTIONS.items():
        if pair_id not in df.index:
            raise KeyError(f"Odottamaton: pair_id '{pair_id}' ei löydy CSV:stä.")

        # Mooliosuus- ja painokorjaukset.
        df.loc[pair_id, "mole_fraction_A"] = fix["mole_fraction_A"]
        df.loc[pair_id, "mole_fraction_B"] = fix["mole_fraction_B"]
        df.loc[pair_id, "weight_fraction_A"] = fix["weight_fraction_A"]
        df.loc[pair_id, "weight_fraction_B"] = fix["weight_fraction_B"]

        # Sensurointi-korjaus.
        df.loc[pair_id, "induction_time_censored"] = fix["induction_time_censored"]

        # Säilytysolojen vahvistus (lämpötila pysyy, K-johdannainen lisätään).
        df.loc[pair_id, "storage_T_C"] = fix["storage_T_C"]
        df.loc[pair_id, "storage_T_K"] = fix["storage_T_K"]

        # Yhteiset uudet kentät.
        for col, val in COMMON_FIELDS.items():
            df.loc[pair_id, col] = val

        # Päivitetty notes.
        df.loc[pair_id, "notes"] = build_notes(
            row=df.loc[pair_id].to_dict(),
            ratio_label=fix["ratio_label"],
            T_C=fix["storage_T_C"],
            result_text=fix["result_text"],
        )

        # Lasketaan luokitus uudestaan luokittimella.
        gfa, conf = classify_baird_taylor(
            induction_time_days=COMMON_FIELDS["induction_time_days"],
            storage_T_C=fix["storage_T_C"],
            storage_RH_percent=df.loc[pair_id, "storage_RH_percent"],
            crystallizes_on_dsc_cooling=False,
            induction_time_censored=fix["induction_time_censored"],
            experimental_protocol=COMMON_FIELDS["experimental_protocol"],
            protocol_max_duration_days=COMMON_FIELDS["protocol_max_duration_days"],
        )
        df.loc[pair_id, "gfa_class"] = gfa
        df.loc[pair_id, "gfa_class_label"] = gfa_label_from_class(gfa)
        df.loc[pair_id, "label_confidence"] = conf

    # Tarkista mole_fraction-summa jokaisella rivillä (toleranssi 0.001).
    df_check = df.copy()
    summa = df_check["mole_fraction_A"].astype(float) + df_check["mole_fraction_B"].astype(float)
    if not ((summa - 1.0).abs() < 0.001).all():
        bad = df_check.loc[(summa - 1.0).abs() >= 0.001, ["pair_id", "mole_fraction_A", "mole_fraction_B"]]
        raise ValueError(f"mole_fraction_A + B != 1.0 (±0.001) joillain riveillä:\n{bad}")
    print("[fix_lobmann_2011] mole_fraction-summat OK (±0.001)")

    # Verifioi, että luokitukset vastaavat odotuksia.
    print("[fix_lobmann_2011] Tarkistetaan luokitukset odotuksia vastaan:")
    mismatches: list[str] = []
    for pair_id, (exp_gfa, exp_conf) in EXPECTED_CLASSIFICATIONS.items():
        got_gfa = df.loc[pair_id, "gfa_class"]
        got_conf = df.loc[pair_id, "label_confidence"]
        # int-koersio: pandas voi tehdä gfa_class:sta float-NaN -> käsitellään.
        got_gfa_int = int(got_gfa) if pd.notna(got_gfa) else None
        match = (got_gfa_int == exp_gfa) and (got_conf == exp_conf)
        marker = "OK" if match else "MISMATCH"
        print(f"  {pair_id}: got=({got_gfa_int}, {got_conf}) "
              f"expected=({exp_gfa}, {exp_conf})  [{marker}]")
        if not match:
            mismatches.append(pair_id)
    if mismatches:
        raise AssertionError(f"Luokitusristiriidat: {mismatches}")

    # Pakota sarakejärjestys vastaamaan YAML:n virallista järjestystä.
    spec = load_schema_yaml(SCHEMA_YAML)
    expected_order = column_order(spec)
    df_ordered = df.reset_index(drop=True)[expected_order]

    # Pandera-validointi (strict-skeema; tuntematon sarake tai puute hylätään).
    schema = build_schema(SCHEMA_YAML)
    schema.validate(df_ordered, lazy=True)
    print(f"[fix_lobmann_2011] Pandera-validointi OK ({len(df_ordered)} riviä, "
          f"{len(df_ordered.columns)} saraketta)")

    # Tallenna takaisin. index=False koska pair_id on jo sarake.
    df_ordered.to_csv(CSV_PATH, index=False)
    print(f"[fix_lobmann_2011] Tallennettu {CSV_PATH}")

    # Yhteenveto.
    print("\n=== Yhteenveto: luokitus ennen ja jälkeen ===")
    print(f"{'pair_id':<18} {'before':<22} {'after':<22}")
    for pair_id in CORRECTIONS:
        before = before_classification.get(pair_id, ("?", "?"))
        after = (
            int(df.loc[pair_id, 'gfa_class']) if pd.notna(df.loc[pair_id, 'gfa_class']) else None,
            df.loc[pair_id, 'label_confidence'],
        )
        print(f"{pair_id:<18} {str(before):<22} {str(after):<22}")

    print("\n=== Mitä muutettiin (pip-style summary) ===")
    print("  ~ mole fractions: lobmann2011_01, _02 0.667/0.333 -> 0.333/0.667")
    print("                    lobmann2011_05, _06 0.333/0.667 -> 0.547/0.453")
    print("  ~ weight_fraction_A/B: täytetty massoista (oli aiemmin tyhjä)")
    print("  ~ induction_time_censored: _03, _04, _05 False -> True")
    print("  + experimental_protocol = dry_short_term (kaikille)")
    print("  + protocol_max_duration_days = 21.0 (kaikille)")
    print("  + storage_T_K = 277.15 (4 C) tai 298.15 (25 C)")
    print("  ~ gfa_class: kaikki rivit lasketaan uudelleen classify_baird_taylor:lla")
    print("              _03, _04, _05: Class II -> Class III (sensored >=14 vrk kuivassa)")
    print("  ~ notes: korvattu yhdenmukaisella englanninkielisellä template-merkinnällä")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
