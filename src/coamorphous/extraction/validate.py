"""Lisävalidoinnit ekstraktion master-CSV-riveille.

MIKSI tämä on Pandera-skeeman lisäksi
-------------------------------------
Pandera tarkistaa tyypit, enumit ja yksittäisten kenttien rajat
(esim. 0 <= mole_fraction <= 1). Mutta master-CSV vaatii myös
*riveittäin* johdonmukaisuutta:

* Mooliosuuksien (A + B) tulee summautua 1.0:ksi.
* Painoosuuksien (A + B) samoin.
* Kanonisten SMILES:ien tulee olla RDKit:n mielestä valideja.
* Säilytysolojen tulee olla yhteensopivia ilmoitetun protokollan kanssa
  (esim. ``ich_q1a_accelerated`` -> 40 °C / 75 % RH ± toleranssi).

Nämä ovat *liiketoimintasääntöjä*, joita Pandera-skeeman olisi vaikea
ilmaista deklaratiivisesti. Pidetään ne erillisessä moduulissa, jotta
notebook voi ajaa ne yhdellä kutsulla ja saada selkeän virhelistan.
"""

from __future__ import annotations

import logging
from typing import Optional

from coamorphous.corpus.canonicalize import validate_smiles

logger = logging.getLogger(__name__)

# Toleranssi mole/weight-summan tarkistukseen. ±0.01 on löysä mutta riittävä:
# pyöristysvirheet tuottavat tyypillisesti <= 0.005 eroja.
FRACTION_SUM_TOLERANCE: float = 0.01

# Säilytysoloalueet protokollalle. Tarkemmat rajat kuin classify.py:n
# borderline-vyöhykkeet, koska tässä validoinnissa ei ole kyse luokituksesta
# vaan ekstraktion johdonmukaisuudesta.
ICH_Q1A_ACCEL_T_RANGE: tuple[float, float] = (35.0, 45.0)
ICH_Q1A_ACCEL_RH_RANGE: tuple[float, float] = (70.0, 80.0)
ICH_Q1A_LONG_T_RANGE: tuple[float, float] = (20.0, 30.0)
ICH_Q1A_LONG_RH_RANGE: tuple[float, float] = (55.0, 65.0)


def _is_close_to_one(a: Optional[float], b: Optional[float]) -> bool:
    """Apuri: onko ``a + b`` riittävän lähellä 1.0:tä toleranssin sisällä?"""
    if a is None or b is None:
        return False
    return abs((a + b) - 1.0) <= FRACTION_SUM_TOLERANCE


def check_mole_fractions_sum(row: dict) -> Optional[str]:
    """Tarkista mole_fraction_A + mole_fraction_B = 1.0 ± 0.01.

    Returns
    -------
    str or None
        Virheviesti jos summa ei täsmää; ``None`` jos OK tai jos jompikumpi
        on ``None`` (puuttuva tieto on Panderan vastuulla, ei tämän).
    """
    a = row.get("mole_fraction_A")
    b = row.get("mole_fraction_B")
    if a is None or b is None:
        # Sallitaan: weight_fraction voi olla ainoa raportoitu suhde.
        return None
    if not _is_close_to_one(a, b):
        return (
            f"mole_fraction_A ({a}) + mole_fraction_B ({b}) = {a + b:.4f}, "
            f"odotetaan 1.0 ± {FRACTION_SUM_TOLERANCE}"
        )
    return None


def check_weight_fractions_sum(row: dict) -> Optional[str]:
    """Tarkista weight_fraction_A + weight_fraction_B = 1.0 ± 0.01."""
    a = row.get("weight_fraction_A")
    b = row.get("weight_fraction_B")
    if a is None or b is None:
        return None
    if not _is_close_to_one(a, b):
        return (
            f"weight_fraction_A ({a}) + weight_fraction_B ({b}) = {a + b:.4f}, "
            f"odotetaan 1.0 ± {FRACTION_SUM_TOLERANCE}"
        )
    return None


def check_smiles_validity(row: dict) -> list[str]:
    """Tarkista että kanoniset SMILES:t parsittavissa RDKit:llä.

    MIKSI: ekstraktion myöhemmissä vaiheissa (deskriptorit) RDKit-virheet
    ovat hankalia debugata, jos virheellinen SMILES on jo joukossa. Parempi
    havaita se heti.

    Returns
    -------
    list of str
        Virheviestien lista (tyhjä jos kaikki OK). Erillinen viesti per
        ongelmallinen SMILES, jotta lukija näkee tarkalleen mikä on vikaa.
    """
    errors: list[str] = []
    for letter in ("A", "B"):
        col = f"drug_{letter}_smiles_canonical"
        smiles = row.get(col)
        if smiles is None:
            # Tyhjä on validi tila, jos PubChem-haku epäonnistui — silloin
            # rivi on jo merkitty needs_review:ksi muulla logiikalla.
            continue
        if not validate_smiles(smiles):
            errors.append(f"{col}: virheellinen SMILES {smiles!r}")
    return errors


def check_storage_consistency(row: dict) -> Optional[str]:
    """Tarkista että storage_T_C ja storage_RH_percent vastaavat protokollaa.

    Sovellettavaksi vain ``ich_q1a_*``-protokolliin: kuiva- ja
    Tg+15 K -protokollia ei tarkisteta tässä, koska niillä on muut
    odotukset (esim. RH ≈ 0 % kuivassa).

    Returns
    -------
    str or None
        Virheviesti jos ristiriita; ``None`` jos OK tai protokolla ei
        kuulu tarkistettavien joukkoon.
    """
    protocol = row.get("experimental_protocol")
    T = row.get("storage_T_C")
    RH = row.get("storage_RH_percent")

    if protocol == "ich_q1a_accelerated":
        T_lo, T_hi = ICH_Q1A_ACCEL_T_RANGE
        RH_lo, RH_hi = ICH_Q1A_ACCEL_RH_RANGE
        label = "ICH Q1A accelerated (40/75)"
    elif protocol == "ich_q1a_long_term":
        T_lo, T_hi = ICH_Q1A_LONG_T_RANGE
        RH_lo, RH_hi = ICH_Q1A_LONG_RH_RANGE
        label = "ICH Q1A long-term (25/60)"
    else:
        return None

    if T is None or RH is None:
        return (
            f"experimental_protocol={protocol} edellyttää storage_T_C ja "
            f"storage_RH_percent -arvot, mutta saatiin T={T}, RH={RH}."
        )

    if not (T_lo <= T <= T_hi):
        return (
            f"{label}: storage_T_C={T} ei välillä [{T_lo}, {T_hi}]"
        )
    if not (RH_lo <= RH <= RH_hi):
        return (
            f"{label}: storage_RH_percent={RH} ei välillä [{RH_lo}, {RH_hi}]"
        )
    return None


def run_all_validations(row: dict) -> list[str]:
    """Aja kaikki riviä-kohti -validoinnit ja palauta virheiden lista.

    Lista on tyhjä, jos rivi on OK. Lukijalle helppo tarkistus:

    >>> errors = run_all_validations(row)
    >>> if errors: print("VIRHEET:", errors)
    """
    errors: list[str] = []

    # Yksittäisten viestien tarkastukset.
    for check in (
        check_mole_fractions_sum,
        check_weight_fractions_sum,
        check_storage_consistency,
    ):
        msg = check(row)
        if msg is not None:
            errors.append(msg)

    # Listapohjainen (voi tuottaa monta viestiä per kutsu).
    errors.extend(check_smiles_validity(row))

    return errors
