"""Lähdekohtaisten ekstraktiotiedostojen yhdistäminen master-CSV:ksi.

MIKSI tämä moduuli on olemassa
------------------------------
H1:n ekstraktio etenee lähde kerrallaan: jokainen tutkimus saa oman
notebookin (``notebooks/01_corpus_extraction/``), joka tallentaa rivit
muotoon ``data/interim/{first_author}_{year}.csv``. Tämä tiedosto
yhdistää interim-tiedostot yhdeksi master-korpukseksi:

* Sarakejärjestys yhtenäistetään YAML-skeemaan.
* Skeemavalidointi ajetaan uudelleen yhdistetyllä DataFrame:lla.
* Duplikaatit (sama pari samasta lähteestä) raportoidaan, eivät hiljenny.

Yhdistämisen pitäminen erillisessä funktiossa (eikä notebookissa) tarkoittaa,
että koko korpus voidaan rakentaa uudelleen yhdellä komennolla CI:ssä tai
Makefilessa.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from coamorphous.corpus.schema import build_schema, column_order, load_schema_yaml

logger = logging.getLogger(__name__)


def _read_one_interim(path: Path) -> pd.DataFrame:
    """Lue yksi interim-CSV ja varmista, että se on olemassa ja ei-tyhjä."""
    if not path.is_file():
        raise FileNotFoundError(f"Interim-CSV ei löydy: {path}")
    df = pd.read_csv(path)
    logger.info("Luettu %d riviä tiedostosta %s", len(df), path.name)
    return df


def merge_interim(
    interim_paths: Iterable[Path],
    schema_yaml: Path,
    validate: bool = True,
) -> pd.DataFrame:
    """Yhdistä lähdekohtaiset interim-CSV:t yhdeksi master-DataFrame:ksi.

    Parameters
    ----------
    interim_paths : iterable of Path
        Polkuja tiedostoihin ``data/interim/*.csv``.
    schema_yaml : Path
        Polku ``configs/corpus_schema.yaml`` -tiedostoon.
    validate : bool, default True
        Jos True, ajetaan Pandera-validointi yhdistetylle DataFrame:lle.
        Voidaan asettaa False kehityksen aikana, mutta tuotannossa aina True.

    Returns
    -------
    pandas.DataFrame
        Master-korpus YAML-skeeman sarakejärjestyksessä.

    Raises
    ------
    pandera.errors.SchemaError
        Jos validointi on käytössä ja yhdistetty data ei läpäise sitä.
    """
    spec = load_schema_yaml(schema_yaml)
    cols = column_order(spec)

    frames: List[pd.DataFrame] = []
    for p in interim_paths:
        frames.append(_read_one_interim(Path(p)))

    if not frames:
        # Tyhjä lista on validi tila — palautetaan tyhjä DataFrame oikein
        # nimettyine sarakkeineen. MIKSI: helpottaa CI:n ja "ei vielä mitään
        # ekstraktoitu" -tilan käsittelyä.
        logger.warning("Ei interim-tiedostoja yhdistettäväksi; palautetaan tyhjä korpus.")
        return pd.DataFrame(columns=cols)

    merged = pd.concat(frames, ignore_index=True, sort=False)

    # Lisää puuttuvat sarakkeet NaN:llä, jotta kaikki YAML:n sarakkeet ovat
    # mukana, vaikka yksittäinen lähde ei niitä raportoinut.
    for col in cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    # Pakota sarakejärjestys YAML:n mukaiseksi.
    merged = merged[cols]

    # Duplikaattitarkistus pair_id:n perusteella. MIKSI: pair_id on uniikki
    # tunniste, ja duplikaatti viittaa joko ekstraktion virheeseen tai
    # samaan riviin kahdesta lähteestä, mikä vaatii manuaalisen päätöksen.
    duplicates = merged[merged.duplicated(subset=["pair_id"], keep=False)]
    if not duplicates.empty:
        logger.warning(
            "Löytyi %d duplikaattia pair_id:llä: %s",
            len(duplicates),
            duplicates["pair_id"].unique().tolist(),
        )

    if validate:
        schema = build_schema(schema_yaml)
        merged = schema.validate(merged, lazy=True)

    logger.info("Master-korpus rakennettu: %d riviä, %d saraketta", len(merged), len(cols))
    return merged


def write_empty_master(schema_yaml: Path, output_path: Path) -> None:
    """Kirjoita tyhjä master-CSV vain headereilla, järjestys YAML:sta.

    Parameters
    ----------
    schema_yaml : Path
        Skeema-YAML.
    output_path : Path
        Hakemistopolku, johon CSV kirjoitetaan
        (``data/processed/coamorphous_corpus_v1.csv``).

    Notes
    -----
    Tämän kutsuminen on H1:n alkuvaiheessa hyödyllistä: se konkretisoi
    sarakejärjestyksen ja antaa testattavissa olevan headeririvin, jota
    ekstraktion notebookit voivat tarkistaa.
    """
    spec = load_schema_yaml(schema_yaml)
    cols = column_order(spec)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=cols).to_csv(output_path, index=False)
    logger.info("Tyhjä master-CSV kirjoitettu: %s (%d saraketta)", output_path, len(cols))
