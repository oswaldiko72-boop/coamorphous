"""Pandera-skeeman dynaaminen rakennus YAML-määrittelystä.

MIKSI tämä moduuli on olemassa
------------------------------
Master-CSV:n sarakkeet kuvataan ihmisluettavassa muodossa
``configs/corpus_schema.yaml`` -tiedostossa. Sen sijaan, että ylläpidettäisiin
toista, koodissa kovakoodattua skeemaa, generoimme Pandera-skeeman dynaamisesti
YAML:sta. Näin:

* Dokumentaatio (YAML) ja validointi (Pandera) eivät voi mennä epäsynkroniin.
* Sarakejärjestys CSV:ssä on YAML:n järjestys — yksi totuuden lähde.
* Uuden sarakkeen lisääminen vaatii vain YAML-muutoksen ja testin.

Pandera-skeema (DataFrameSchema) on strict: tuntematon sarake aiheuttaa
validointivirheen. Tämä on tahallista — emme halua hiljaa kadottaa
ekstraktion sivutuotteena syntyneitä ylimääräisiä sarakkeita.

Tieteellinen konteksti
----------------------
Skeemavalidointi on harvoin "jännittävää" tutkimustyössä, mutta se on
ainoa mekanismi, joka takaa että H2/H3-vaiheessa luettu CSV vastaa sitä
muotoa, jota H1:n koodi on tuottanut. Se on myös ensimmäinen vaihe, jossa
huomataan inhimilliset kirjoitusvirheet (esim. ``"Class 2"`` vs ``2``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pandera.pandas as pa
import yaml

logger = logging.getLogger(__name__)


# Kartta YAML:n dtype-merkkijonoista Pandera-tyyppeihin.
# MIKSI eksplisiittinen kartta: Pandera tukee monia kirjoitustapoja
# (esim. "string", str, "object"), ja haluamme yhden virallisen mapping-tavan
# joka helpottaa skeeman lukemista ihmiselle.
_DTYPE_MAP: Dict[str, Any] = {
    "string": pa.String,
    "int": pa.Int64,
    "float": pa.Float64,
    "bool": pa.Bool,
}


def load_schema_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Lue korpuksen skeema YAML-tiedostosta.

    Parameters
    ----------
    yaml_path : pathlib.Path
        Polku ``configs/corpus_schema.yaml`` -tiedostoon.

    Returns
    -------
    dict
        YAML:n sisältö Python-rakenteena. Pääavaimet ``columns`` ja ``enums``.

    Raises
    ------
    FileNotFoundError
        Jos polku ei viittaa olemassa olevaan tiedostoon.
    yaml.YAMLError
        Jos YAML on syntaktisesti virheellinen.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Skeema-YAML ei löydy: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        spec: Dict[str, Any] = yaml.safe_load(f)

    if "columns" not in spec:
        raise ValueError(
            f"Skeema-YAML {yaml_path} ei sisällä pakollista 'columns' avainta."
        )
    return spec


def column_order(spec: Dict[str, Any]) -> List[str]:
    """Palauta CSV-sarakkeiden virallinen järjestys YAML-spesifikaation mukaan.

    MIKSI: kun kirjoitamme tyhjän master-CSV:n tai validoimme uutta dataa,
    sarakejärjestyksen pitää olla deterministinen ja sama joka ajossa.
    """
    return list(spec["columns"].keys())


def _check_in_enum(allowed: List[Any]) -> pa.Check:
    """Pandera-check, joka rajaa arvot annettuun listaan.

    Käytetään enum-sarakkeisiin (esim. ``process_method``).
    """
    allowed_set = set(allowed)

    # element_wise=True koska tarkistus tehdään yhdelle solulle kerrallaan.
    return pa.Check(
        lambda x: (x is None) or (pd.isna(x)) or (x in allowed_set),
        element_wise=True,
        error=f"arvo ei ole sallittujen joukossa: {allowed}",
    )


def _build_column(col_name: str, col_spec: Dict[str, Any], enums: Dict[str, List[Any]]) -> pa.Column:
    """Rakenna yksi Pandera-sarake YAML-määrittelystä.

    Parameters
    ----------
    col_name : str
        Sarakkeen nimi (käytetään virheviesteissä).
    col_spec : dict
        YAML:n alipuu yhdelle sarakkeelle (avaimet: dtype, nullable, ge, le,
        unique, enum, description).
    enums : dict
        Kaikki nimetyt enumit YAML:sta (avain → sallittujen arvojen lista).
    """
    dtype_key = col_spec["dtype"]
    if dtype_key not in _DTYPE_MAP:
        raise ValueError(
            f"Sarakkeen '{col_name}' dtype '{dtype_key}' ei ole tuettu. "
            f"Sallitut: {list(_DTYPE_MAP.keys())}"
        )

    checks: List[pa.Check] = []

    # Numeeristen rajojen (ge=greater-or-equal, le=less-or-equal) tarkistukset.
    # MIKSI: estää ilmiselvät virheet kuten negatiivinen induktioaika tai
    # mole_fraction > 1.0.
    if "ge" in col_spec:
        checks.append(pa.Check.ge(col_spec["ge"]))
    if "le" in col_spec:
        checks.append(pa.Check.le(col_spec["le"]))

    # Enum-tarkistus, jos sarake viittaa nimettyyn enumiin.
    if "enum" in col_spec:
        enum_name = col_spec["enum"]
        if enum_name not in enums:
            raise ValueError(
                f"Sarake '{col_name}' viittaa enumiin '{enum_name}', "
                f"jota ei ole määritelty YAML:n 'enums' osiossa."
            )
        checks.append(_check_in_enum(enums[enum_name]))

    return pa.Column(
        dtype=_DTYPE_MAP[dtype_key],
        checks=checks if checks else None,
        nullable=col_spec.get("nullable", True),
        unique=col_spec.get("unique", False),
        required=True,
        description=col_spec.get("description", ""),
        # coerce=True yrittää muuntaa arvot oikeaan tyyppiin (esim. "1" -> 1).
        # Tämä on hyödyllistä CSV-lukemisen jälkeen, jossa kaikki on alkuun str.
        coerce=True,
    )


def build_schema(yaml_path: Path) -> pa.DataFrameSchema:
    """Rakenna Pandera DataFrameSchema YAML-määrittelystä.

    Parameters
    ----------
    yaml_path : pathlib.Path
        Polku skeema-YAML:in.

    Returns
    -------
    pandera.DataFrameSchema
        Strict-skeema, jonka ``validate(df)`` tarkistaa sekä tyypit että
        sallitut arvot.

    Notes
    -----
    Skeema on ``strict=True``, ``ordered=False``: tuntemattomat sarakkeet
    aiheuttavat virheen, mutta sarakejärjestyksen voi vapaasti vaihtaa
    ekstraktion aikana. Lopullisessa CSV:ssä järjestys taataan
    ``column_order()`` -funktiolla.
    """
    spec = load_schema_yaml(yaml_path)
    enums: Dict[str, List[Any]] = spec.get("enums", {})

    columns: Dict[str, pa.Column] = {}
    for col_name, col_spec in spec["columns"].items():
        columns[col_name] = _build_column(col_name, col_spec, enums)

    schema = pa.DataFrameSchema(
        columns=columns,
        strict=True,
        ordered=False,
        # name näkyy Panderan virheviesteissä — auttaa lokien luettavuudessa.
        name="coamorphous_corpus_v1",
    )
    logger.debug(
        "Pandera-skeema rakennettu: %d saraketta YAML:sta %s",
        len(columns),
        yaml_path,
    )
    return schema
