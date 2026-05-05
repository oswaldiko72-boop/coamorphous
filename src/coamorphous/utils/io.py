"""I/O-apurit korpuksen luku- ja kirjoitusoperaatioille.

MIKSI tämä moduuli on olemassa
------------------------------
pandas.read_csv ja DataFrame.to_csv toimivat, mutta:

* CSV ei säilytä tyyppejä — int-sarake tulkitaan float:ksi heti kun
  yksikin NaN ilmestyy. Tämä rikkoo Pandera-validoinnin myöhemmin.
* Parquet säilyttää tyypit ja on huomattavasti pienempi suurilla DataFrameilla.

Siksi:

* ``read_corpus`` osaa lukea sekä CSV:n että Parquet:n ja palauttaa aina
  saman tyyppisen DataFrame:n.
* ``write_corpus`` kirjoittaa molemmat formaatit, jotta CSV:tä voi avata
  Excelissä manuaaliseen tarkistukseen, ja Parquet:tä käytetään
  notebookien välillä.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_corpus(path: Path) -> pd.DataFrame:
    """Lue korpus CSV- tai Parquet-tiedostosta.

    Parameters
    ----------
    path : Path
        Tiedoston polku. Pääte (``.csv`` tai ``.parquet``) ratkaisee
        lukutavan.

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Tuntematon tiedostopääte {suffix!r}; tuetut: .csv, .parquet"
        )
    logger.info("Luettu %d riviä tiedostosta %s", len(df), path)
    return df


def write_corpus(df: pd.DataFrame, path: Path) -> None:
    """Kirjoita korpus CSV- tai Parquet-tiedostoon.

    Parameters
    ----------
    df : pandas.DataFrame
    path : Path
        Tiedoston polku. Pääte ratkaisee tallennusmuodon.

    Notes
    -----
    Hakemistorakenne luodaan automaattisesti, jotta kutsujan ei tarvitse
    huolehtia ``mkdir`` -kutsuista.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(
            f"Tuntematon tiedostopääte {suffix!r}; tuetut: .csv, .parquet"
        )
    logger.info("Kirjoitettu %d riviä tiedostoon %s", len(df), path)
