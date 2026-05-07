"""Yhdistä lähdekohtaiset interim-CSV:t master-korpukseksi.

Tämä on tuotantopipelinen ulospäin näkyvä wrapper, joka:

1. Lukee kaikki ``data/interim/*.csv`` -tiedostot.
2. Yhdistää ne yhdeksi DataFrame:ksi (``merge_interim``).
3. Yhtenäistää lääkenimet InChIKey-ankkurin avulla (``Naproxen`` /
   ``naproxen`` -> sama).
4. Snäppää mole_fraction-arvot yksinkertaisiin suhteisiin (``0.667`` ja
   ``0.6667`` -> ``0.6667``); alkuperäiset säilyvät ``*_raw``-sarakkeissa.
5. Raportoi ``pair_id``- ja InChIKey-pari-duplikaatit (ei pudota rivejä).
6. Validoi yhdistetyn datan Pandera-skeemaa vasten.
7. Kirjoittaa ``data/processed/coamorphous_corpus_v1.csv``.

Käyttö
------
::

    python tools/build_master_corpus.py
    python tools/build_master_corpus.py --output data/processed/v2.csv
    python tools/build_master_corpus.py --no-validate     # kehitysvaiheessa

Skripti tulostaa lokia INFO-tasolla, jotta varoitukset (duplikaatit,
nimien normalisoinnit) näkyvät terminaalissa eivätkä jää piiloon.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from coamorphous.corpus.merge import merge_interim


def _resolve_project_root(start: Path) -> Path:
    """Etsi projektin juuri kävelemällä ylöspäin kunnes ``pyproject.toml``
    löytyy. Tämä antaa skriptille saman cwd-riippumattomuuden kuin
    notebookilla on."""
    cur = start.resolve()
    while cur != cur.parent:
        if (cur / "pyproject.toml").exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError(
        "pyproject.toml ei löytynyt mistään ylähakemistosta — onko skripti "
        "ajossa coamorphous-repon ulkopuolella?"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rakenna master-korpus interim-CSV:istä.",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=None,
        help="Hakemisto josta etsitään *.csv (oletus: data/interim/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Master-CSV:n polku (oletus: data/processed/coamorphous_corpus_v1.csv).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Skeema-YAML:n polku (oletus: configs/corpus_schema.yaml).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Ohita Pandera-skeemavalidointi (kehitykseen).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Näytä vain WARNING-tason lokit (oletus: INFO).",
    )
    args = parser.parse_args(argv)

    # Loki: WARNING-taso piilottaa "luettu N riviä"-rivit, mutta säilyttää
    # duplikaattivaroitukset näkyvissä. INFO-tasoisena näkee koko etenemisen.
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("build_master_corpus")

    # Resolvoi polut projektin juuren suhteen, jotta skripti toimii cwd:stä
    # riippumatta (esim. VS Coden Run-nappi voi käyttää eri cwd:tä).
    project_root = _resolve_project_root(Path(__file__).parent)
    interim_dir = args.interim_dir or (project_root / "data" / "interim")
    output_path = args.output or (
        project_root / "data" / "processed" / "coamorphous_corpus_v1.csv"
    )
    schema_yaml = args.schema or (project_root / "configs" / "corpus_schema.yaml")

    if not interim_dir.is_dir():
        logger.error("Interim-hakemisto ei löydy: %s", interim_dir)
        return 1

    interim_paths = sorted(interim_dir.glob("*.csv"))
    if not interim_paths:
        logger.error("Ei CSV-tiedostoja hakemistossa %s", interim_dir)
        return 1

    logger.info(
        "Yhdistetään %d interim-tiedostoa: %s",
        len(interim_paths),
        [p.name for p in interim_paths],
    )

    df = merge_interim(
        interim_paths,
        schema_yaml=schema_yaml,
        validate=not args.no_validate,
    )

    # Kirjoita output. mkdir varmistaa että data/processed/ on olemassa
    # vaikka repo on freshly cloned eikä siellä ole vielä master-korpusta.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # ;-sarakeerotin Excel-yhteensopivuuden takia (suomalainen locale avaa
    # tiedoston tuplaklikkauksella). Desimaalimerkki pidetään pisteenä, jotta
    # tiedostoa voi lukea myös pandalla ja muilla työkaluilla ilman erikoisia
    # locale-asetuksia — Excel osaa tulkita "0.5" lukuna numerosarakkeessa
    # vaikka locale olisi suomi.
    df.to_csv(output_path, index=False, sep=";")

    logger.info(
        "Master-korpus kirjoitettu: %s (%d riviä, %d saraketta)",
        output_path,
        len(df),
        len(df.columns),
    )

    # Ihmisille luettava yhteenveto stdoutiin (loki menee stderriin).
    # Suhteellinen polku jos output on projektin sisällä, muuten absoluuttinen
    # — tämä helpottaa testeissä, joissa output on /tmp:ssä.
    try:
        display_path = output_path.relative_to(project_root)
    except ValueError:
        display_path = output_path
    print()
    print(f"[OK] Master-korpus: {display_path}")
    print(f"     Rivejä: {len(df)}")
    print(f"     Lähteitä: {df['source_first_author'].nunique()} kpl")
    print(f"     Vuosiväli: {int(df['source_year'].min())}-{int(df['source_year'].max())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
