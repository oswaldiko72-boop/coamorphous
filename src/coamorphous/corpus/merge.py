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
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from coamorphous.corpus.schema import build_schema, column_order, load_schema_yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Lääkenimien kanonisointi InChIKey-ankkurin avulla
# =============================================================================


def pick_canonical_name(names: Iterable[str]) -> str:
    """Valitse kanoninen muoto nimivarianttien listasta.

    Prioriteetti:

    1. **Yleisin** ('most_common') — heijastaa korpuksen valtavirtaa.
    2. **Pisin** — suosii kokonimiä lyhenteiden yli, esim. ``"Ezetimibe"``
       vs. ``"EZB"``, kun molempia esiintyy saman verran.
    3. **Aakkosjärjestys** — viimeinen tiebreaker, varmistaa determinismin.

    Tyhjät merkkijonot ja ``None``:t suodatetaan pois ennen päättelyä.

    Parameters
    ----------
    names : iterable of str
        Lista nimivarianteista, tyypillisesti samasta InChIKey:stä.

    Returns
    -------
    str
        Kanoninen muoto. Tyhjä ``""`` jos ei yhtään validia syötettä.

    Examples
    --------
    >>> pick_canonical_name(["naproxen", "Naproxen", "Naproxen"])
    'Naproxen'
    >>> pick_canonical_name(["EZB", "Ezetimibe"])
    'Ezetimibe'
    >>> pick_canonical_name(["Naproxen", "naproxen"])
    'Naproxen'
    """
    counter = Counter(n for n in names if isinstance(n, str) and n.strip())
    if not counter:
        return ""
    # min(...) ja avaintuple, jossa pienempi = parempi:
    #   -count        (eniten esiintymisiä voittaa)
    #   -len(name)    (pisin nimi voittaa)
    #   name          (aakkosellisesti pienin voittaa)
    return min(counter, key=lambda n: (-counter[n], -len(n), n))


def snap_to_simple_ratio(
    x: Optional[float],
    max_denominator: int = 20,
    tolerance: float = 0.002,
    decimals: int = 4,
) -> Optional[float]:
    """Snäppää arvo lähimpään yksinkertaiseen rationaalisuhteeseen p/q.

    Tarkoitus: yhtenäistää eri tarkkuudella raportoidut samat suhteet, kuten
    ``0.667`` ja ``0.6667``, jotka molemmat tarkoittavat 2/3:a. Jos arvo ei
    ole minkään yksinkertaisen suhteen lähellä (esim. ``0.506``, joka tulee
    massapainotuksesta), se palautetaan ennallaan.

    Parameters
    ----------
    x : float or None
        Arvo joukosta [0, 1] (esim. mole_fraction). ``None`` tai ``NaN``
        palautuu sellaisenaan. Joukon ulkopuoliset arvot ohitetaan.
    max_denominator : int, default 20
        Suurin nimittäjä jota kokeillaan. 20 kattaa kaikki tyypilliset
        ko-amorfisten parien suhteet (1:1, 2:1, 3:1, ..., 19:1, ja niiden
        käänteiset).
    tolerance : float, default 0.002
        Maksimi etäisyys ``|x - p/q|`` jotta snäppäys hyväksytään. ``0.002``
        on tarkoituksellisesti tiukka — esim. 1:1 (0.500) ja 0.506 jäävät
        molemmat ennalleen, mutta 0.667 (etäisyys 2/3:sta = 0.0003) snäppää.
    decimals : int, default 4
        Pyöristystarkkuus snäpätylle arvolle. 4 kattaa kaikki tarvitsemamme
        suhteet (1/3 ≈ 0.3333, 1/6 ≈ 0.1667, 1/7 ≈ 0.1429).

    Returns
    -------
    float or None
        Snäpätty arvo (pyöristetty ``decimals``:n tarkkuuteen) jos joku
        ``p/q`` osuu toleranssin sisälle; muuten alkuperäinen ``x``.
        ``None``/``NaN``-syötteet palautuvat sellaisinaan.

    Examples
    --------
    >>> snap_to_simple_ratio(0.667)
    0.6667
    >>> snap_to_simple_ratio(0.6667)
    0.6667
    >>> snap_to_simple_ratio(0.506)  # massapainotus, ei yksinkertainen suhde
    0.506
    >>> snap_to_simple_ratio(0.5)
    0.5
    >>> snap_to_simple_ratio(0.909)  # 10/11 = 0.9091, snäppää
    0.9091
    """
    if x is None or pd.isna(x):
        return x
    if not 0.0 <= x <= 1.0:
        # Ei mole_fraction-alueella — älä snäppää, palauta sellaisenaan.
        return x

    # Etsi lähin p/q joka on toleranssin sisällä. Käytetään pienintä nimittäjää
    # tasapelin ratkaisuun: 1/2 voittaa 2/4:n vaikka molemmat osuvat.
    best_value: Optional[float] = None
    best_diff = float("inf")
    best_q = max_denominator + 1
    for q in range(1, max_denominator + 1):
        # Lähin kokonaisluku p siten, että 0 <= p <= q ja p/q ≈ x.
        p = round(x * q)
        if p < 0 or p > q:
            continue
        candidate = p / q
        diff = abs(candidate - x)
        if diff > tolerance:
            continue
        # Pienempi nimittäjä on yksinkertaisempi suhde -> tasapelissä se voittaa.
        if diff < best_diff or (diff == best_diff and q < best_q):
            best_value = candidate
            best_diff = diff
            best_q = q

    if best_value is None:
        return x
    return round(best_value, decimals)


def harmonize_mole_fractions(
    df: pd.DataFrame,
    max_denominator: int = 20,
    tolerance: float = 0.002,
    decimals: int = 4,
) -> pd.DataFrame:
    """Yhtenäistä ``mole_fraction_A`` ja ``mole_fraction_B`` snäppäämällä
    yksinkertaisiin suhteisiin (1/2, 1/3, 2/3, 1/5, ...).

    MIKSI tärkeä: lähteet raportoivat saman 2:1-suhteen sekä ``0.667`` että
    ``0.6667`` -muodossa. ML-mallin opetuksessa tämä näkyy kahtena eri
    piirrearvona vaikka kemiallinen sisältö on identtinen. Snäppäys
    yhdistää ne yhteen kanoniseen arvoon menettämättä aitoa tarkkaa
    massapainotettua dataa kuten ``0.506`` (Knapik 2019).

    Käyttäytyminen:

    * Alkuperäiset arvot säilytetään sarakkeisiin ``mole_fraction_A_raw`` ja
      ``mole_fraction_B_raw`` auditointia varten.
    * ``mole_fraction_A`` snäpätään ``snap_to_simple_ratio``:n mukaan.
    * ``mole_fraction_B`` lasketaan ``round(1 - snapped_A, decimals)`` jotta
      summa = 1 takuulla. Tämä eliminoi pyöristyksen aiheuttamat
      epäkonsistenttisuudet, esim. snäpätty A=0.6667 + raportoitu B=0.333
      = 0.9997, mutta korjataan nyt B=0.3333.
    * Rivit joilla ``mole_fraction_A`` on NaN jätetään ennalleen
      molemmissa sarakkeissa.
    * ``weight_fraction_A`` ja ``weight_fraction_B`` jätetään koskematta —
      ne ovat eri konsepti (massa vs. mooli).

    Parameters
    ----------
    df : pandas.DataFrame
        Master-CSV-tason DataFrame, jossa ovat sarakkeet ``mole_fraction_A``
        ja ``mole_fraction_B``. Jos ``mole_fraction_A_raw`` /
        ``mole_fraction_B_raw`` puuttuvat, ne luodaan; jos olemassa, ne
        ylikirjoitetaan nykyisellä mole_fraction-arvolla ennen snäppäystä.
    max_denominator, tolerance, decimals
        Välitetään ``snap_to_simple_ratio``:lle.

    Returns
    -------
    pandas.DataFrame
        Uusi DataFrame (kopio). Alkuperäistä ei muteta.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "mole_fraction_A": [0.667, 0.6667, 0.506, None],
    ...     "mole_fraction_B": [0.333, 0.3333, 0.494, None],
    ... })
    >>> out = harmonize_mole_fractions(df)
    >>> out["mole_fraction_A"].tolist()
    [0.6667, 0.6667, 0.506, nan]
    >>> out["mole_fraction_A_raw"].tolist()
    [0.667, 0.6667, 0.506, nan]
    """
    required = {"mole_fraction_A", "mole_fraction_B"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"harmonize_mole_fractions: DataFrame:sta puuttuu sarakkeita: "
            f"{sorted(missing)}"
        )

    out = df.copy()

    # 1) Tallenna alkuperäiset *_raw-sarakkeisiin (auditointia varten).
    out["mole_fraction_A_raw"] = out["mole_fraction_A"]
    out["mole_fraction_B_raw"] = out["mole_fraction_B"]

    # 2) Snäppää A. NaN-rivit palautuvat NaN:nä ja jäävät ennalleen.
    snapped_A = out["mole_fraction_A"].apply(
        lambda v: snap_to_simple_ratio(v, max_denominator, tolerance, decimals)
    )

    # 3) Aseta B = 1 - snapped_A jos A on määritelty, muuten ennallaan.
    # Tämä takaa summa = 1 myös kun snäppäys muutti A:ta.
    mask = snapped_A.notna()
    out.loc[mask, "mole_fraction_A"] = snapped_A[mask]
    out.loc[mask, "mole_fraction_B"] = (1.0 - snapped_A[mask]).round(decimals)

    return out


def canonicalize_names_by_inchikey(df: pd.DataFrame) -> pd.DataFrame:
    """Yhtenäistä ``drug_A_name`` ja ``drug_B_name`` niin, että sama InChIKey
    saa aina saman kanonisen nimen koko korpuksen yli.

    Sama lääke voi esiintyä A-sarakkeessa yhdessä paperissa ja B-sarakkeessa
    toisessa — tämä funktio huomioi nimet **molemmista sarakkeista** kun se
    valitsee kanonisen muodon per InChIKey, ja kirjoittaa yhtenäistetyn nimen
    takaisin molempiin positioihin.

    Rivit, joilta InChIKey puuttuu, jätetään ennalleen — ilman kemiallista
    ankkuria emme voi varmuudella sanoa, että kaksi nimeä viittaavat samaan
    aineeseen.

    Parameters
    ----------
    df : pandas.DataFrame
        Master-CSV-tason DataFrame, jossa ovat sarakkeet ``drug_A_name``,
        ``drug_B_name``, ``drug_A_inchikey``, ``drug_B_inchikey``.

    Returns
    -------
    pandas.DataFrame
        Uusi DataFrame (kopio) jossa nimet on normalisoitu. Alkuperäistä
        DataFrame:a ei muuteta.

    Notes
    -----
    Funktio ei kosketa ``drug_*_smiles_canonical``- tai
    ``drug_*_inchikey``-sarakkeita — ne ovat jo kanonisia (RDKit + PubChem).
    Auditointi-tason raakatieto säilyy ekstraktion ``RawPair.drug_*_name_raw``
    -kentissä, jotka eivät kulkeudu master-CSV:hen tämän rajan yli.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "drug_A_name": ["naproxen", "Naproxen", "indomethacin"],
    ...     "drug_B_name": ["indomethacin", "indomethacin", "naproxen"],
    ...     "drug_A_inchikey": ["NAP", "NAP", "IND"],
    ...     "drug_B_inchikey": ["IND", "IND", "NAP"],
    ... })
    >>> out = canonicalize_names_by_inchikey(df)
    >>> set(out["drug_A_name"]) | set(out["drug_B_name"])
    {'Naproxen', 'indomethacin'}
    """
    required = {"drug_A_name", "drug_B_name", "drug_A_inchikey", "drug_B_inchikey"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"canonicalize_names_by_inchikey: DataFrame:sta puuttuu "
            f"sarakkeita: {sorted(missing)}"
        )

    out = df.copy()

    # 1) Kerää kaikki nimivariantit per InChIKey BOTH A- ja B-sarakkeista, jotta
    # sama lääke saa saman kanonisen nimen riippumatta kummalla puolella se on.
    names_by_inchikey: dict[str, List[str]] = {}
    for letter in ("A", "B"):
        ik_col = f"drug_{letter}_inchikey"
        name_col = f"drug_{letter}_name"
        for ik, name in zip(out[ik_col], out[name_col]):
            if not isinstance(ik, str) or not ik.strip():
                continue
            if not isinstance(name, str) or not name.strip():
                continue
            names_by_inchikey.setdefault(ik, []).append(name)

    # 2) Päättele kanoninen nimi per InChIKey.
    canonical = {ik: pick_canonical_name(names) for ik, names in names_by_inchikey.items()}

    # 3) Sovella molempiin sarakkeisiin. Säilytä alkuperäinen jos InChIKey
    # puuttuu tai sille ei löytynyt validia kanonista nimeä.
    for letter in ("A", "B"):
        ik_col = f"drug_{letter}_inchikey"
        name_col = f"drug_{letter}_name"
        mask = out[ik_col].apply(
            lambda v: isinstance(v, str) and v in canonical and bool(canonical[v])
        )
        if mask.any():
            out.loc[mask, name_col] = out.loc[mask, ik_col].map(canonical)
    return out


def _inchikey_pair_key(
    inchikey_A: Optional[str], inchikey_B: Optional[str]
) -> Optional[frozenset[str]]:
    """Muodosta järjestysriippumaton avain InChIKey-parista.

    Returns
    -------
    frozenset of two strings, or None
        ``None`` jos kumpi tahansa InChIKey puuttuu (NaN tai tyhjä) — silloin
        paria ei voi luotettavasti tunnistaa duplikaatiksi. ``frozenset`` on
        valittu ``tuple(sorted(...))``-vaihtoehdon sijaan, jotta yhden lääkkeen
        homo-pari (A=B, vrt. teoreettinen single-component glass) erottuu
        validilla yhden alkion joukolla, mutta käytännössä A ja B ovat aina eri
        lääkkeitä joten frozenset:ssa on kaksi alkiota.
    """
    # pandas voi antaa NaN floattina tai None:na riippuen kuinka CSV luettiin;
    # yhdistetään molemmat puuttuvuudet samaan tarkistukseen.
    if inchikey_A is None or inchikey_B is None:
        return None
    if not isinstance(inchikey_A, str) or not isinstance(inchikey_B, str):
        return None
    if not inchikey_A.strip() or not inchikey_B.strip():
        return None
    return frozenset({inchikey_A, inchikey_B})


def find_cross_source_duplicates(df: pd.DataFrame) -> List[dict]:
    """Etsi rivit, joissa sama InChIKey-pari esiintyy useammasta lähteestä.

    "Sama pari" tarkoittaa järjestysriippumatonta InChIKey-pari-identiteettiä:
    esim. (Naproxen, Ibuprofen) ja (Ibuprofen, Naproxen) ovat sama pari.
    "Cross-source" tarkoittaa, että rivit tulevat eri ``source_doi``- tai
    ``source_first_author`` + ``source_year`` -kombinaatiosta — saman lähteen
    sisäiset toistot eivät ole tämän funktion vastuulla (niitä käsittelee
    ``merge_interim``:n ``pair_id``-tarkistus).

    MIKSI tärkeä: jos sama pari esiintyy useassa lähteessä ja merkitään
    erillisinä riveinä master-CSV:ssä, ML-mallin opetuksessa pari "näkyy"
    useammin ja vinouttaa luokkajakaumaa. Funktio ei pudota rivejä — saman
    parin eri kompositiot tai säilytysolot ovat oikeasti eri datapisteitä,
    joten päätös on ihmisen tehtävä. Funktio vain raportoi.

    Parameters
    ----------
    df : pandas.DataFrame
        Master-CSV-tason DataFrame, jossa ovat ainakin sarakkeet
        ``drug_A_inchikey``, ``drug_B_inchikey``, ``source_doi``,
        ``source_first_author``, ``source_year``, ``pair_id``.
        Rivit, joilta puuttuu kumpikin InChIKey, ohitetaan hiljaa.

    Returns
    -------
    list of dict
        Yksi dict per duplikaattiryhmä, kentät:

        * ``inchikey_pair`` — ``tuple`` (sortattu), kahden lääkkeen InChIKey:t.
        * ``pair_ids`` — lista pair_id-arvoja ryhmässä.
        * ``sources`` — lista ``"{first_author}{year}"``-merkkijonoja.
        * ``row_indices`` — lista DataFrame-indeksiä (alkuperäinen järjestys).

        Lista on tyhjä jos cross-source-duplikaatteja ei ole. Saman lähteen
        toistuvat rivit eivät päädy listalle, vaikka ne jakaisivat
        InChIKey-parin (esim. eri kompositiot Knapik 2015:n EZB-IDP-rivit).

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "pair_id": ["a2009_01", "b2012_03"],
    ...     "drug_A_inchikey": ["AAA", "BBB"],
    ...     "drug_B_inchikey": ["BBB", "AAA"],   # käännetty järjestys
    ...     "source_first_author": ["alleso", "fink"],
    ...     "source_year": [2009, 2012],
    ...     "source_doi": ["10.x/aaa", "10.y/bbb"],
    ... })
    >>> groups = find_cross_source_duplicates(df)
    >>> len(groups)
    1
    >>> sorted(groups[0]["pair_ids"])
    ['a2009_01', 'b2012_03']
    """
    required_cols = {
        "drug_A_inchikey",
        "drug_B_inchikey",
        "source_first_author",
        "source_year",
        "pair_id",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(
            f"find_cross_source_duplicates: DataFrame:sta puuttuu sarakkeita: "
            f"{sorted(missing)}"
        )

    # Ryhmittele InChIKey-parin avaimella. Käytetään sanakirjaa, jossa avain on
    # frozenset ja arvo lista (rivi-indeksi, source-tunniste, pair_id).
    groups: dict[frozenset[str], list[tuple[int, str, str]]] = {}
    for idx, row in df.iterrows():
        key = _inchikey_pair_key(
            row.get("drug_A_inchikey"), row.get("drug_B_inchikey")
        )
        if key is None:
            # Ei voida arvioida ilman InChIKey:tä — sivuutetaan.
            continue
        # Source-tunniste yhdistää first_author + year:n compactiksi merkkijonoksi
        # ("alleso2009"), joka on luettavampi lokeissa kuin DOI.
        first_author = row.get("source_first_author") or "?"
        year = row.get("source_year")
        source = f"{first_author}{int(year)}" if pd.notna(year) else f"{first_author}?"
        groups.setdefault(key, []).append((idx, source, row.get("pair_id")))

    # Suodata vain ryhmät, joissa on ≥2 ERI lähdettä. Saman lähteen toistot
    # ohitetaan, koska ne eivät ole cross-source duplikaatteja.
    duplicate_groups: List[dict] = []
    for key, members in groups.items():
        sources = {m[1] for m in members}
        if len(sources) < 2:
            continue
        # Sortataan inchikey:t determinismin vuoksi.
        inchikey_pair = tuple(sorted(key))
        duplicate_groups.append(
            {
                "inchikey_pair": inchikey_pair,
                "pair_ids": [m[2] for m in members],
                "sources": [m[1] for m in members],
                "row_indices": [m[0] for m in members],
            }
        )

    # Stabiili järjestys raportoinnille: sortataan sortatun InChIKey-parin mukaan.
    duplicate_groups.sort(key=lambda g: g["inchikey_pair"])
    return duplicate_groups


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

    # Lisää puuttuvat sarakkeet skeematyyppiään vastaavalla tyhjällä
    # Series:llä, jotta Pandera-validointi ei kaadu pelkkään pd.NA:han
    # (joka olisi dtype 'object', ei float/int/bool).
    # MIKSI tärkeä: Pari-tason johdetut piirteet (delta_*, tanimoto_*) eivät
    # täyty interim-vaiheessa vaan H1 2.4:ssä — silloin niiden pitää silti
    # validoitua oikealla tyypillä jo nyt.
    _empty_for_dtype = {
        "string": (pd.NA, "string"),
        "int": (pd.NA, "Int64"),
        "float": (float("nan"), "float64"),
        "bool": (pd.NA, "boolean"),
    }
    for col in cols:
        if col not in merged.columns:
            col_spec = spec["columns"].get(col, {})
            dtype_key = col_spec.get("dtype", "string")
            fill_value, pandas_dtype = _empty_for_dtype.get(
                dtype_key, (pd.NA, "object")
            )
            merged[col] = pd.Series(
                [fill_value] * len(merged), dtype=pandas_dtype
            )

    # Pakota sarakejärjestys YAML:n mukaiseksi.
    merged = merged[cols]

    # Yhtenäistä lääkenimet InChIKey-ankkurin avulla, jotta esim. "naproxen"
    # ja "Naproxen" eri lähteistä päätyvät samaan kanoniseen muotoon.
    # Tämä tehdään ENNEN duplikaattitarkistusta vain nimien siisteyden vuoksi
    # raportoinnissa — duplikaattilogiikka itsessään käyttää InChIKey-pareja,
    # joten lopputulos olisi sama kummassa järjestyksessä tahansa.
    merged = canonicalize_names_by_inchikey(merged)

    # Yhtenäistä mole_fraction-arvot snäppäämällä yksinkertaisiin suhteisiin
    # (1/2, 2/3, 1/5, ...), jotta esim. 0.667 ja 0.6667 (molemmat 2:1) eivät
    # päädy ML-mallin opetukseen kahtena eri piirteenä. Alkuperäiset arvot
    # säilytetään mole_fraction_*_raw -sarakkeissa.
    merged = harmonize_mole_fractions(merged)

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

    # Cross-source duplikaattitarkistus InChIKey-parin perusteella. MIKSI:
    # pair_id on lähde-uniikki konstruktiollaan, joten se ei tunnista samaa
    # paria eri lähteistä. Sama lääkepari useassa lähteessä vinouttaa ML-mallin
    # opetusta, mutta toisaalta saman parin eri kompositiot/säilytysolot ovat
    # oikeasti eri datapisteitä — joten emme pudota rivejä, vain raportoimme.
    inchikey_dups = find_cross_source_duplicates(merged)
    if inchikey_dups:
        logger.warning(
            "Löytyi %d cross-source InChIKey-duplikaattiryhmää:", len(inchikey_dups)
        )
        for group in inchikey_dups:
            logger.warning(
                "  pari %s -> pair_ids=%s, lähteet=%s",
                group["inchikey_pair"],
                group["pair_ids"],
                sorted(set(group["sources"])),
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
