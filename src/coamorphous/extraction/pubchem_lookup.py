"""PubChem PUG REST -rajapinnan suorat haut.

MIKSI tämä moduuli on olemassa
------------------------------
Lähdeartikkelit käyttävät yleensä lääkeaineen triviaalinimeä ("naproxen",
"indomethacin"), mutta master-CSV vaatii kanonisen SMILES:n ja InChIKey:n
duplikaattien tunnistamiseen ja kemiallisten deskriptorien laskentaan.
PubChem on avoimen lääkkeenkemian tietokanta, jossa on PUG REST -rajapinta
nimi/CAS -> SMILES/InChIKey -muunnokseen.

Tärkeä periaate: emme anna LLM:n keksiä SMILES-merkkijonoja muistista,
koska ne voivat olla virheellisiä. Tämä moduuli on yksi totuuden lähde
ja kutsutaan ohjelmallisesti ``enrich.pubchem_lookup_pair`` -funktiosta.

Rajoitukset
-----------
PubChem PUG REST on rajoitettu ~5 pyyntöön sekunnissa per IP. Käytämme
``with_retry`` -dekoraattoria käsittelemään tilapäiset virheet (HTTP 503,
ConnectionError) ekspotentiaalisella backoffilla.

Reference
---------
PUG REST docs: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Optional, TypeVar

import requests

logger = logging.getLogger(__name__)

# PUG REST -juuri. Vakio modulissa, jotta testit voivat tarvittaessa
# monkeypatchata sen mock-palvelimeen.
PUBCHEM_BASE_URL: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# HTTP-pyyntöjen oletuserttiulosaika sekunteina. Pidetään lyhyenä (10 s),
# jotta verkkohäiriö ei jumita ekstraktiota; ``with_retry`` yrittää uudelleen.
DEFAULT_TIMEOUT_S: float = 10.0

# Yleisiä suolasanoja, jotka poistetaan ennen PubChem-hakua. Tärkeää, koska
# lähde voi sanoa "naproxen sodium", mutta haluamme vapaa-emäs-rakenteen
# ML-piirteiden laskentaan.
_SALT_SUFFIXES: tuple[str, ...] = (
    " hydrochloride",
    " sodium",
    " potassium",
    " calcium",
    " sulfate",
    " sulphate",
    " mesylate",
    " maleate",
    " citrate",
    " acetate",
    " phosphate",
    " hcl",
)

T = TypeVar("T")


class PubChemError(RuntimeError):
    """Heitettiin, kun PubChem-haku epäonnistuu pysyvästi.

    Erillinen poikkeustyyppi helpottaa kutsuvan koodin suodatusta:
    ``enrich.pubchem_lookup_pair`` nappaa vain tämän, ei kaikkia
    ``RuntimeError``-poikkeuksia.
    """


def normalize_drug_name(name: str) -> str:
    """Yksinkertaistettu lääkeaineen nimen normalisointi PubChem-hakuun.

    Säännöt:
      * lowercase + strip,
      * "γ-" (kreikkalainen gamma) korvataan "gamma-":lla, koska PubChem
        ei ymmärrä unicode-symboleita haussa (vastaavasti α/β),
      * suolaliitteet (esim. " hydrochloride", " sodium") poistetaan
        — haetaan vapaa-emäs-rakenne, joka on ML:n kannalta merkittävämpi.

    Alkuperäinen nimi pitäisi säilyttää eri kentässä auditointia varten —
    tämä funktio palauttaa vain hakemiseen sopivan version.

    Examples
    --------
    >>> normalize_drug_name("Naproxen Sodium")
    'naproxen'
    >>> normalize_drug_name("γ-Indomethacin")
    'gamma-indomethacin'
    >>> normalize_drug_name("Indomethacin HCl")
    'indomethacin'
    """
    if name is None:
        return ""
    s = name.strip().lower()

    # Kreikkalaiset kirjaimet pre-prosessina ennen suolanpoistoa.
    s = s.replace("γ", "gamma").replace("α", "alpha").replace("β", "beta")

    # Tavallisten unicode-yhdysmerkkien yhdenmukaistus.
    s = s.replace("\u2010", "-").replace("\u2013", "-").replace("\u2014", "-")

    # Poista suolasuffiksit häntäpäästä iteratiivisesti — esim.
    # "indomethacin hydrochloride hcl" -> "indomethacin".
    changed = True
    while changed:
        changed = False
        for suffix in _SALT_SUFFIXES:
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
                changed = True
                break

    # Tiivistä peräkkäiset välilyönnit yhdeksi.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def with_retry(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    backoff: float = 1.0,
    **kwargs: Any,
) -> T:
    """Aja ``func`` uudelleenyrityksellä ekspotentiaalisella backoffilla.

    PubChem PUG REST rajoittaa 5 pyyntöä sekunnissa, ja palauttaa
    HTTP 503 (PUGREST.ServerBusy) tilapäisesti. Tämä apuri yrittää
    uudelleen ``max_retries`` kertaa siten, että odotus on
    ``backoff * 2^attempt`` sekuntia.

    Parameters
    ----------
    func : callable
        Funktio, joka palauttaa pyytämämme tuloksen tai heittää
        ``requests.RequestException`` / ``PubChemError``.
    *args, **kwargs
        Välitetään suoraan ``func``-kutsulle.
    max_retries : int, default 3
        Yritysten enimmäismäärä. Ensimmäinen yritys ei ole "retry"
        tämän laskurin mielessä.
    backoff : float, default 1.0
        Pohjaodotus sekunteina; viive on ``backoff * 2^attempt``.

    Returns
    -------
    T
        ``func``:n palautusarvo onnistuneella yrityksellä.

    Raises
    ------
    PubChemError
        Jos kaikki yritykset epäonnistuivat — viimeisin virhe wrapataan.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (requests.RequestException, PubChemError) as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            sleep_s = backoff * (2 ** attempt)
            logger.warning(
                "PubChem-pyyntö epäonnistui (yritys %d/%d): %s. "
                "Odotetaan %.1f s ennen uudelleenyritystä.",
                attempt + 1, max_retries + 1, exc, sleep_s,
            )
            time.sleep(sleep_s)
    # Wrappaa lopullinen virhe selvällä viestillä.
    raise PubChemError(
        f"PubChem-pyyntö epäonnistui {max_retries + 1} yrityksen jälkeen: {last_exc}"
    ) from last_exc


def _get_json(url: str, timeout: float = DEFAULT_TIMEOUT_S) -> dict:
    """Hae JSON-vastaus annetusta URL:sta, raise PubChemError virheissä."""
    logger.debug("PubChem GET %s", url)
    resp = requests.get(url, timeout=timeout)
    if resp.status_code == 404:
        # 404 = ei tuloksia, ei verkkovirhe; ei retry-yritettävää.
        raise PubChemError(f"PubChem 404 (ei osumia): {url}")
    if resp.status_code >= 500:
        # 5xx on tilapäinen, retry kannattaa.
        raise requests.HTTPError(f"PubChem {resp.status_code}: {url}")
    resp.raise_for_status()
    return resp.json()


def _cid_from_response(data: dict) -> Optional[int]:
    """Pura ensimmäinen CID PUG-vastauksen ``IdentifierList.CID``-listasta."""
    cids = data.get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None
    return int(cids[0])


def _properties_for_cid(cid: int, timeout: float = DEFAULT_TIMEOUT_S) -> dict:
    """Hae CID:lle CanonicalSMILES, InChIKey, MolecularWeight."""
    url = (
        f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/"
        f"CanonicalSMILES,InChIKey,MolecularWeight/JSON"
    )
    data = _get_json(url, timeout=timeout)
    props_list = data.get("PropertyTable", {}).get("Properties", [])
    if not props_list:
        raise PubChemError(f"PubChem palautti tyhjät ominaisuudet CID {cid}:lle")
    return props_list[0]


def _build_lookup_result(cid: int, props: dict, cas: Optional[str] = None) -> dict:
    """Rakenna yhdenmukainen palautusrakenne haku-funktioista.

    Avaimet pidetään yksinkertaisina (cid, smiles, inchikey, mw, cas), jotta
    ``enrich``-moduuli osaa kuluttaa ne suoraan ilman erillistä mappausta.
    MolecularWeight tulee PubChem:istä str-tyyppisenä, joten muunnetaan
    floatiksi.
    """
    mw_raw = props.get("MolecularWeight")
    try:
        mw = float(mw_raw) if mw_raw is not None else None
    except (TypeError, ValueError):
        mw = None

    return {
        "cid": cid,
        "smiles": props.get("CanonicalSMILES"),
        "inchikey": props.get("InChIKey"),
        "mw": mw,
        "cas": cas,
    }


def search_by_name(name: str, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[dict]:
    """Hae yhdistettä PubChem:istä nimellä.

    Parameters
    ----------
    name : str
        Lääkeaineen nimi sellaisena kuin se esiintyy lähdeartikkelissa.
        Funktio normalisoi sen ennen hakua (``normalize_drug_name``).
    timeout : float, default 10
        HTTP-aikakatkaisu sekunteina.

    Returns
    -------
    dict or None
        ``{cid, smiles, inchikey, mw, cas}`` jos haku onnistui;
        ``None`` jos PubChem ei tunnistanut nimeä (404). Verkkovirhe
        nostaa ``PubChemError``:n.

    Notes
    -----
    Ei kutsu ``with_retry``:ä itse; kutsuva koodi voi käyttää sitä
    tarvittaessa: ``with_retry(search_by_name, "naproxen")``. Näin
    moduulin julkinen API pysyy pelkistetynä.
    """
    normalized = normalize_drug_name(name)
    if not normalized:
        logger.info("Tyhjä normalisoitu nimi syötteelle %r; ohitetaan haku.", name)
        return None

    logger.info("PubChem-haku nimellä: %r (alkuperäinen: %r)", normalized, name)
    url = f"{PUBCHEM_BASE_URL}/compound/name/{normalized}/cids/JSON"
    try:
        data = _get_json(url, timeout=timeout)
    except PubChemError as exc:
        logger.info("PubChem ei löytänyt nimeä %r: %s", normalized, exc)
        return None

    cid = _cid_from_response(data)
    if cid is None:
        return None

    props = _properties_for_cid(cid, timeout=timeout)
    return _build_lookup_result(cid, props)


def search_by_cas(cas: str, timeout: float = DEFAULT_TIMEOUT_S) -> Optional[dict]:
    """Hae yhdistettä PubChem:istä CAS-rekisterinumerolla.

    PubChem hyväksyy CAS:n nimi-päätepisteen kautta — sama URL-rakenne
    toimii sekä triviaalinimelle että CAS:lle. Tämä on PubChem:n
    dokumentoitu käyttäytyminen.

    Returns
    -------
    dict or None
        Sama rakenne kuin ``search_by_name``, mutta ``cas``-kenttä
        täytetään syötetyllä CAS:lla.
    """
    if not cas or not cas.strip():
        return None
    cas_clean = cas.strip()

    logger.info("PubChem-haku CAS:lla: %r", cas_clean)
    url = f"{PUBCHEM_BASE_URL}/compound/name/{cas_clean}/cids/JSON"
    try:
        data = _get_json(url, timeout=timeout)
    except PubChemError as exc:
        logger.info("PubChem ei löytänyt CAS:ta %r: %s", cas_clean, exc)
        return None

    cid = _cid_from_response(data)
    if cid is None:
        return None

    props = _properties_for_cid(cid, timeout=timeout)
    return _build_lookup_result(cid, props, cas=cas_clean)
