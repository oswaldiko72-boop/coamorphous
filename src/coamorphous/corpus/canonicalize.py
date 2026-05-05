"""SMILES-kanonisointi ja InChIKey-haku RDKit:llä.

MIKSI tämä moduuli on olemassa
------------------------------
Sama molekyyli voidaan kirjoittaa useilla eri SMILES-merkkijonoilla — atomien
järjestys, aromaattisuuden esitystapa (Kekulé vs. aromaattinen) ja tähän liittyvä
implisiittinen vetyjen laskenta vaihtelevat tapauskohtaisesti. Esimerkiksi
indometasiini voidaan kirjoittaa muodossa
``CC1=C(C2=CC(=CC=C2N1C(=O)C3=CC=C(C=C3)Cl)OC)CC(=O)O`` tai useilla muilla
ekvivalenteilla tavoilla.

Jotta voimme:

* yhdistää saman molekyylin esiintymät eri lähteistä,
* tarkistaa duplikaatit korpuksessa,
* hakea PubChem/DrugBank-tietoja (jotka käyttävät InChIKey-avainta),

täytyy SMILES kanonisoida deterministisesti. RDKit:n
``Chem.MolToSmiles(mol, canonical=True)`` toteuttaa Daylight-tyylisen
kanonisointialgoritmin, joka tuottaa saman merkkijonon riippumatta siitä,
miten molekyyli kirjoitettiin.

InChIKey on InChI-merkkijonon SHA-256-pohjainen 27-merkkinen tiiviste, joka
toimii kemiallisena hashina; sitä käytetään useimmissa kemian tietokannoissa
(PubChem, DrugBank, ChEMBL) yksilölliseksi tunnisteeksi.
"""

from __future__ import annotations

import logging
from typing import Optional

from rdkit import Chem
from rdkit import RDLogger

logger = logging.getLogger(__name__)

# RDKit kirjoittaa parsinta-varoituksia stderriin (esim. "Explicit valence
# greater than permitted"). Hiljennetään ne — käsittelemme virheet
# eksplisiittisesti tarkistamalla, onko Mol-objekti None.
RDLogger.DisableLog("rdApp.*")


class CanonicalizationError(ValueError):
    """Heitettiin, kun SMILES ei ole RDKit:llä jäsenneltävissä.

    MIKSI oma poikkeustyyppi: kutsuvan koodin (esim. ekstraktion
    notebookien) on helppo erottaa kanonisoinnin epäonnistuminen muista
    ValueError-tyyppisistä virheistä.
    """


def _parse(smiles: str) -> Chem.Mol:
    """Sisäinen apufunktio: jäsennä SMILES Mol-objektiksi tai heitä virhe.

    RDKit palauttaa ``None`` jäsentämiselle epäonnistuneelle SMILES:lle eikä
    heitä poikkeusta. Keskitämme tämän tarkistuksen yhteen paikkaan, jotta
    kaikki julkiset funktiot heittävät yhdenmukaisen virheen.
    """
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        raise CanonicalizationError(
            f"Tyhjä tai virheellinen SMILES-syöte: {smiles!r}"
        )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise CanonicalizationError(
            f"RDKit ei pystynyt jäsentämään SMILES-merkkijonoa: {smiles!r}"
        )
    return mol


def canonical_smiles(smiles: str) -> str:
    """Palauta SMILES-merkkijonon kanoninen muoto.

    Parameters
    ----------
    smiles : str
        Mielivaltainen SMILES-merkkijono (esim. lähdeartikkelista kopioitu).

    Returns
    -------
    str
        Kanoninen SMILES, joka on sama kaikille saman molekyylin
        ekvivalenteille kirjoitusmuodoille.

    Raises
    ------
    CanonicalizationError
        Jos syöte on tyhjä tai RDKit ei pysty jäsentämään.

    Examples
    --------
    >>> canonical_smiles("OC(=O)Cc1ccccc1")
    'O=C(O)Cc1ccccc1'
    """
    mol = _parse(smiles)
    # canonical=True on RDKit:n oletus, mutta merkitään se eksplisiittisesti
    # — koodista lukijalle on heti ilmeistä, että kyseessä on kanonisointi.
    return Chem.MolToSmiles(mol, canonical=True)


def inchikey_from_smiles(smiles: str) -> str:
    """Laske SMILES-merkkijonosta InChIKey.

    InChIKey on 27-merkkinen kemiallisen rakenteen hash. Se jakautuu
    kolmeen osaan: ``XXXXXXXXXXXXXX-YYYYYYYYFV-P``, joista ensimmäinen
    14 merkkiä koodaa konnektiviteettia, seuraavat 10 stereokemiaa ja
    isotooppeja, ja viimeinen merkki on protonointitilan tunniste.

    Parameters
    ----------
    smiles : str
        Mielivaltainen SMILES-merkkijono.

    Returns
    -------
    str
        InChIKey-merkkijono.

    Raises
    ------
    CanonicalizationError
        Jos SMILES ei ole jäsenneltävissä, tai InChIKey-laskenta
        palauttaa tyhjän merkkijonon (esim. liian eksoottinen rakenne).
    """
    mol = _parse(smiles)
    key: str = Chem.MolToInchiKey(mol)
    if not key:
        raise CanonicalizationError(
            f"InChIKey-laskenta epäonnistui SMILES:lle {smiles!r}"
        )
    return key


def validate_smiles(smiles: str) -> bool:
    """Palauta True jos SMILES on RDKit:llä jäsenneltävissä, muuten False.

    MIKSI erillinen funktio, vaikka ``canonical_smiles`` heittää virheen:
    ekstraktion notebookeissa halutaan usein laskea, kuinka moni rivi
    läpäisee validoinnin, ennen kuin nostetaan poikkeus. Tämä on
    "ennakkotarkistus" virheellistä syötettä varten.

    Parameters
    ----------
    smiles : str
        SMILES-ehdokas.

    Returns
    -------
    bool
        True jos parsittavissa, False muutoin (mukaan lukien None/tyhjä).
    """
    try:
        _parse(smiles)
        return True
    except CanonicalizationError:
        return False


def safe_canonical(smiles: Optional[str]) -> Optional[str]:
    """Kanonisoi SMILES, palauta ``None`` jos jäsentäminen epäonnistuu.

    Hyödyllinen pandas DataFrame -applylle, jossa halutaan epäkelvon rivin
    sijaan ``None`` ja erillinen logiviesti, ei poikkeusta joka pysäyttää
    koko ekstraktion.
    """
    if smiles is None:
        return None
    try:
        return canonical_smiles(smiles)
    except CanonicalizationError as exc:
        logger.warning("Kanonisoinnin epäonnistuminen: %s", exc)
        return None
