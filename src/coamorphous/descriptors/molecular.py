"""Per-molekyyli kemialliset deskriptorit (RDKit) ja ECFP-sormenjäljet.

MIKSI tämä moduuli on olemassa
------------------------------
ML-malli ei voi ottaa SMILES-merkkijonoa syötteenä; se tarvitsee
numeerisen vektorin. Lääkemolekyyleille käytettyjä standardideskriptoreita
ovat:

* **MW** — molekyylimassa (g/mol). Vaikuttaa difuusioon ja Tg:hen.
* **LogP** — oktanoli-vesi -jakautumiskerroin. Korreloi liukoisuuden ja
  amorfisten seosten sekoittuvuuden kanssa.
* **TPSA** — topologinen polaarinen pinta-ala. Heijastaa vetysidoskykyä.
* **HBD / HBA** — vetysidoksen luovuttajat / vastaanottajat. Ratkaiseva
  tekijä ko-amorfisten seosten stabiilisuudessa, koska heteromolekulaariset
  vetysidokset stabiloivat amorfista faasia.
* **NumRotatableBonds** — kiertyvien sidosten määrä. Korkea arvo = paljon
  konformaatioita = matala kiteytymistaipumus.

ECFP (Extended-Connectivity Fingerprint, Rogers & Hahn 2010) on hash-pohjainen
sormenjälki, joka koodaa atomin ympäristön tiettyyn säteeseen asti.
ECFP4 = halkaisija 4 (eli säde 2), ECFP6 = halkaisija 6.

Tämä moduuli tarjoaa puhtaan API:n, jota voi vektorisoidusti soveltaa
DataFrame-sarakkeisiin (esim. ``df['drug_A_smiles_canonical'].apply(...)``).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski

logger = logging.getLogger(__name__)


# Mitkä RDKit-deskriptorit lasketaan oletuksena.
# MIKSI tämä lista, ei kaikki Descriptors:n tarjoamat ~200 muuttujaa:
# Pieni, hyvin perusteltu joukko on aluksi parempi kuin laaja, kohinainen
# joukko — H1:n baseline-mallit toimivat näillä, ja Mordred-laajempi joukko
# voidaan ottaa käyttöön myöhemmin H1 2.4:ssä.
DEFAULT_DESCRIPTOR_FUNCTIONS: Dict[str, callable] = {
    "MW": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "HBD": Lipinski.NumHDonors,
    "HBA": Lipinski.NumHAcceptors,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "FractionCSP3": Descriptors.FractionCSP3,
    "NumHeavyAtoms": Descriptors.HeavyAtomCount,
}


def compute_molecular_descriptors(
    smiles: str,
    descriptor_funcs: Optional[Dict[str, callable]] = None,
) -> Dict[str, float]:
    """Laske per-molekyyli RDKit-deskriptorit kanonisesta SMILES:sta.

    Parameters
    ----------
    smiles : str
        Kanoninen SMILES (mielellään kanonisoitu jo
        :func:`coamorphous.corpus.canonicalize.canonical_smiles` -funktiolla).
    descriptor_funcs : dict, optional
        Karttaus deskriptorin nimestä RDKit-funktioon. Jos ``None``,
        käytetään :data:`DEFAULT_DESCRIPTOR_FUNCTIONS`.

    Returns
    -------
    dict
        ``{"MW": 357.79, "LogP": 4.27, ...}``. Jos SMILES on virheellinen,
        kaikki arvot ovat ``np.nan`` ja loki sisältää varoituksen.

    Notes
    -----
    Funktio palauttaa NaN-täytetyn dictin virhetilanteessa sen sijaan,
    että nostaisi poikkeuksen — tämä on tahallinen valinta, jotta yhden
    rivin virhe ei pysäytä koko korpuksen prosessointia. Validointi
    tapahtuu Pandera-skeemassa.
    """
    if descriptor_funcs is None:
        descriptor_funcs = DEFAULT_DESCRIPTOR_FUNCTIONS

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        logger.warning("compute_molecular_descriptors: virheellinen SMILES %r", smiles)
        return {name: float("nan") for name in descriptor_funcs}

    return {name: float(fn(mol)) for name, fn in descriptor_funcs.items()}


def ecfp_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """Laske ECFP-sormenjälki (Morgan fingerprint) NumPy-vektorina.

    Parameters
    ----------
    smiles : str
        Kanoninen SMILES.
    radius : int, default 2
        ECFP-säde. ``radius=2`` vastaa ECFP4:ää (halkaisija = 2 * radius).
        ``radius=3`` vastaa ECFP6:ta.
    n_bits : int, default 2048
        Sormenjäljen pituus. 2048 on yleinen valinta tarkkuuden ja
        muistinkäytön välillä.

    Returns
    -------
    numpy.ndarray of shape (n_bits,) tai None
        0/1-vektori. ``None`` jos SMILES ei ole jäsenneltävissä.

    Notes
    -----
    ECFP-sormenjäljet ovat *hash-pohjaisia*: kaksi eri rakennetta voi
    osua samaan bittiin (collision). Suuremmat ``n_bits`` vähentävät
    törmäyksiä mutta kasvattavat muistinkäyttöä.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        logger.warning("ecfp_fingerprint: virheellinen SMILES %r", smiles)
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    # ConvertToNumpyArray tarvitsee preallokoidun vektorin. Tämä on RDKit:n
    # nopein reitti; vaihtoehtoinen DataStructs.cDataStructs-kierto on
    # huomattavasti hitaampi suurilla aineistoilla.
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, arr)
    return arr
