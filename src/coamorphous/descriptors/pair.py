"""Pari-tason deskriptorit ko-amorfisille pareille (H1 kohta 2.4).

MIKSI tämä moduuli on olemassa
------------------------------
Yhden molekyylin deskriptorit (ks. ``molecular.py``) eivät yksinään riitä
ennustamaan, miten *kaksi* molekyyliä käyttäytyvät yhdessä amorfisessa
seoksessa. Tarvitaan piirteitä, jotka kuvaavat:

* **Erotuksia** (``delta_*``) — kun komponentit ovat hyvin erilaisia
  (esim. delta_MW iso), seoksen Tg ja sekoittuvuus muuttuvat ennustettavasti.
* **Summia** (``sum_*``) — ko-amorfisen seoksen Tg approksimoituu usein
  Gordon-Taylor-yhtälöllä, joka käyttää lineaarikombinaatioita.
* **Samankaltaisuutta** (``tanimoto_*``) — Tanimoto ECFP-sormenjäljistä
  mittaa, miten samanlaisia molekyylit ovat rakenteellisesti.
  Tutkimuskirjallisuus (esim. Lobmann et al.) viittaa siihen, että erittäin
  samankaltaiset molekyylit eivät stabiloi toisiaan parhaiten — tarvitaan
  *komplementaarisuutta*, ei pelkkää samankaltaisuutta.
* **Vetysidoskomplementaarisuutta** — heuristinen mittari sille, voivatko
  molekyylit muodostaa molemminpuolisia A-B-vetysidoksia, mikä on
  ko-amorfisten seosten tärkein stabilointimekanismi.

Kaikki tässä moduulissa on selkeästi erotettu per-molekyyli-laskennasta,
jotta yksiköiden testaus on yksinkertaista.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

from coamorphous.descriptors.molecular import (
    compute_molecular_descriptors,
    ecfp_fingerprint,
)

logger = logging.getLogger(__name__)


def tanimoto_similarity(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> Optional[float]:
    """Laske Tanimoto-similariteetti kahden molekyylin välillä ECFP:llä.

    Parameters
    ----------
    smiles_a, smiles_b : str
        Kanoniset SMILES-merkkijonot.
    radius : int, default 2
        ECFP-säde (2 = ECFP4, 3 = ECFP6).
    n_bits : int, default 2048

    Returns
    -------
    float or None
        Tanimoto-arvo välillä [0, 1]. ``None`` jos jompikumpi SMILES on
        virheellinen.

    Notes
    -----
    Tanimoto-similariteetti on ``|A ∩ B| / |A ∪ B|`` bittijoukoille.
    Arvo 1.0 = identtiset sormenjäljet, 0.0 = ei yhtäkään yhteistä bittiä.
    """
    mol_a = Chem.MolFromSmiles(smiles_a) if smiles_a else None
    mol_b = Chem.MolFromSmiles(smiles_b) if smiles_b else None
    if mol_a is None or mol_b is None:
        logger.warning(
            "tanimoto_similarity: jäsentäminen epäonnistui (a=%r, b=%r)",
            smiles_a,
            smiles_b,
        )
        return None

    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=radius, nBits=n_bits)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=radius, nBits=n_bits)
    return float(TanimotoSimilarity(fp_a, fp_b))


def hbond_complementarity(hbd_a: float, hba_a: float, hbd_b: float, hba_b: float) -> float:
    """Heuristinen vetysidoskomplementaarisuusindeksi.

    Määritelmä:

        score = min(HBD_A, HBA_B) + min(HBD_B, HBA_A)

    MIKSI tämä määritelmä:
    Komplementaarisuus tarkoittaa, että toisen molekyylin luovuttajat
    voivat löytää toisen molekyylin vastaanottajia. ``min`` valitaan,
    koska vetysidos vaatii sekä luovuttajan että vastaanottajan — jos
    A:lla on 5 luovuttajaa mutta B:llä vain 1 vastaanottaja, niin
    rajoittavana tekijänä on B:n vastaanottajien määrä.

    Parameters
    ----------
    hbd_a, hba_a : float
        A:n vetysidoksen luovuttajien ja vastaanottajien lukumäärä.
    hbd_b, hba_b : float
        Vastaavasti B:lle.

    Returns
    -------
    float
        Komplementaarisuusindeksi (>=0). Korkeampi = enemmän mahdollisia
        A-B-vetysidoksia.
    """
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (hbd_a, hba_a, hbd_b, hba_b)):
        return float("nan")
    return float(min(hbd_a, hba_b) + min(hbd_b, hba_a))


def compute_pair_descriptors(smiles_a: str, smiles_b: str) -> Dict[str, float]:
    """Laske kaikki pari-tason deskriptorit YAML:ssa varatuilla nimillä.

    Parameters
    ----------
    smiles_a, smiles_b : str
        Kanoniset SMILES-merkkijonot.

    Returns
    -------
    dict
        Avaimet vastaavat ``configs/corpus_schema.yaml`` ryhmää F:
        ``delta_MW``, ``delta_LogP``, ``delta_TPSA``, ``delta_HBD``,
        ``delta_HBA``, ``sum_MW``, ``tanimoto_ECFP4``, ``tanimoto_ECFP6``,
        ``hbond_complementarity``.

    Notes
    -----
    Kaikki tulokset ovat ``float`` (NaN jos jompikumpi SMILES virheellinen),
    mikä tekee suoran taulukon päivityksen helpoksi:

        >>> df.loc[i, list(out)] = compute_pair_descriptors(a, b).values()
    """
    desc_a = compute_molecular_descriptors(smiles_a)
    desc_b = compute_molecular_descriptors(smiles_b)

    # Itseisarvoinen erotus on tyypillinen valinta, koska A/B-järjestys
    # on mielivaltainen — emme halua, että pari (X, Y) ja (Y, X) saavat
    # eri delta-arvon.
    def diff(key: str) -> float:
        return float(abs(desc_a[key] - desc_b[key]))

    out: Dict[str, float] = {
        "delta_MW": diff("MW"),
        "delta_LogP": diff("LogP"),
        "delta_TPSA": diff("TPSA"),
        "delta_HBD": diff("HBD"),
        "delta_HBA": diff("HBA"),
        "sum_MW": float(desc_a["MW"] + desc_b["MW"]),
        "tanimoto_ECFP4": tanimoto_similarity(smiles_a, smiles_b, radius=2) or float("nan"),
        "tanimoto_ECFP6": tanimoto_similarity(smiles_a, smiles_b, radius=3) or float("nan"),
        "hbond_complementarity": hbond_complementarity(
            desc_a["HBD"], desc_a["HBA"], desc_b["HBD"], desc_b["HBA"]
        ),
    }
    return out
