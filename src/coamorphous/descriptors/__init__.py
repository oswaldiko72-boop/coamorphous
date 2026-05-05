"""coamorphous.descriptors — kemialliset deskriptorit (H1 kohta 2.4).

MIKSI tämä alipaketti on olemassa
---------------------------------
ML-mallit eivät käsittele molekyylejä suoraan; ne tarvitsevat numeerisia
piirteitä. Tässä alipaketissa lasketaan:

* per-molekyyli RDKit-deskriptorit (MW, LogP, TPSA, HBD/HBA jne.)
* per-molekyyli ECFP-sormenjäljet (extended-connectivity fingerprints)
* pari-tason johdetut piirteet (delta, sum, Tanimoto, vetysidoskomplementaarisuus)

Pari-tason piirteet ovat tutkimuskysymyksen ydin: ne yrittävät vangita,
miksi *kaksi* molekyyliä yhdessä muodostavat stabiilin amorfisen seoksen,
mitä yksittäinen molekyyli ei voi.
"""

from coamorphous.descriptors.molecular import (
    compute_molecular_descriptors,
    ecfp_fingerprint,
)
from coamorphous.descriptors.pair import (
    compute_pair_descriptors,
    hbond_complementarity,
    tanimoto_similarity,
)

__all__ = [
    "compute_molecular_descriptors",
    "ecfp_fingerprint",
    "compute_pair_descriptors",
    "hbond_complementarity",
    "tanimoto_similarity",
]
