"""coamorphous.corpus — datakorpuksen rakentaminen ja validointi.

MIKSI tämä alipaketti on olemassa:
    Korpuksen laatu määrittää kaikki ML-mallit, jotka rakennetaan myöhemmin.
    Yhden virheellisen rivin (väärä SMILES, väärä luokka, duplikaatti)
    vaikutus voi näkyä koko aineiston ennusteissa. Siksi koko datapipeline
    — skeemavalidointi, SMILES-kanonisointi, GFA- ja stabiilisuusluokitus
    sekä lähteiden yhdistäminen — on koottu yhteen ja yksikkötestattu.
"""

from coamorphous.corpus.schema import build_schema, load_schema_yaml
from coamorphous.corpus.canonicalize import (
    CanonicalizationError,
    canonical_smiles,
    inchikey_from_smiles,
    validate_smiles,
)
from coamorphous.corpus.classify import (
    classify_gfa_dsc,
    classify_stability_label_confidence,
    classify_stability_protocol,
    classify_stability_week_bin,
)

__all__ = [
    "build_schema",
    "load_schema_yaml",
    "CanonicalizationError",
    "canonical_smiles",
    "inchikey_from_smiles",
    "validate_smiles",
    "classify_gfa_dsc",
    "classify_stability_label_confidence",
    "classify_stability_protocol",
    "classify_stability_week_bin",
]
