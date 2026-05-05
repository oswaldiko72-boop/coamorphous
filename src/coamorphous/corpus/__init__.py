"""coamorphous.corpus — datakorpuksen rakentaminen ja validointi.

MIKSI tämä alipaketti on olemassa:
    Korpuksen laatu määrittää kaikki ML-mallit, jotka rakennetaan myöhemmin.
    Yhden virheellisen rivin (väärä SMILES, väärä luokka, duplikaatti)
    vaikutus voi näkyä koko aineiston ennusteissa. Siksi koko datapipeline
    — skeemavalidointi, SMILES-kanonisointi, Baird-Taylor luokitus ja
    lähteiden yhdistäminen — on koottu yhteen ja yksikkötestattu.
"""

from coamorphous.corpus.schema import build_schema, load_schema_yaml
from coamorphous.corpus.canonicalize import (
    CanonicalizationError,
    canonical_smiles,
    inchikey_from_smiles,
    validate_smiles,
)
from coamorphous.corpus.classify import classify_baird_taylor

__all__ = [
    "build_schema",
    "load_schema_yaml",
    "CanonicalizationError",
    "canonical_smiles",
    "inchikey_from_smiles",
    "validate_smiles",
    "classify_baird_taylor",
]
