"""Yhteiset pytest-fikstuurit.

MIKSI tämä tiedosto on olemassa
-------------------------------
Useat testit tarvitsevat polun ``configs/corpus_schema.yaml`` -tiedostoon.
Sen löytäminen testikoodista on kömpelöä, jos jokainen testi tekee sen
itse — keskitetään fikstuuriin.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Projektin juurihakemisto (sisältää configs/, src/, tests/)."""
    # Tämä tiedosto on tests/conftest.py, joten juurikansio on yksi taso ylempänä.
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def schema_yaml(project_root: Path) -> Path:
    """Polku korpuksen skeema-YAML:in."""
    return project_root / "configs" / "corpus_schema.yaml"
