"""Yksikkötestit SMILES-kanonisoinnille.

MIKSI nämä testit ovat olemassa
-------------------------------
Kanonisoinnin pitää olla *idempotentti* (kanonisoitu pysyy kanonisena) ja
*ekvivalenttitietoinen* (sama molekyyli eri SMILES-kirjoituksissa tuottaa
saman tuloksen). Jos jompikumpi rikkoutuu, korpukseen voi syntyä
piileviä duplikaatteja, jotka näkyvät vasta ML-mallien
opetus/validointijaossa data-leakage-virheinä.
"""

from __future__ import annotations

import pytest

from coamorphous.corpus.canonicalize import (
    CanonicalizationError,
    canonical_smiles,
    inchikey_from_smiles,
    safe_canonical,
    validate_smiles,
)


# Indometasiini, kirjoitettu kahdessa eri muodossa. Molempien pitää tuottaa
# sama kanoninen SMILES ja sama InChIKey.
INDOMETHACIN_VARIANTS = [
    "CC1=C(C2=CC(=CC=C2N1C(=O)C3=CC=C(C=C3)Cl)OC)CC(=O)O",
    "Cc1c(CC(=O)O)c2cc(OC)ccc2n1C(=O)c1ccc(Cl)cc1",
]

# Etikkahappo: yksinkertainen tapaus, jossa alkuperäinen ja kanoninen voivat
# erota kirjoitusmuodossaan.
ACETIC_ACID_VARIANTS = [
    "CC(=O)O",
    "OC(=O)C",
    "OC(C)=O",
]


class TestCanonicalSmiles:
    def test_indomethacin_variants_collapse_to_single_canonical(self) -> None:
        """Eri kirjoitusmuodot indometasiinille -> sama kanoninen SMILES."""
        canonical_set = {canonical_smiles(s) for s in INDOMETHACIN_VARIANTS}
        assert len(canonical_set) == 1, (
            f"Kanonisointi ei kollapsannut variantteja yhdeksi: {canonical_set}"
        )

    def test_acetic_acid_variants_collapse(self) -> None:
        canonical_set = {canonical_smiles(s) for s in ACETIC_ACID_VARIANTS}
        assert len(canonical_set) == 1

    def test_canonical_is_idempotent(self) -> None:
        """Kanonisointi kahdesti = kanonisointi kerran."""
        s = INDOMETHACIN_VARIANTS[0]
        once = canonical_smiles(s)
        twice = canonical_smiles(once)
        assert once == twice

    def test_invalid_smiles_raises(self) -> None:
        with pytest.raises(CanonicalizationError):
            canonical_smiles("not_a_smiles_string!!!")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(CanonicalizationError):
            canonical_smiles("")

    def test_none_input_raises(self) -> None:
        with pytest.raises(CanonicalizationError):
            canonical_smiles(None)  # type: ignore[arg-type]


class TestInchikey:
    def test_indomethacin_variants_share_inchikey(self) -> None:
        keys = {inchikey_from_smiles(s) for s in INDOMETHACIN_VARIANTS}
        assert len(keys) == 1
        # InChIKey on muotoa XXXXXXXXXXXXXX-YYYYYYYYFV-P (14-10-1 merkkiä).
        (key,) = keys
        parts = key.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 14
        assert len(parts[1]) == 10
        assert len(parts[2]) == 1

    def test_invalid_smiles_raises(self) -> None:
        with pytest.raises(CanonicalizationError):
            inchikey_from_smiles("@@@@@")


class TestValidateSmiles:
    @pytest.mark.parametrize("smi", INDOMETHACIN_VARIANTS + ACETIC_ACID_VARIANTS)
    def test_valid_returns_true(self, smi: str) -> None:
        assert validate_smiles(smi) is True

    @pytest.mark.parametrize("smi", ["", "not_a_smiles", "C(C(C", None])
    def test_invalid_returns_false(self, smi) -> None:
        assert validate_smiles(smi) is False


class TestSafeCanonical:
    def test_none_returns_none(self) -> None:
        assert safe_canonical(None) is None

    def test_invalid_returns_none(self) -> None:
        # Tärkeä kontrasti canonical_smiles:iin: tämä EI heitä virhettä,
        # vaan palauttaa None — käytetään pandas-applyssä.
        assert safe_canonical("garbage123!!") is None

    def test_valid_returns_canonical(self) -> None:
        out = safe_canonical("OC(C)=O")
        assert out == canonical_smiles("OC(C)=O")
