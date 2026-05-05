"""Yksikkötestit ekstraktiopipelinen moduuleille.

MIKSI nämä testit ovat olemassa
-------------------------------
Ekstraktiopipeline yhdistää useita ulkoisia järjestelmiä (PubChem REST,
RDKit, Pandera-skeema, classify_gfa_dsc + classify_stability_*). Yksikkötestit:

* varmistavat, että rajapintamuoto pysyy yhtenäisenä,
* mockaavat PubChem:n niin että testit ajetaan offline ja ne ovat nopeita,
* todistavat että luokitus virtaa oikein ``RawPair`` -> master-rivi,
* takaavat, että validointi havaitsee tyypilliset ongelmat.

Mockauksen perustelu
--------------------
PubChem PUG REST on hidas ja sen saatavuus voi vaihdella. Käytämme
``unittest.mock.patch``:lla ``requests.get``:in monkeypatchausta, jolloin
testit ajavat ilman verkkoyhteyttä ja ovat deterministisiä.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from coamorphous.extraction.enrich import (
    assign_pair_id,
    canonicalize_pair,
    compute_classification,
    raw_pair_to_master_row,
)
from coamorphous.extraction.extraction_schema import RawPair
from coamorphous.extraction.pubchem_lookup import (
    PubChemError,
    normalize_drug_name,
    search_by_name,
    with_retry,
)
from coamorphous.extraction.validate import (
    check_mole_fractions_sum,
    check_smiles_validity,
    check_storage_consistency,
    check_weight_fractions_sum,
    run_all_validations,
)


# -----------------------------------------------------------------------------
# Mock-vastaukset
# -----------------------------------------------------------------------------
# MIKSI dict-pohjaiset mockit eikä HTTP-mock-palvelin: PubChem-kutsut menevät
# kahdessa vaiheessa (CID-haku, sitten ominaisuudet), ja molemmille on oma
# vastaus. Pidämme ne nimettyinä vakioina, jotta testit lukijalle ovat
# luettavia ja muutosten vaikutus näkyy yhdessä paikassa.

NAPROXEN_CID = 156391
# PubChem palauttaa isomerisen SMILES:n (sisältää stereokemian). Naproxen
# on (S)-enantiomeerina markkinoilla, joten @@H-merkintä @H-merkinnän
# sijaan. InChIKey poikkeaa tästä syystä racemic-versiosta.
NAPROXEN_INCHIKEY = "CMWTZPSULFXXJA-VIFPVBQESA-N"
NAPROXEN_SMILES = "C[C@@H](C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O"
NAPROXEN_MW = 230.26

INDOMETHACIN_CID = 3715
INDOMETHACIN_INCHIKEY = "CGIGDMFJXJATDK-UHFFFAOYSA-N"
INDOMETHACIN_SMILES = "Cc1c(CC(=O)O)c2cc(OC)ccc2n1C(=O)c1ccc(Cl)cc1"
INDOMETHACIN_MW = 357.79


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Luo MagicMock, joka käyttäytyy kuten ``requests.Response``."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


def _pubchem_cid_payload(cid: int) -> dict:
    """PubChem PUG REST: CID-haku-vastaus."""
    return {"IdentifierList": {"CID": [cid]}}


def _pubchem_props_payload(cid: int, smiles: str, inchikey: str, mw: float) -> dict:
    """PubChem PUG REST: ominaisuus-haku-vastaus.

    Avain on 'SMILES' (ei 'CanonicalSMILES'); ks. pubchem_lookup-moduulin
    docstring. Mock-vastauksen muoto vastaa nykyistä PUG REST -palautusta.
    """
    return {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": cid,
                    "SMILES": smiles,
                    "InChIKey": inchikey,
                    "MolecularWeight": str(mw),
                }
            ]
        }
    }


def _make_pubchem_router() -> Any:
    """Tee `requests.get` -korvaaja, joka reitittää URL:n perusteella.

    PubChem-haku tehdään kahdella URL:lla: ``/compound/name/{name}/cids/JSON``
    ja ``/compound/cid/{cid}/property/.../JSON``. Tämä reitittäjä emuloi
    molemmat ja tukee tunnetut nimet (naproxen, indomethacin).
    """
    name_to_cid = {
        "naproxen": NAPROXEN_CID,
        "indomethacin": INDOMETHACIN_CID,
    }
    cid_to_props = {
        NAPROXEN_CID: (NAPROXEN_SMILES, NAPROXEN_INCHIKEY, NAPROXEN_MW),
        INDOMETHACIN_CID: (INDOMETHACIN_SMILES, INDOMETHACIN_INCHIKEY, INDOMETHACIN_MW),
    }

    def fake_get(url: str, timeout: float | None = None, **_: Any) -> MagicMock:
        # Nimi-haku
        for name, cid in name_to_cid.items():
            if f"/compound/name/{name}/cids/JSON" in url:
                return _mock_response(_pubchem_cid_payload(cid))
        # Ominaisuushaku
        for cid, (smi, key, mw) in cid_to_props.items():
            if f"/compound/cid/{cid}/property/" in url:
                return _mock_response(_pubchem_props_payload(cid, smi, key, mw))
        # Tuntematon nimi -> 404 (PubChem-konventio)
        return _mock_response({}, status_code=404)

    return fake_get


# -----------------------------------------------------------------------------
# pubchem_lookup
# -----------------------------------------------------------------------------


class TestNormalizeDrugName:
    def test_lowercases_and_strips(self) -> None:
        assert normalize_drug_name("  Naproxen  ") == "naproxen"

    def test_removes_salt_suffix(self) -> None:
        assert normalize_drug_name("Naproxen Sodium") == "naproxen"

    def test_replaces_greek_gamma(self) -> None:
        assert normalize_drug_name("γ-Indomethacin") == "gamma-indomethacin"

    def test_handles_multiple_salts(self) -> None:
        # "indomethacin hydrochloride hcl" -> molemmat suolaliitteet pois
        assert normalize_drug_name("indomethacin hydrochloride hcl") == "indomethacin"

    def test_empty_input(self) -> None:
        assert normalize_drug_name("") == ""


class TestSearchByName:
    def test_search_by_name_naproxen(self) -> None:
        with patch("coamorphous.extraction.pubchem_lookup.requests.get",
                   side_effect=_make_pubchem_router()):
            result = search_by_name("naproxen")

        assert result is not None
        assert result["cid"] == NAPROXEN_CID
        assert result["inchikey"] == NAPROXEN_INCHIKEY
        assert result["smiles"] == NAPROXEN_SMILES
        assert result["mw"] == pytest.approx(NAPROXEN_MW)

    def test_search_by_name_unknown_returns_none(self) -> None:
        with patch("coamorphous.extraction.pubchem_lookup.requests.get",
                   side_effect=_make_pubchem_router()):
            result = search_by_name("not_a_real_drug_xyz")
        assert result is None


class TestWithRetry:
    def test_retries_on_request_exception(self) -> None:
        # Funktio epäonnistuu kerran, sitten onnistuu.
        import requests as req

        attempts: list[int] = []

        def flaky() -> str:
            attempts.append(1)
            if len(attempts) < 2:
                raise req.ConnectionError("simulated")
            return "OK"

        # backoff=0 jotta testi ei jää odottamaan oikeasti.
        result = with_retry(flaky, max_retries=3, backoff=0.0)
        assert result == "OK"
        assert len(attempts) == 2

    def test_raises_pubchem_error_after_exhausting_retries(self) -> None:
        import requests as req

        def always_fails() -> str:
            raise req.ConnectionError("dead")

        with pytest.raises(PubChemError):
            with_retry(always_fails, max_retries=2, backoff=0.0)


# -----------------------------------------------------------------------------
# validate
# -----------------------------------------------------------------------------


class TestValidationFunctions:
    def test_mole_fraction_sum_passes(self) -> None:
        assert check_mole_fractions_sum({"mole_fraction_A": 0.5, "mole_fraction_B": 0.5}) is None

    def test_mole_fraction_sum_fails(self) -> None:
        msg = check_mole_fractions_sum({"mole_fraction_A": 0.5, "mole_fraction_B": 0.4})
        assert msg is not None
        assert "0.9" in msg or "0.90" in msg

    def test_mole_fraction_sum_skips_when_missing(self) -> None:
        # Jompikumpi None -> None palautus, koska weight_fraction voi olla
        # ainoa raportoitu suhde.
        assert check_mole_fractions_sum({"mole_fraction_A": None, "mole_fraction_B": 0.5}) is None

    def test_weight_fraction_sum_within_tolerance(self) -> None:
        # 0.392 + 0.608 = 1.000 (Löbmann _03 weight)
        assert check_weight_fractions_sum(
            {"weight_fraction_A": 0.392, "weight_fraction_B": 0.608}
        ) is None

    def test_smiles_validity_invalid_canonical(self) -> None:
        errors = check_smiles_validity({
            "drug_A_smiles_canonical": "this_is_not_a_smiles",
            "drug_B_smiles_canonical": NAPROXEN_SMILES,
        })
        assert len(errors) == 1
        assert "drug_A_smiles_canonical" in errors[0]

    def test_storage_consistency_ich_accelerated_ok(self) -> None:
        assert check_storage_consistency({
            "experimental_protocol": "ich_q1a_accelerated",
            "storage_T_C": 40.0,
            "storage_RH_percent": 75.0,
        }) is None

    def test_storage_consistency_ich_accelerated_wrong_T(self) -> None:
        msg = check_storage_consistency({
            "experimental_protocol": "ich_q1a_accelerated",
            "storage_T_C": 25.0,  # liian alhainen kiihdytetylle
            "storage_RH_percent": 75.0,
        })
        assert msg is not None
        assert "storage_T_C" in msg

    def test_storage_consistency_dry_short_term_skipped(self) -> None:
        # dry_short_term-protokolla ei kuulu tarkistuksen piiriin: RH ≈ 0 %
        # on hyväksytty.
        assert check_storage_consistency({
            "experimental_protocol": "dry_short_term",
            "storage_T_C": 25.0,
            "storage_RH_percent": 0.0,
        }) is None

    def test_run_all_validations_collects(self) -> None:
        # Kaksi virhettä: mole-summa ja virheellinen SMILES.
        bad = {
            "mole_fraction_A": 0.3,
            "mole_fraction_B": 0.5,
            "drug_A_smiles_canonical": "junk",
            "drug_B_smiles_canonical": NAPROXEN_SMILES,
        }
        errors = run_all_validations(bad)
        assert len(errors) >= 2


# -----------------------------------------------------------------------------
# RawPair-mallin validaattorit (Pydantic-tason tarkistukset)
# -----------------------------------------------------------------------------


class TestRawPairValidators:
    """Pydantic-mallin model_validator-tarkistukset.

    MIKSI erillinen kerros vs. validate.py:n dict-validaattorit:
    Pydantic-tarkistukset estävät virheellisen objektin luomisen jo
    konstruktorissa, jolloin alavirran koodi (enrich, descriptors) saa
    aina taatusti konsistentin RawPair-instanssin. validate.py:n funktiot
    toimivat samanaikaisesti dict-tasolla raportoivina varmistuksina.
    """

    def _base_kwargs(self, **overrides: Any) -> dict:
        defaults = {
            "drug_A_name_raw": "Simvastatin",
            "drug_B_name_raw": "Glipizide",
            "drug_A_role": "api",
            "drug_B_role": "api",
            "ratio_source_quote": "1:1 mol",
            "source_table_or_figure": "Table 3",
            "source_quote": "Test source quote.",
        }
        defaults.update(overrides)
        return defaults

    def test_weight_fractions_sum_to_one_passes(self) -> None:
        # Löbmann 2012 1:1 SVS:GPZ -> w_A=0.4844, w_B=0.5156, sum=1.0
        raw = RawPair(
            **self._base_kwargs(
                weight_fraction_A=0.4844,
                weight_fraction_B=0.5156,
            )
        )
        assert raw.weight_fraction_A == 0.4844

    def test_weight_fractions_within_tolerance_passes(self) -> None:
        # 0.001 ero hyväksytään (sallittu pyöristysvirhe).
        raw = RawPair(
            **self._base_kwargs(
                weight_fraction_A=0.500,
                weight_fraction_B=0.501,
            )
        )
        assert raw.weight_fraction_B == 0.501

    def test_weight_fractions_sum_too_large_fails(self) -> None:
        # 0.6 + 0.5 = 1.1 -> > 0.01 toleranssi -> ValidationError.
        with pytest.raises(ValueError, match="weight_fraction"):
            RawPair(
                **self._base_kwargs(
                    weight_fraction_A=0.6,
                    weight_fraction_B=0.5,
                )
            )

    def test_weight_fractions_sum_too_small_fails(self) -> None:
        # 0.3 + 0.4 = 0.7 -> kaukana 1.0:sta -> ValidationError.
        with pytest.raises(ValueError, match="weight_fraction"):
            RawPair(
                **self._base_kwargs(
                    weight_fraction_A=0.3,
                    weight_fraction_B=0.4,
                )
            )

    def test_only_one_weight_fraction_skips_check(self) -> None:
        # Vain toinen weight_fraction annettu -> tarkistus ohitetaan.
        # Mooliosuuden kautta täytetään "vähintään yksi pari" -vaatimus.
        raw = RawPair(
            **self._base_kwargs(
                mole_fraction_A=0.5,
                mole_fraction_B=0.5,
                weight_fraction_A=0.999,  # B puuttuu -> ei summavalidointia
            )
        )
        assert raw.weight_fraction_A == 0.999
        assert raw.weight_fraction_B is None


# -----------------------------------------------------------------------------
# enrich
# -----------------------------------------------------------------------------


class TestCanonicalizeViaEnrich:
    def test_canonicalize_pair_handles_valid_smiles(self) -> None:
        out = canonicalize_pair({
            "drug_A_smiles_original": NAPROXEN_SMILES,
            "drug_B_smiles_original": INDOMETHACIN_SMILES,
        })
        # Kanonisointi RDKit:llä voi tuottaa eri merkkijonon, mutta sen pitää
        # olla ei-tyhjä ja InChIKey:n täytyy täsmätä alkuperäiseen.
        assert out["drug_A_smiles_canonical"] is not None
        assert out["drug_B_smiles_canonical"] is not None
        assert out["drug_A_inchikey"] == NAPROXEN_INCHIKEY
        assert out["drug_B_inchikey"] == INDOMETHACIN_INCHIKEY

    def test_canonicalize_pair_handles_none(self) -> None:
        out = canonicalize_pair({
            "drug_A_smiles_original": None,
            "drug_B_smiles_original": INDOMETHACIN_SMILES,
        })
        assert out["drug_A_smiles_canonical"] is None
        assert out["drug_A_inchikey"] is None
        assert out["drug_B_inchikey"] == INDOMETHACIN_INCHIKEY


class TestComputeClassificationViaEnrich:
    """Varmista, että compute_classification kutsuu uudet luokitusfunktiot
    oikeilla parametreilla ja palauttaa kaikki kuusi kohdesarakkeen arvoa."""

    def _make_raw(self, **overrides: Any) -> RawPair:
        # MIKSI: rakenna minimi-validi RawPair, jota testit voivat säätää.
        # ratio_source_quote ja source_quote ovat pakollisia kenttiä.
        defaults = {
            "drug_A_name_raw": "Naproxen",
            "drug_B_name_raw": "Indomethacin",
            "drug_A_role": "api",
            "drug_B_role": "api",
            "mole_fraction_A": 0.5,
            "mole_fraction_B": 0.5,
            "ratio_source_quote": "1:1 mol",
            "source_table_or_figure": "Table 1",
            "source_quote": "Test source quote.",
        }
        defaults.update(overrides)
        return RawPair(**defaults)

    def test_classify_dry_short_term_returns_full_dict(self) -> None:
        raw = self._make_raw(
            induction_time_days=21.0,
            induction_time_censored=True,
            storage_T_C=25.0,
            storage_RH_percent=0.0,
            experimental_protocol="dry_short_term",
            protocol_max_duration_days=21.0,
            crystallizes_on_dsc_cooling=False,
            crystallizes_on_dsc_reheating=False,
        )
        result = compute_classification(raw)
        # GFA-luokitus täydestä DSC-syklistä: ei kiteydy kummassakaan -> Class III.
        assert result["gfa_class_dsc"] == 3
        assert result["gfa_label_confidence"] == "high"
        assert result["gfa_dsc_evidence"] == "dsc_cycle_full_reported"
        # Stabiilisuus: 21 vrk = 3-4w bin, dry_short_term -> low.
        assert result["stability_week_bin"] == "3-4w"
        assert result["stability_protocol_match"] == "dry_short_term"
        assert result["stability_label_confidence"] == "low"

    def test_classify_passes_protocol_through(self) -> None:
        # ICH Q1A 40/75, 60 vrk -> stability 2-3m bin, high confidence.
        raw = self._make_raw(
            induction_time_days=60.0,
            storage_T_C=40.0,
            storage_RH_percent=75.0,
            experimental_protocol="ich_q1a_accelerated",
            crystallizes_on_dsc_cooling=False,
            induction_time_censored=False,
        )
        result = compute_classification(raw)
        assert result["stability_week_bin"] == "2-3m"
        assert result["stability_protocol_match"] == "ich_q1a_accelerated"
        assert result["stability_label_confidence"] == "high"
        # GFA: vain cooling=False, reheating=None, ei paper_states_class
        # -> (None, 'unknown', 'dsc_thermogram_inferred').
        assert result["gfa_class_dsc"] is None
        assert result["gfa_label_confidence"] == "unknown"
        assert result["gfa_dsc_evidence"] == "dsc_thermogram_inferred"

    def test_paper_classification_falls_through(self) -> None:
        # Ei DSC-tietoa, vain artikkelin eksplisiittinen maininta:
        # luotetaan korkealla luottamuksella, evidence stated_explicitly.
        raw = self._make_raw(
            induction_time_days=None,
            paper_states_gfa_class=2,
            storage_T_C=25.0,
            storage_RH_percent=60.0,
            experimental_protocol="ich_q1a_long_term",
            needs_review=True,  # induktioajan puuttumisen vuoksi
        )
        result = compute_classification(raw)
        assert result["gfa_class_dsc"] == 2
        assert result["gfa_label_confidence"] == "high"
        assert result["gfa_dsc_evidence"] == "stated_explicitly"
        assert result["stability_week_bin"] == "unknown"


class TestRawPairToMasterRow:
    """Päästä-päähän -testi: RawPair + mock PubChem -> master-rivi."""

    def _base_raw(self, **overrides: Any) -> RawPair:
        defaults = {
            "drug_A_name_raw": "Naproxen",
            "drug_B_name_raw": "Indomethacin",
            "drug_A_role": "api",
            "drug_B_role": "api",
            "mole_fraction_A": 0.5,
            "mole_fraction_B": 0.5,
            "ratio_source_quote": "1:1 mol",
            "source_table_or_figure": "Table 1",
            "source_quote": "We prepared 1:1 NAP:IND mixtures by quench cooling.",
            "process_method": "quench_cooling",
            "induction_time_days": 21.0,
            "induction_time_censored": True,
            "storage_T_C": 25.0,
            "storage_RH_percent": 0.0,
            "experimental_protocol": "dry_short_term",
            "protocol_max_duration_days": 21.0,
            "crystallizes_on_dsc_cooling": False,
            "crystallizes_on_dsc_reheating": False,
            "pxrd_amorphous": True,
        }
        defaults.update(overrides)
        return RawPair(**defaults)

    def test_dry_short_term_pipeline_yields_class_iii(self) -> None:
        raw = self._base_raw()
        meta = {
            "source_doi": "10.1021/mp2002973",
            "source_first_author": "lobmann",
            "source_year": 2011,
            "extraction_date": "2026-05-04",
        }
        with patch("coamorphous.extraction.pubchem_lookup.requests.get",
                   side_effect=_make_pubchem_router()):
            row = raw_pair_to_master_row(raw, meta, extracted_by="claude_code")

        # GFA-luokitus täydestä DSC-syklistä (cooling=False, reheating=False).
        assert row["gfa_class_dsc"] == 3
        assert row["gfa_class_label"] == "Class III"
        assert row["gfa_label_confidence"] == "high"
        assert row["gfa_dsc_evidence"] == "dsc_cycle_full_reported"
        # Stabiilisuusluokitus: 21 vrk = 3-4w, dry_short_term -> low.
        assert row["stability_week_bin"] == "3-4w"
        assert row["stability_protocol_match"] == "dry_short_term"
        assert row["stability_label_confidence"] == "low"

        # PubChem-osumat populoivat SMILES + InChIKey.
        assert row["drug_A_inchikey"] == NAPROXEN_INCHIKEY
        assert row["drug_B_inchikey"] == INDOMETHACIN_INCHIKEY
        assert row["drug_A_smiles_canonical"] is not None
        assert row["drug_B_smiles_canonical"] is not None

        # Lähdemetadata virtaa läpi.
        assert row["source_doi"] == "10.1021/mp2002973"
        assert row["source_first_author"] == "lobmann"
        assert row["extraction_date"] == "2026-05-04"
        assert row["extracted_by"] == "claude_code"

        # storage_T_K lasketaan T_C:stä.
        assert row["storage_T_K"] == pytest.approx(25.0 + 273.15)

        # pair_id annetaan myöhemmin, ei rivin ekstraktiossa.
        assert row["pair_id"] is None

    def test_pubchem_failure_marks_review(self) -> None:
        # Käytetään tuntematonta nimeä -> mock palauttaa 404 -> rivi merkitään.
        raw = self._base_raw(drug_A_name_raw="not_a_real_drug_xyz")
        meta = {"source_doi": "x", "source_first_author": "x", "source_year": 2020}
        with patch("coamorphous.extraction.pubchem_lookup.requests.get",
                   side_effect=_make_pubchem_router()):
            row = raw_pair_to_master_row(raw, meta)

        # SMILES ei löytynyt drug_A:lle.
        assert row["drug_A_smiles_canonical"] is None
        # Notes-kentässä on review-merkintä.
        assert row["notes"] is not None
        assert "[review]" in row["notes"]


class TestAssignPairId:
    def test_sequential_numbering(self) -> None:
        rows = [{"pair_id": None}, {"pair_id": None}, {"pair_id": None}]
        assign_pair_id(rows, "lobmann", 2011)
        assert rows[0]["pair_id"] == "lobmann2011_01"
        assert rows[1]["pair_id"] == "lobmann2011_02"
        assert rows[2]["pair_id"] == "lobmann2011_03"

    def test_first_author_normalized(self) -> None:
        rows = [{"pair_id": None}]
        assign_pair_id(rows, "Van Den Berg", 2024)
        # Välilyönnit poistetaan, lowercase
        assert rows[0]["pair_id"] == "vandenberg2024_01"
