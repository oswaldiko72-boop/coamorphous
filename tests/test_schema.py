"""Yksikkötestit Pandera-skeemalle.

MIKSI nämä testit ovat olemassa
-------------------------------
Skeema on hiljainen vahti: jos validointi joustaa, virheellistä dataa
voi päästä master-korpukseen. Testaamme:

* skeema rakentuu YAML:sta ilman virhettä,
* sarakejärjestys on YAML:n järjestyksessä,
* puuttuva pakollinen sarake hylätään,
* enum-rikkomus hylätään,
* numeerinen rajoitus (esim. mole_fraction <= 1) hylätään,
* tuntematon sarake hylätään (strict=True).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.errors as pa_errors
import pytest

from coamorphous.corpus.schema import build_schema, column_order, load_schema_yaml


@pytest.fixture
def schema(schema_yaml: Path):
    return build_schema(schema_yaml)


@pytest.fixture
def valid_row() -> dict:
    """Yksi täydellinen rivi, joka läpäisee validoinnin.

    Käytetään pohjana, jota mutatoidaan yksittäisiä virheitä testattaessa.
    """
    return {
        "pair_id": "test2026_01",
        "entry_id": "TestTable_row1",
        "drug_A_name": "Indometacin",
        "drug_B_name": "Naproxen",
        "drug_A_role": "api",
        "drug_B_role": "api",
        "drug_A_smiles_original": None,
        "drug_B_smiles_original": None,
        "drug_A_smiles_canonical": None,
        "drug_B_smiles_canonical": None,
        "drug_A_inchikey": None,
        "drug_B_inchikey": None,
        "drug_A_cas": None,
        "drug_B_cas": None,
        "mole_fraction_A": 0.5,
        "mole_fraction_B": 0.5,
        "mole_fraction_A_raw": 0.5,
        "mole_fraction_B_raw": 0.5,
        "weight_fraction_A": None,
        "weight_fraction_B": None,
        "ratio_reported_as": "mole_fraction",
        "process_method": "melt_quench",
        "process_details": None,
        "gfa_class_dsc": 3,
        "gfa_class_label": "Class III",
        "gfa_dsc_evidence": "dsc_cycle_full_reported",
        "gfa_label_confidence": "high",
        "stability_week_bin": "6-7m",
        "stability_protocol_match": "ich_q1a_accelerated",
        "stability_label_confidence": "high",
        "induction_time_days": 200.0,
        "induction_time_censored": False,
        "storage_T_C": 40.0,
        "storage_RH_percent": 75.0,
        "Tg_K": 333.0,
        "Tg_uncertainty_K": 1.0,
        "Tg_heating_rate_K_min": 10.0,
        "Tm_A_K": None,
        "Tm_B_K": None,
        "pxrd_amorphous": True,
        "detection_methods": "PXRD,DSC",
        "source_doi": "10.1234/example",
        "source_first_author": "Test",
        "source_year": 2026,
        "source_table_or_figure": "Table 2",
        "extraction_date": "2026-05-04",
        "extracted_by": "PH",
        "notes": None,
        "experimental_protocol": "ich_q1a_accelerated",
        "protocol_max_duration_days": 180.0,
        "storage_T_K": 313.15,
        "delta_MW": None,
        "delta_LogP": None,
        "delta_TPSA": None,
        "delta_HBD": None,
        "delta_HBA": None,
        "sum_MW": None,
        "tanimoto_ECFP4": None,
        "tanimoto_ECFP6": None,
        "hbond_complementarity": None,
    }


class TestSchemaConstruction:
    def test_schema_builds(self, schema_yaml: Path) -> None:
        schema = build_schema(schema_yaml)
        assert len(schema.columns) > 0

    def test_column_order_matches_yaml(self, schema_yaml: Path) -> None:
        spec = load_schema_yaml(schema_yaml)
        cols = column_order(spec)
        # Eka pitää olla pair_id (identiteetti aloittaa).
        assert cols[0] == "pair_id"
        # Pair-tason johdetut piirteet ovat lopussa.
        assert "tanimoto_ECFP4" in cols
        # Ei duplikaatteja.
        assert len(cols) == len(set(cols))

    def test_yaml_missing_columns_key_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("version: '1'\n", encoding="utf-8")
        with pytest.raises(ValueError, match="columns"):
            load_schema_yaml(bad)


class TestSchemaValidation:
    def test_valid_row_passes(self, schema, valid_row: dict) -> None:
        df = pd.DataFrame([valid_row])
        # Ei pidä heittää.
        schema.validate(df, lazy=True)

    def test_missing_required_column_fails(self, schema, valid_row: dict) -> None:
        # Poistetaan pakollinen pair_id — Pandera (strict + required) hylkää.
        bad_row = dict(valid_row)
        bad_row.pop("pair_id")
        df = pd.DataFrame([bad_row])
        with pytest.raises((pa_errors.SchemaError, pa_errors.SchemaErrors)):
            schema.validate(df, lazy=True)

    def test_invalid_enum_value_fails(self, schema, valid_row: dict) -> None:
        bad_row = dict(valid_row)
        bad_row["process_method"] = "spray-drying"  # YAML:ssa "spray_drying"
        df = pd.DataFrame([bad_row])
        with pytest.raises((pa_errors.SchemaError, pa_errors.SchemaErrors)):
            schema.validate(df, lazy=True)

    def test_mole_fraction_out_of_range_fails(self, schema, valid_row: dict) -> None:
        bad_row = dict(valid_row)
        bad_row["mole_fraction_A"] = 1.5  # > 1.0
        df = pd.DataFrame([bad_row])
        with pytest.raises((pa_errors.SchemaError, pa_errors.SchemaErrors)):
            schema.validate(df, lazy=True)

    def test_unknown_column_fails(self, schema, valid_row: dict) -> None:
        bad_row = dict(valid_row)
        bad_row["unexpected_extra"] = "something"
        df = pd.DataFrame([bad_row])
        with pytest.raises((pa_errors.SchemaError, pa_errors.SchemaErrors)):
            schema.validate(df, lazy=True)

    def test_negative_induction_time_fails(self, schema, valid_row: dict) -> None:
        bad_row = dict(valid_row)
        bad_row["induction_time_days"] = -5.0
        df = pd.DataFrame([bad_row])
        with pytest.raises((pa_errors.SchemaError, pa_errors.SchemaErrors)):
            schema.validate(df, lazy=True)

    def test_invalid_gfa_class_fails(self, schema, valid_row: dict) -> None:
        bad_row = dict(valid_row)
        bad_row["gfa_class_dsc"] = 4  # vain 1/2/3 sallittu
        df = pd.DataFrame([bad_row])
        with pytest.raises((pa_errors.SchemaError, pa_errors.SchemaErrors)):
            schema.validate(df, lazy=True)
