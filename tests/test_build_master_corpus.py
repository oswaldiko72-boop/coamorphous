"""Yksikkötestit ``tools/build_master_corpus.py`` -CLI-wrapperille.

MIKSI: Wrapper on tuotantopipelinen liittymä — jos se rikkoutuu, master-CSV ei
päivity. Testataan että:

* Default-polut resolvoidaan oikein (cwd-riippumattomuus).
* `--no-validate` ohittaa Panderan.
* Output-tiedosto luodaan ja sisältää kaikki interim-rivit.
* Tyhjässä interim-hakemistossa palautetaan virhekoodi siististi.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


# Lataa skripti dynaamisesti, koska tools/ ei ole package eikä ole sys.path:issa.
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_master_corpus.py"
_spec = importlib.util.spec_from_file_location("build_master_corpus", SCRIPT_PATH)
build_master_corpus = importlib.util.module_from_spec(_spec)
sys.modules["build_master_corpus"] = build_master_corpus
_spec.loader.exec_module(build_master_corpus)


@pytest.fixture
def project_root() -> Path:
    """Repon juuri (sama kuin tools/-skriptin oletus)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def minimal_interim_dir(tmp_path: Path, project_root: Path) -> Path:
    """Luo tilapäinen interim-hakemisto yhdellä rivillä, jolla skeema läpäisee.

    Käytämme oikeaa schema-YAML:ää, jotta validointipath testataan päästä
    päähän.
    """
    interim = tmp_path / "interim"
    interim.mkdir()

    # Yksi minimirivi, joka täyttää skeeman pakolliset kentät.
    row = {
        "pair_id": "test2026_01",
        "entry_id": "TestRow",
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
        "source_first_author": "test",
        "source_year": 2026,
        "source_table_or_figure": "Table 2",
        "extraction_date": "2026-05-04",
        "extracted_by": "PH",
        "notes": None,
        "experimental_protocol": "ich_q1a_accelerated",
        "protocol_max_duration_days": 180.0,
        "storage_T_K": 313.15,
    }
    pd.DataFrame([row]).to_csv(interim / "test_2026.csv", index=False)
    return interim


class TestBuildMasterCorpus:
    def test_writes_master_csv(
        self, tmp_path: Path, minimal_interim_dir: Path, project_root: Path
    ) -> None:
        output = tmp_path / "out" / "master.csv"
        rc = build_master_corpus.main(
            [
                "--interim-dir", str(minimal_interim_dir),
                "--output", str(output),
                "--schema", str(project_root / "configs" / "corpus_schema.yaml"),
            ]
        )
        assert rc == 0
        assert output.is_file()
        # Master-CSV kirjoitetaan ;-erottimella Excel-yhteensopivuuden takia.
        df = pd.read_csv(output, sep=";")
        assert len(df) == 1
        # Master-CSV sisältää myös merge-vaiheessa lisätyt sarakkeet.
        assert "mole_fraction_A_raw" in df.columns

    def test_creates_output_dir(
        self, tmp_path: Path, minimal_interim_dir: Path, project_root: Path
    ) -> None:
        # data/processed/ ei välttämättä ole olemassa — wrapperin pitää luoda.
        output = tmp_path / "newdir" / "subdir" / "master.csv"
        rc = build_master_corpus.main(
            [
                "--interim-dir", str(minimal_interim_dir),
                "--output", str(output),
                "--schema", str(project_root / "configs" / "corpus_schema.yaml"),
            ]
        )
        assert rc == 0
        assert output.parent.is_dir()
        assert output.is_file()

    def test_empty_interim_dir_returns_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_interim"
        empty.mkdir()
        rc = build_master_corpus.main(
            [
                "--interim-dir", str(empty),
                "--output", str(tmp_path / "out.csv"),
            ]
        )
        assert rc == 1
        assert not (tmp_path / "out.csv").exists()

    def test_missing_interim_dir_returns_error(self, tmp_path: Path) -> None:
        rc = build_master_corpus.main(
            [
                "--interim-dir", str(tmp_path / "nonexistent"),
                "--output", str(tmp_path / "out.csv"),
            ]
        )
        assert rc == 1

    def test_no_validate_flag_skips_pandera(
        self, tmp_path: Path, minimal_interim_dir: Path, project_root: Path
    ) -> None:
        # Riko validointi: lisää extra-sarake, jota skeema ei tunnista.
        bad_csv = minimal_interim_dir / "bad.csv"
        df = pd.read_csv(minimal_interim_dir / "test_2026.csv")
        df["unknown_extra_column"] = "foo"
        df["pair_id"] = "test2026_02"
        df.to_csv(bad_csv, index=False)

        # --no-validate ohittaa Panderan -> valmistuu ilman virhettä.
        output = tmp_path / "out.csv"
        rc = build_master_corpus.main(
            [
                "--interim-dir", str(minimal_interim_dir),
                "--output", str(output),
                "--schema", str(project_root / "configs" / "corpus_schema.yaml"),
                "--no-validate",
            ]
        )
        assert rc == 0
        assert output.is_file()
