"""Yksikkötestit GFA- ja stabiilisuusluokitukselle.

MIKSI nämä testit ovat olemassa
-------------------------------
Luokitusvirhe näkyy suoraan ML-mallin opetusdatassa. Aiemmin yksittäinen
``classify_baird_taylor``-funktio sekoitti GFA:n ja stabiilisuuden, mikä
teki testeistä vaikealukuisia ja peitti, kumpi ilmiö milläkin testillä
varmistettiin. Nyt testaamme kahdessa erillisessä polussa:

* ``classify_gfa_dsc`` — Baird et al. 2010 -luokitus DSC-syklistä.
* ``classify_stability_week_bin`` — induction_time_days -arvon binitys.
* ``classify_stability_protocol`` — säilytysolojen tunnistus.
* ``classify_stability_label_confidence`` — protokollasta johdettu luottamus.
"""

from __future__ import annotations

import pytest

from coamorphous.corpus.classify import (
    WEEK_BIN_OVER_YEAR,
    WEEK_BIN_UNKNOWN,
    classify_gfa_dsc,
    classify_stability_label_confidence,
    classify_stability_protocol,
    classify_stability_week_bin,
)


# =============================================================================
# GFA-luokitus (DSC-pohjainen)
# =============================================================================


class TestClassifyGfaDsc:
    """Baird et al. 2010 -luokitus DSC heating-cooling-reheating syklistä."""

    def test_class_i_crystallizes_on_cooling_full_cycle(self) -> None:
        # Kiteytyy jäähdytyksellä, uudelleenlämmityskin raportoitu (kiteytyy):
        # Class I high, evidence dsc_cycle_full_reported.
        gfa, conf, evidence = classify_gfa_dsc(
            crystallizes_on_cooling=True,
            crystallizes_on_reheating=True,
        )
        assert gfa == 1
        assert conf == "high"
        assert evidence == "dsc_cycle_full_reported"

    def test_class_i_cooling_only(self) -> None:
        # Vain jäähdytyssykli raportoitu, kiteytyy: Class I high,
        # evidence dsc_thermogram_inferred (yksittäisen termogrammin piirre).
        gfa, conf, evidence = classify_gfa_dsc(
            crystallizes_on_cooling=True,
            crystallizes_on_reheating=None,
        )
        assert gfa == 1
        assert conf == "high"
        assert evidence == "dsc_thermogram_inferred"

    def test_class_ii_no_cooling_yes_reheating(self) -> None:
        # Ei kiteydy jäähdytyksellä, kiteytyy uudelleenlämmityksessä: Class II.
        gfa, conf, evidence = classify_gfa_dsc(
            crystallizes_on_cooling=False,
            crystallizes_on_reheating=True,
        )
        assert gfa == 2
        assert conf == "high"
        assert evidence == "dsc_cycle_full_reported"

    def test_class_iii_no_cooling_no_reheating(self) -> None:
        # Ei kiteydy kummassakaan syklissä: Class III high.
        gfa, conf, evidence = classify_gfa_dsc(
            crystallizes_on_cooling=False,
            crystallizes_on_reheating=False,
        )
        assert gfa == 3
        assert conf == "high"
        assert evidence == "dsc_cycle_full_reported"

    def test_cooling_only_no_crystallization_uses_paper_class(self) -> None:
        # cooling=False, reheating=None: ei voi erottaa II/III ilman
        # uudelleenlämmityksen havaintoa. Käytetään paper_states_class
        # täydentävänä signaalina (matala luottamus, dsc_thermogram_inferred).
        gfa, conf, evidence = classify_gfa_dsc(
            crystallizes_on_cooling=False,
            crystallizes_on_reheating=None,
            paper_states_class=3,
        )
        assert gfa == 3
        assert conf == "low"
        assert evidence == "dsc_thermogram_inferred"

    def test_cooling_only_no_crystallization_no_paper_class(self) -> None:
        # cooling=False, reheating=None, paper_states_class=None:
        # palautetaan None / unknown — luokkaa ei voi määrittää.
        gfa, conf, evidence = classify_gfa_dsc(
            crystallizes_on_cooling=False,
            crystallizes_on_reheating=None,
            paper_states_class=None,
        )
        assert gfa is None
        assert conf == "unknown"
        assert evidence == "dsc_thermogram_inferred"

    def test_paper_classification_only(self) -> None:
        # Ei DSC-tietoa, vain artikkelin eksplisiittinen maininta:
        # luotetaan tähän korkealla confidencellä, evidence stated_explicitly.
        gfa, conf, evidence = classify_gfa_dsc(
            crystallizes_on_cooling=None,
            crystallizes_on_reheating=None,
            paper_states_class=2,
        )
        assert gfa == 2
        assert conf == "high"
        assert evidence == "stated_explicitly"

    def test_no_data_returns_none(self) -> None:
        # Ei mitään tietoa: (None, 'unknown', 'not_reported').
        gfa, conf, evidence = classify_gfa_dsc(
            crystallizes_on_cooling=None,
            crystallizes_on_reheating=None,
            paper_states_class=None,
        )
        assert gfa is None
        assert conf == "unknown"
        assert evidence == "not_reported"

    def test_class_i_overrides_paper_states(self) -> None:
        # Suora DSC-havainto ohittaa artikkelin ilmoituksen, jos ne ovat
        # ristiriidassa.
        gfa, conf, evidence = classify_gfa_dsc(
            crystallizes_on_cooling=True,
            crystallizes_on_reheating=True,
            paper_states_class=3,  # ristiriita: artikkeli väittää Class III
        )
        assert gfa == 1
        assert conf == "high"
        assert evidence == "dsc_cycle_full_reported"


# =============================================================================
# Stabiilisuusajan binitys
# =============================================================================


class TestClassifyStabilityWeekBin:
    """induction_time_days -> diskreetti viikko-/kuukausibini."""

    @pytest.mark.parametrize(
        ("days", "expected"),
        [
            (0.0, "<1w"),
            (3.0, "<1w"),
            (6.99, "<1w"),
            (7.0, "1-2w"),
            (10.0, "1-2w"),
            (13.99, "1-2w"),
            (14.0, "2-3w"),
            (21.0, "3-4w"),
            (27.99, "3-4w"),
            (28.0, "1-2m"),
            (45.0, "1-2m"),
            (60.0, "2-3m"),
            (75.0, "2-3m"),
            (90.0, "3-4m"),
            (120.0, "4-5m"),
            (150.0, "5-6m"),
            (180.0, "6-7m"),
            (186.0, "6-7m"),
            (210.0, "7-8m"),
            (240.0, "8-9m"),
            (270.0, "9-10m"),
            (300.0, "10-11m"),
            (335.0, "11-12m"),
            (364.99, "11-12m"),
        ],
    )
    def test_bin_assignment(self, days: float, expected: str) -> None:
        assert classify_stability_week_bin(days) == expected

    def test_at_or_above_year_returns_over_year(self) -> None:
        assert classify_stability_week_bin(365.0) == WEEK_BIN_OVER_YEAR
        assert classify_stability_week_bin(500.0) == WEEK_BIN_OVER_YEAR
        assert classify_stability_week_bin(2000.0) == WEEK_BIN_OVER_YEAR

    def test_none_returns_unknown(self) -> None:
        assert classify_stability_week_bin(None) == WEEK_BIN_UNKNOWN

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            classify_stability_week_bin(-1.0)

    def test_censored_does_not_change_bin(self) -> None:
        # Censored-status pidetään erillisessä CSV-sarakkeessa eikä se
        # vaikuta itse bin-arvoon.
        assert (
            classify_stability_week_bin(186.0, induction_time_censored=True)
            == "6-7m"
        )
        assert (
            classify_stability_week_bin(186.0, induction_time_censored=False)
            == "6-7m"
        )


# =============================================================================
# Stabiilisuusprotokollan tunnistus
# =============================================================================


class TestClassifyStabilityProtocol:
    """storage_T_C/storage_RH_percent + experimental_protocol -> protocol_match."""

    def test_ich_q1a_accelerated_exact(self) -> None:
        assert classify_stability_protocol(40.0, 75.0) == "ich_q1a_accelerated"

    def test_ich_q1a_accelerated_loose_tolerance(self) -> None:
        # 38 °C / 73 % RH -> edelleen kiihdytetty (väljä toleranssi).
        assert classify_stability_protocol(38.0, 73.0) == "ich_q1a_accelerated"
        assert classify_stability_protocol(45.0, 80.0) == "ich_q1a_accelerated"

    def test_ich_q1a_long_term_exact(self) -> None:
        assert classify_stability_protocol(25.0, 60.0) == "ich_q1a_long_term"

    def test_dry_silica_gel_is_non_standard(self) -> None:
        # Kuiva (RH ≈ 0 %) ilman eksplisiittistä dry_short_term-merkkiä
        # luokitellaan non_standardiksi — kuivakokeen kesto ja tulkinta
        # vaihtelee, joten ekstraktoijan on oltava eksplisiittinen.
        assert classify_stability_protocol(4.0, 0.0) == "non_standard"
        assert classify_stability_protocol(25.0, 0.0) == "non_standard"

    def test_explicit_protocol_overrides(self) -> None:
        # Eksplisiittinen experimental_protocol ohittaa olojen päättelyn.
        assert (
            classify_stability_protocol(
                4.0, 0.0, experimental_protocol="dry_short_term"
            )
            == "dry_short_term"
        )
        # Vaikka olot olisivat 40/75, eksplisiittinen non_standard pysyy.
        assert (
            classify_stability_protocol(
                40.0, 75.0, experimental_protocol="non_standard"
            )
            == "non_standard"
        )

    def test_missing_conditions_with_protocol(self) -> None:
        assert (
            classify_stability_protocol(
                None, None, experimental_protocol="tg_plus_15K"
            )
            == "tg_plus_15K"
        )

    def test_missing_conditions_without_protocol(self) -> None:
        assert classify_stability_protocol(None, None) == "non_standard"

    def test_unknown_protocol_string_falls_back_to_conditions(self) -> None:
        # Tuntematon protokollamerkki ei kuulu enumiin -> päätellään oloista.
        assert (
            classify_stability_protocol(
                40.0, 75.0, experimental_protocol="not_in_enum"
            )
            == "ich_q1a_accelerated"
        )

    def test_intermediate_conditions_are_non_standard(self) -> None:
        # 33 °C / 50 % RH on kummankin ICH-standardin ulkopuolella
        # (long-term: T<=30 ja RH 55-65; accelerated: T>=35 ja RH 70-80)
        # -> non_standard.
        assert classify_stability_protocol(33.0, 50.0) == "non_standard"


# =============================================================================
# Stabiilisuusluokituksen luotettavuus
# =============================================================================


class TestClassifyStabilityLabelConfidence:
    """ICH Q1A -protokollat -> 'high', muut -> 'low'."""

    @pytest.mark.parametrize(
        ("protocol_match", "expected"),
        [
            ("ich_q1a_accelerated", "high"),
            ("ich_q1a_long_term", "high"),
            ("dry_short_term", "low"),
            ("tg_plus_15K", "low"),
            ("dsc_in_situ", "low"),
            ("non_standard", "low"),
        ],
    )
    def test_confidence_mapping(self, protocol_match: str, expected: str) -> None:
        assert (
            classify_stability_label_confidence(protocol_match=protocol_match)
            == expected
        )

    def test_extra_experimental_protocol_is_ignored(self) -> None:
        # Funktio dokumentoi, että experimental_protocol-parametri otetaan
        # vastaan mutta päätös tehdään protocol_match-arvon perusteella.
        assert (
            classify_stability_label_confidence(
                protocol_match="ich_q1a_accelerated",
                experimental_protocol="non_standard",
            )
            == "high"
        )
