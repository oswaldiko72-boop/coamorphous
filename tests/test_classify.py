"""Yksikkötestit Baird-Taylor luokitukselle.

MIKSI nämä testit ovat olemassa
-------------------------------
Luokitusvirhe näkyy suoraan ML-mallin opetusdatassa. Testaamme:

* selvät tapaukset (Class I/II/III)
* DSC-jäähdytyksen kiteytyminen pakottaa Class I:n
* raja-arvot (7 vrk, 180 vrk, 365 vrk) ja niiden borderline-vyöhykkeet
  *molemmissa* suunnissa (162 vrk on borderline yhtä lailla kuin 195 vrk)
* sensuroinnin (``induction_time_censored``) vaikutus rajalla
* puuttuva data palauttaa ``None`` luokan
* negatiivinen induktioaika nostaa virheen (sanity check)
"""

from __future__ import annotations

import pytest

from coamorphous.corpus.classify import (
    BORDERLINE_FRACTION,
    CLASS_I_MAX_DAYS,
    CLASS_II_MAX_DAYS_ACCELERATED,
    CLASS_III_MIN_DAYS_AMBIENT,
    classify_baird_taylor,
)


class TestDSCRule:
    """DSC-jäähdytyksen kiteytyminen on ehdoton Class I -indikaattori."""

    def test_dsc_crystallizes_forces_class_i(self) -> None:
        gfa, conf = classify_baird_taylor(
            induction_time_days=200.0,  # pitkä, mutta ei merkkaa
            storage_T_C=40.0,
            storage_RH_percent=75.0,
            crystallizes_on_dsc_cooling=True,
        )
        assert gfa == 1
        assert conf == "high"


class TestClassI:
    """< 7 vrk = Class I."""

    def test_well_below_threshold(self) -> None:
        gfa, conf = classify_baird_taylor(3.0, 40.0, 75.0, False)
        assert gfa == 1
        assert conf == "high"

    def test_just_below_threshold_is_borderline(self) -> None:
        # 6.5 vrk on borderline (7 - 0.7 = 6.3, eli rajan 7 ympärillä ±10%).
        gfa, conf = classify_baird_taylor(6.5, 40.0, 75.0, False)
        assert gfa == 1
        assert conf == "borderline"


class TestClassIIAccelerated:
    """7-180 vrk kiihdytetyssä (40 °C / 75 % RH) = Class II."""

    def test_middle_of_range(self) -> None:
        gfa, conf = classify_baird_taylor(60.0, 40.0, 75.0, False)
        assert gfa == 2
        assert conf == "high"

    def test_just_above_class_i_threshold_is_borderline(self) -> None:
        # 7.5 vrk on borderline 7 vrk:n rajalla.
        gfa, conf = classify_baird_taylor(7.5, 40.0, 75.0, False)
        assert gfa == 2
        assert conf == "borderline"

    def test_162_days_is_borderline_near_class_iii_threshold(self) -> None:
        # MIKSI tämä testi: 162 vrk = 180 - 18 on tasan ±10 % alapuolella
        # Class III -kynnystä. Aiempi koodi olisi luokitellut tämän
        # (2, 'high'), koska borderline-tarkistus tehtiin vain Class I -rajan
        # ympärillä. Korjattu logiikka tunnistaa molemmat rajat.
        gfa, conf = classify_baird_taylor(162.0, 40.0, 75.0, False)
        assert gfa == 2
        assert conf == "borderline"


class TestClassIIIAccelerated:
    """>= 180 vrk kiihdytetyssä = Class III (sensurointiriippuvainen rajalla)."""

    def test_well_above_threshold(self) -> None:
        gfa, conf = classify_baird_taylor(365.0, 40.0, 75.0, False)
        assert gfa == 3
        assert conf == "high"

    def test_exactly_at_threshold_default_censoring_is_class_iii_borderline(self) -> None:
        # Sensurointitietoa ei anneta (None) -> säilytetään aiempi semantiikka:
        # 180 vrk on Class III borderline.
        gfa, conf = classify_baird_taylor(180.0, 40.0, 75.0, False)
        assert gfa == 3
        assert conf == "borderline"

    def test_six_and_half_months_borderline(self) -> None:
        # 6.5 kk = 195 vrk on borderline rajalla 180 (180*1.10 = 198).
        gfa, conf = classify_baird_taylor(195.0, 40.0, 75.0, False)
        assert gfa == 3
        assert conf == "borderline"

    def test_well_clear_of_borderline(self) -> None:
        # 250 vrk on selvästi borderline-vyöhykkeen ulkopuolella (180*1.10=198).
        gfa, conf = classify_baird_taylor(250.0, 40.0, 75.0, False)
        assert gfa == 3
        assert conf == "high"

    def test_180_days_censored_true_is_class_iii_high(self) -> None:
        # MIKSI: censored=True ja t = 180 vrk tarkoittaa, että koe kesti
        # tasan 180 vrk *ilman* havaittua kiteytymistä. Tämä on suora
        # todiste Class III -kriteeristä ("ei kiteytymistä >= 180 vrk")
        # -> high confidence rajan tarkkuudesta huolimatta.
        gfa, conf = classify_baird_taylor(
            180.0, 40.0, 75.0, False, induction_time_censored=True
        )
        assert gfa == 3
        assert conf == "high"

    def test_180_days_censored_false_is_class_ii_borderline(self) -> None:
        # PERUSTELU: censored=False ja t = 180 vrk tarkoittaa, että
        # kiteytyminen *havaittiin* tasan 180 vrk:n kohdalla. Class III
        # vaatii "ei kiteytymistä >= 180 vrk", joten kiteytyminen rajalla
        # rikkoo kriteerin "juuri ja juuri" -> Class II borderline.
        # (Vaihtoehto (3, 'borderline') olisi puolustettavissa, mutta
        # valitsemme tiukan tulkinnan, joka noudattaa kriteerin kirjainta.)
        gfa, conf = classify_baird_taylor(
            180.0, 40.0, 75.0, False, induction_time_censored=False
        )
        assert gfa == 2
        assert conf == "borderline"


class TestClassIIIAmbient:
    """>= 365 vrk tavanomaisessa (25 °C / 60 % RH) = Class III."""

    def test_well_above_threshold(self) -> None:
        gfa, conf = classify_baird_taylor(500.0, 25.0, 60.0, False)
        assert gfa == 3
        assert conf == "high"

    def test_exactly_at_threshold_is_borderline(self) -> None:
        gfa, conf = classify_baird_taylor(365.0, 25.0, 60.0, False)
        assert gfa == 3
        assert conf == "borderline"

    def test_below_threshold_in_ambient_is_class_ii(self) -> None:
        # Tavanomaisessa 200 vrk ei riitä Class III:ksi (eri sääntö kuin kiihdytetyssä).
        gfa, conf = classify_baird_taylor(200.0, 25.0, 60.0, False)
        assert gfa == 2


class TestUnknownConditions:
    """Tuntemattomat säilytysolot -> 'low' confidence tai (None, 'low')."""

    def test_no_storage_info_class_ii_range_returns_low(self) -> None:
        gfa, conf = classify_baird_taylor(60.0, None, None, False)
        assert gfa == 2
        assert conf == "low"

    def test_long_induction_unknown_conditions_returns_none(self) -> None:
        # MIKSI: Class III -kriteeri vaatii oloista vahvistuksen (40/75 tai
        # 25/60). 200 vrk ilman olotietoa ei voi todeta Class III:ksi -
        # parempi palauttaa (None, 'low') kuin arvata Class II.
        gfa, conf = classify_baird_taylor(200.0, None, None, False)
        assert gfa is None
        assert conf == "low"


class TestMissingData:
    def test_no_induction_time_no_dsc_returns_none(self) -> None:
        gfa, conf = classify_baird_taylor(None, 40.0, 75.0, False)
        assert gfa is None
        assert conf == "low"

    def test_negative_induction_time_raises(self) -> None:
        with pytest.raises(ValueError):
            classify_baird_taylor(-1.0, 40.0, 75.0, False)


class TestCensoringEdgeCases:
    """Sensurointi vaikuttaa myös muissa kohdissa kuin 180/365 rajalla."""

    def test_censored_below_class_i_threshold_returns_none(self) -> None:
        # Sensuroitu < 7 vrk: koe loppui ennen Class I -rajaa ilman
        # kiteytymistä -> stabiilisuusaika voi olla mikä tahansa -> ei luokita.
        gfa, conf = classify_baird_taylor(
            5.0, 40.0, 75.0, False, induction_time_censored=True
        )
        assert gfa is None
        assert conf == "low"

    def test_censored_in_class_ii_range_lowers_confidence(self) -> None:
        # Sensuroitu 60 vrk: stabiilisuus *vähintään* 60 vrk. Voi olla
        # Class II tai Class III -> Class II low confidence.
        gfa, conf = classify_baird_taylor(
            60.0, 40.0, 75.0, False, induction_time_censored=True
        )
        assert gfa == 2
        assert conf == "low"


class TestBackwardCompatibility:
    """MIKSI: varmistetaan, että vanhat kutsut (ilman censored-parametria)
    palauttavat saman tuloksen kuin ennen muutosta. Default = None säilyttää
    aiemman semantiikan."""

    @pytest.mark.parametrize(
        ("t", "expected_gfa", "expected_conf"),
        [
            (3.0, 1, "high"),
            (6.5, 1, "borderline"),
            (60.0, 2, "high"),
            (7.5, 2, "borderline"),
            (180.0, 3, "borderline"),
            (195.0, 3, "borderline"),
            (250.0, 3, "high"),
            (365.0, 3, "high"),
        ],
    )
    def test_default_censoring_matches_legacy(
        self, t: float, expected_gfa: int, expected_conf: str
    ) -> None:
        # Kutsutaan ilman induction_time_censored-parametria.
        gfa, conf = classify_baird_taylor(t, 40.0, 75.0, False)
        assert gfa == expected_gfa
        assert conf == expected_conf


class TestThresholdConstants:
    """Sanity check: vakioiden arvot vastaavat 2.3:n sääntöjä."""

    def test_constants(self) -> None:
        assert CLASS_I_MAX_DAYS == 7.0
        assert CLASS_II_MAX_DAYS_ACCELERATED == 180.0
        assert CLASS_III_MIN_DAYS_AMBIENT == 365.0
        assert 0 < BORDERLINE_FRACTION < 0.5
