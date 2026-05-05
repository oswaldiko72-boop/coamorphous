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


class TestDryShortTermProtocol:
    """Kuivasäilytys (P2O5 / silica gel, RH ≈ 0 %).

    MIKSI nämä testit: monet H1:n lähteet (mm. Löbmann 2011) eivät käytä
    ICH Q1A -protokollaa, vaan kuivakokeita 4-25 °C:ssa noin 21 vrk:n ajan.
    Tämä haara mahdollistaa tällaisten lähteiden luokituksen ja varmistaa,
    että label_confidence pysyy 'low':na ei-standardiksi tunnistetussa
    protokollassa.
    """

    def test_dry_short_term_censored_long(self) -> None:
        # 21 vrk sensuroituna kuivassa: ei kiteytymistä havaittu pitkän
        # ajan kuluessa -> todennäköisesti Class III, mutta ei-standardin
        # protokollan vuoksi 'low' confidence.
        gfa, conf = classify_baird_taylor(
            21.0, 25.0, 0.0, False,
            induction_time_censored=True,
            experimental_protocol="dry_short_term",
        )
        assert gfa == 3
        assert conf == "low"

    def test_dry_short_term_censored_medium(self) -> None:
        # 10 vrk sensuroituna kuivassa: välilukema (>= 7 mutta < 14)
        # -> luultavasti Class II, low confidence.
        gfa, conf = classify_baird_taylor(
            10.0, 25.0, 0.0, False,
            induction_time_censored=True,
            experimental_protocol="dry_short_term",
        )
        assert gfa == 2
        assert conf == "low"

    def test_dry_short_term_observed_crystallization(self) -> None:
        # 15 vrk havaittu kiteytyminen kuivassa: kesti vähintään viikon
        # -> Class II, low confidence.
        gfa, conf = classify_baird_taylor(
            15.0, 25.0, 0.0, False,
            induction_time_censored=False,
            experimental_protocol="dry_short_term",
        )
        assert gfa == 2
        assert conf == "low"

    def test_dry_short_term_fast_crystallization(self) -> None:
        # 3 vrk havaittu kiteytyminen kuivassa: nopea kiteytyminen jopa
        # kuivassakin viittaa Class I:een.
        gfa, conf = classify_baird_taylor(
            3.0, 25.0, 0.0, False,
            induction_time_censored=False,
            experimental_protocol="dry_short_term",
        )
        assert gfa == 1
        assert conf == "low"

    def test_dry_short_term_censored_below_threshold_returns_none(self) -> None:
        # Sensuroitu < 7 vrk kuivassa: koe loppui liian aikaisin
        # -> ei voi luokitella.
        gfa, conf = classify_baird_taylor(
            5.0, 25.0, 0.0, False,
            induction_time_censored=True,
            experimental_protocol="dry_short_term",
        )
        assert gfa is None
        assert conf == "low"


class TestTgPlus15KProtocol:
    """Tg+15 K -kineettinen testi (Class I:n alkuperäinen määrittely)."""

    def test_tg_plus_15K_class_i(self) -> None:
        # < 7 vrk Tg+15 K:ssa = Class I:n suora todiste, high confidence.
        gfa, conf = classify_baird_taylor(
            5.0, None, None, False,
            experimental_protocol="tg_plus_15K",
        )
        assert gfa == 1
        assert conf == "high"

    def test_tg_plus_15K_not_class_i(self) -> None:
        # >= 7 vrk Tg+15 K:ssa: ei Class I, mutta tarkkaa II/III-luokkaa
        # ei voi päätellä tästä testistä yksin -> (2, 'low').
        gfa, conf = classify_baird_taylor(
            10.0, None, None, False,
            experimental_protocol="tg_plus_15K",
        )
        assert gfa == 2
        assert conf == "low"


class TestProtocolMaxDuration:
    """protocol_max_duration_days -johdonmukaisuustarkistus."""

    def test_protocol_duration_consistency_raises(self) -> None:
        # Sensuroitu havainto > kokeen kesto on mahdoton -> ValueError.
        with pytest.raises(ValueError, match="protocol_max_duration_days"):
            classify_baird_taylor(
                30.0, 40.0, 75.0, False,
                induction_time_censored=True,
                protocol_max_duration_days=21.0,
            )

    def test_protocol_duration_within_bounds_passes(self) -> None:
        # 21 vrk sensuroituna kun kokeen maksimi on 21 vrk: OK.
        gfa, conf = classify_baird_taylor(
            21.0, 25.0, 0.0, False,
            induction_time_censored=True,
            experimental_protocol="dry_short_term",
            protocol_max_duration_days=21.0,
        )
        assert gfa == 3
        assert conf == "low"

    def test_protocol_duration_only_checked_when_censored(self) -> None:
        # Sensuroimaton havainto voi periaatteessa ylittää sammuneen kokeen
        # keston (esim. retrospektiivinen analyysi), joten tarkistus ei
        # päde sille.
        gfa, _ = classify_baird_taylor(
            30.0, 40.0, 75.0, False,
            induction_time_censored=False,
            protocol_max_duration_days=21.0,
        )
        # Ei nosta — havainto sallittu.
        assert gfa == 2


class TestNonStandardProtocol:
    """non_standard pakottaa label_confidence='low' standardilogiikan jälkeen."""

    def test_non_standard_class_iii_downgraded_to_low(self) -> None:
        # 365 vrk 40/75 olisi normaalisti (3, 'high'), mutta non_standard
        # -merkki pakottaa luottamuksen alas.
        gfa, conf = classify_baird_taylor(
            365.0, 40.0, 75.0, False,
            experimental_protocol="non_standard",
        )
        assert gfa == 3
        assert conf == "low"

    def test_class_iii_censored_unknown_protocol(self) -> None:
        # MIKSI: Allesø 2009:n 1:1 NAP-CIM kesti 186 vrk amorfisena 4 °C / 0 % RH
        # silica gel -desikkaattorissa (kuiva ja kylmä = lievempi olo kuin
        # ICH 40/75). Kokeilu loppui ilman havaittua kiteytymistä
        # (induction_time_censored=True). Aiempi logiikka palautti
        # (None, 'low') koska olot eivät täsmää 40/75:tä eikä 25/60:tä.
        # Uusi haara tunnistaa sensuroidun >=180 vrk havainnon Class III:ksi
        # mutta low confidence -painokertoimella.
        gfa, conf = classify_baird_taylor(
            186.0, 4.0, 0.0, False,
            induction_time_censored=True,
            experimental_protocol="non_standard",
        )
        assert gfa == 3
        assert conf == "low"

    def test_class_iii_censored_dry_extended(self) -> None:
        # Pidempi sensuroitu koe (200 vrk) kuivassa huoneenlämmössä
        # (25 °C / 0 % RH). Sensuroitu vahvistaa "ei kiteytymistä >= 180 vrk"
        # -kriteerin, mutta kuiva olo on lievempi kuin 40/75 -> (3, 'low').
        gfa, conf = classify_baird_taylor(
            200.0, 25.0, 0.0, False,
            induction_time_censored=True,
            experimental_protocol="non_standard",
        )
        assert gfa == 3
        assert conf == "low"

    def test_class_iii_censored_unknown_conditions_no_protocol(self) -> None:
        # Sama haara laukeaa myös ilman eksplisiittistä non_standard-protokollaa,
        # kunhan olot ovat tuntemattomat ja induction_time_censored=True.
        # Tämä on tärkeää, koska aiempi semantiikka palautti (None, 'low')
        # tässä tapauksessa, ja uusi haara muuttaa sen (3, 'low'):ksi.
        gfa, conf = classify_baird_taylor(
            186.0, None, None, False,
            induction_time_censored=True,
        )
        assert gfa == 3
        assert conf == "low"

    def test_class_iii_censored_below_180_in_unknown_conditions_unaffected(self) -> None:
        # Sensuroitu < 180 vrk tuntemattomissa oloissa: uusi haara EI laukea.
        # Vanha _classify_class_ii_zone-logiikka palauttaa (2, 'low').
        # Tarkistetaan, ettei uusi haara ohita Class II -aluetta.
        gfa, conf = classify_baird_taylor(
            150.0, None, None, False,
            induction_time_censored=True,
        )
        assert gfa == 2
        assert conf == "low"


class TestLegacyCompatibility:
    """Vahvistus: nykyiset kutsut ilman uusia parametreja toimivat ennallaan.

    Tämä testi on vakuutus regressioita vastaan: kun lisäsimme uudet
    parametrit (experimental_protocol, protocol_max_duration_days), niiden
    oletusarvojen täytyy säilyttää aiempi käyttäytyminen.
    """

    @pytest.mark.parametrize(
        ("t", "T_C", "RH", "expected_gfa", "expected_conf"),
        [
            (3.0, 40.0, 75.0, 1, "high"),
            (60.0, 40.0, 75.0, 2, "high"),
            (180.0, 40.0, 75.0, 3, "borderline"),
            (250.0, 40.0, 75.0, 3, "high"),
            (500.0, 25.0, 60.0, 3, "high"),
        ],
    )
    def test_no_protocol_param_matches_legacy(
        self, t: float, T_C: float, RH: float,
        expected_gfa: int, expected_conf: str,
    ) -> None:
        # Kutsutaan ilman experimental_protocol- ja protocol_max_duration_days
        # -parametreja: tuloksen pitää olla sama kuin ennen muutosta.
        gfa, conf = classify_baird_taylor(t, T_C, RH, False)
        assert gfa == expected_gfa
        assert conf == expected_conf
