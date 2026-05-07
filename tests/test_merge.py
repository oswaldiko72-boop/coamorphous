"""Yksikkötestit interim-CSV:iden yhdistämiselle ja duplikaattihavainnolle.

MIKSI nämä testit ovat olemassa
-------------------------------
``merge_interim`` yhdistää lähdekohtaiset ekstraktiot master-korpukseksi.
Cross-source -duplikaatit (sama pari useassa lähteessä) vinouttavat ML-mallin
opetusta jos niitä ei tunnisteta — ``find_cross_source_duplicates`` toteuttaa
InChIKey-pohjaisen tunnistuksen, joka on robusti nimien kirjoitusasun
vaihteluille (Naproxen vs naproxen vs Naproxen sodium).
"""

from __future__ import annotations

import pandas as pd
import pytest

from coamorphous.corpus.merge import (
    _inchikey_pair_key,
    canonicalize_names_by_inchikey,
    find_cross_source_duplicates,
    harmonize_mole_fractions,
    pick_canonical_name,
    snap_to_simple_ratio,
)


# =============================================================================
# _inchikey_pair_key — järjestysriippumaton avain
# =============================================================================


class TestInchikeyPairKey:
    def test_order_independent(self) -> None:
        # (A, B) ja (B, A) tuottavat saman avaimen — pari on järjestysriippumaton.
        assert _inchikey_pair_key("AAA", "BBB") == _inchikey_pair_key("BBB", "AAA")

    def test_two_distinct_keys(self) -> None:
        key = _inchikey_pair_key("AAA", "BBB")
        assert key == frozenset({"AAA", "BBB"})

    def test_none_input_returns_none(self) -> None:
        assert _inchikey_pair_key(None, "BBB") is None
        assert _inchikey_pair_key("AAA", None) is None
        assert _inchikey_pair_key(None, None) is None

    def test_nan_input_returns_none(self) -> None:
        # pandas voi antaa NaN floattina puuttuvalle InChIKey:lle CSV:n luvun
        # jälkeen — varmistetaan että tämäkin käsitellään puuttuvana.
        import math

        assert _inchikey_pair_key(math.nan, "BBB") is None
        assert _inchikey_pair_key("AAA", math.nan) is None

    def test_empty_string_returns_none(self) -> None:
        # Tyhjä merkkijono tai pelkkä whitespace ei ole validi InChIKey.
        assert _inchikey_pair_key("", "BBB") is None
        assert _inchikey_pair_key("   ", "BBB") is None


# =============================================================================
# find_cross_source_duplicates
# =============================================================================


def _make_row(
    pair_id: str,
    inchikey_A: str | None,
    inchikey_B: str | None,
    first_author: str,
    year: int,
) -> dict:
    """Apuri minimirivin rakentamiselle testeissä."""
    return {
        "pair_id": pair_id,
        "drug_A_inchikey": inchikey_A,
        "drug_B_inchikey": inchikey_B,
        "source_first_author": first_author,
        "source_year": year,
        "source_doi": f"10.x/{first_author}{year}",
    }


class TestFindCrossSourceDuplicates:
    def test_empty_dataframe_returns_empty_list(self) -> None:
        df = pd.DataFrame(
            columns=[
                "pair_id",
                "drug_A_inchikey",
                "drug_B_inchikey",
                "source_first_author",
                "source_year",
                "source_doi",
            ]
        )
        assert find_cross_source_duplicates(df) == []

    def test_no_duplicates_returns_empty(self) -> None:
        df = pd.DataFrame(
            [
                _make_row("a2009_01", "AAA", "BBB", "alleso", 2009),
                _make_row("a2009_02", "CCC", "DDD", "alleso", 2009),
                _make_row("b2012_01", "EEE", "FFF", "fink", 2012),
            ]
        )
        assert find_cross_source_duplicates(df) == []

    def test_same_pair_different_sources_is_duplicate(self) -> None:
        df = pd.DataFrame(
            [
                _make_row("a2009_01", "AAA", "BBB", "alleso", 2009),
                _make_row("b2012_03", "AAA", "BBB", "fink", 2012),
            ]
        )
        groups = find_cross_source_duplicates(df)
        assert len(groups) == 1
        assert groups[0]["inchikey_pair"] == ("AAA", "BBB")
        assert sorted(groups[0]["pair_ids"]) == ["a2009_01", "b2012_03"]
        assert sorted(set(groups[0]["sources"])) == ["alleso2009", "fink2012"]

    def test_reversed_inchikey_order_still_detected(self) -> None:
        # Drug A ja B voivat olla eri järjestyksessä eri lähteissä — tunnistuksen
        # täytyy käsitellä paria järjestysriippumattomasti.
        df = pd.DataFrame(
            [
                _make_row("a2009_01", "AAA", "BBB", "alleso", 2009),
                _make_row("b2012_03", "BBB", "AAA", "fink", 2012),  # käännetty
            ]
        )
        groups = find_cross_source_duplicates(df)
        assert len(groups) == 1
        assert groups[0]["inchikey_pair"] == ("AAA", "BBB")

    def test_same_source_same_pair_is_not_cross_source_duplicate(self) -> None:
        # Saman lähteen sisäiset toistot (esim. eri kompositiot) eivät ole
        # cross-source-duplikaatteja — niitä käsittelee pair_id-tarkistus.
        df = pd.DataFrame(
            [
                _make_row("knapik2015_01", "AAA", "BBB", "knapik", 2015),
                _make_row("knapik2015_02", "AAA", "BBB", "knapik", 2015),
                _make_row("knapik2015_03", "AAA", "BBB", "knapik", 2015),
            ]
        )
        assert find_cross_source_duplicates(df) == []

    def test_missing_inchikey_skipped(self) -> None:
        # Rivit, joilla puuttuu InChIKey, eivät voi olla duplikaattikandidaatteja.
        df = pd.DataFrame(
            [
                _make_row("a2009_01", None, "BBB", "alleso", 2009),
                _make_row("b2012_03", "AAA", "BBB", "fink", 2012),
                _make_row("c2014_05", "AAA", None, "pajula", 2014),
            ]
        )
        # Vain b2012_03:lla on täysi InChIKey-pari, eikä mikään muu rivi vastaa
        # samaa paria täydellisillä InChIKey:llä — joten ei duplikaattia.
        assert find_cross_source_duplicates(df) == []

    def test_three_sources_same_pair_one_group(self) -> None:
        df = pd.DataFrame(
            [
                _make_row("a2009_01", "AAA", "BBB", "alleso", 2009),
                _make_row("b2012_03", "AAA", "BBB", "fink", 2012),
                _make_row("c2014_05", "BBB", "AAA", "pajula", 2014),  # käännetty
            ]
        )
        groups = find_cross_source_duplicates(df)
        assert len(groups) == 1
        assert sorted(groups[0]["pair_ids"]) == ["a2009_01", "b2012_03", "c2014_05"]
        assert sorted(set(groups[0]["sources"])) == [
            "alleso2009",
            "fink2012",
            "pajula2014",
        ]

    def test_multiple_independent_duplicate_groups(self) -> None:
        df = pd.DataFrame(
            [
                # Ryhmä 1: AAA-BBB kahdessa lähteessä.
                _make_row("a2009_01", "AAA", "BBB", "alleso", 2009),
                _make_row("b2012_03", "AAA", "BBB", "fink", 2012),
                # Ryhmä 2: CCC-DDD kahdessa lähteessä.
                _make_row("a2009_02", "CCC", "DDD", "alleso", 2009),
                _make_row("c2014_05", "CCC", "DDD", "pajula", 2014),
                # Yksittäinen ei-duplikaatti.
                _make_row("d2015_01", "EEE", "FFF", "knapik", 2015),
            ]
        )
        groups = find_cross_source_duplicates(df)
        assert len(groups) == 2
        # Sortattu InChIKey-parin mukaan — AAA-BBB tulee ennen CCC-DDD:tä.
        assert groups[0]["inchikey_pair"] == ("AAA", "BBB")
        assert groups[1]["inchikey_pair"] == ("CCC", "DDD")

    def test_missing_required_column_raises(self) -> None:
        df = pd.DataFrame([{"pair_id": "x", "drug_A_inchikey": "AAA"}])
        with pytest.raises(KeyError, match="find_cross_source_duplicates"):
            find_cross_source_duplicates(df)


# =============================================================================
# pick_canonical_name — kanonisen muodon valintalogiikka
# =============================================================================


class TestPickCanonicalName:
    def test_most_common_wins(self) -> None:
        # "Naproxen" esiintyy 2x, "naproxen" 1x -> yleisin voittaa.
        assert pick_canonical_name(["Naproxen", "Naproxen", "naproxen"]) == "Naproxen"

    def test_longest_breaks_tie_on_count(self) -> None:
        # Tasapeli esiintymismäärässä -> pisin voittaa: "Ezetimibe" > "EZB".
        assert pick_canonical_name(["EZB", "Ezetimibe"]) == "Ezetimibe"

    def test_alphabetical_breaks_tie_on_length(self) -> None:
        # Tasapeli sekä count että len -> aakkosjärjestyksessä pienempi voittaa.
        # ASCII:ssa isot kirjaimet (A=65) tulevat ennen pieniä (a=97).
        assert pick_canonical_name(["Naproxen", "naproxen"]) == "Naproxen"

    def test_single_name_returned_asis(self) -> None:
        assert pick_canonical_name(["Indomethacin"]) == "Indomethacin"

    def test_empty_iterable_returns_empty_string(self) -> None:
        assert pick_canonical_name([]) == ""

    def test_filters_none_and_empty(self) -> None:
        # None ja tyhjät merkkijonot eivät ole valideja kandidaatteja.
        assert pick_canonical_name([None, "", "  ", "Naproxen"]) == "Naproxen"

    def test_count_priority_over_length(self) -> None:
        # Lyhyt mutta yleinen voittaa pidemmän mutta harvinaisen — count
        # menee ensin tärkeysjärjestyksessä.
        result = pick_canonical_name(
            ["EZB", "EZB", "EZB", "Ezetimibe"]
        )
        assert result == "EZB"


# =============================================================================
# canonicalize_names_by_inchikey
# =============================================================================


class TestCanonicalizeNamesByInchikey:
    def test_case_variants_unified(self) -> None:
        df = pd.DataFrame(
            {
                "drug_A_name": ["naproxen", "Naproxen", "Naproxen"],
                "drug_B_name": ["indomethacin", "indomethacin", "indomethacin"],
                "drug_A_inchikey": ["NAP", "NAP", "NAP"],
                "drug_B_inchikey": ["IND", "IND", "IND"],
            }
        )
        out = canonicalize_names_by_inchikey(df)
        # Kaikkien naproxen-rivien pitäisi saada sama kanoninen nimi.
        assert out["drug_A_name"].nunique() == 1
        assert out["drug_A_name"].iloc[0] == "Naproxen"

    def test_consolidates_across_a_and_b_columns(self) -> None:
        # Sama lääke (NAP-InChIKey) esiintyy A:na rivillä 0 ja B:nä rivillä 1.
        # Kanonisoinnin pitää huomioida nimet molemmista sarakkeista ja
        # päivittää molemmat positiot.
        df = pd.DataFrame(
            {
                "drug_A_name": ["naproxen", "indomethacin"],
                "drug_B_name": ["indomethacin", "Naproxen"],
                "drug_A_inchikey": ["NAP", "IND"],
                "drug_B_inchikey": ["IND", "NAP"],
            }
        )
        out = canonicalize_names_by_inchikey(df)
        # NAP esiintyy: "naproxen" (A-sar.) ja "Naproxen" (B-sar.) -> tasapeli
        # count 1+1, sama len -> aakkosjärjestys -> "Naproxen".
        assert out.loc[0, "drug_A_name"] == "Naproxen"
        assert out.loc[1, "drug_B_name"] == "Naproxen"

    def test_abbreviation_replaced_by_full_name(self) -> None:
        df = pd.DataFrame(
            {
                "drug_A_name": ["EZB", "Ezetimibe"],
                "drug_B_name": ["Indomethacin", "Indomethacin"],
                "drug_A_inchikey": ["EZ", "EZ"],
                "drug_B_inchikey": ["IND", "IND"],
            }
        )
        out = canonicalize_names_by_inchikey(df)
        # Tasapeli count, "Ezetimibe" pidempi -> molemmat rivit saavat sen.
        assert (out["drug_A_name"] == "Ezetimibe").all()

    def test_missing_inchikey_leaves_name_alone(self) -> None:
        # Jos InChIKey puuttuu, nimeä ei voi luotettavasti normalisoida.
        df = pd.DataFrame(
            {
                "drug_A_name": ["Mystery", "Naproxen"],
                "drug_B_name": ["Indomethacin", "Indomethacin"],
                "drug_A_inchikey": [None, "NAP"],
                "drug_B_inchikey": ["IND", "IND"],
            }
        )
        out = canonicalize_names_by_inchikey(df)
        assert out.loc[0, "drug_A_name"] == "Mystery"  # ennallaan
        assert out.loc[1, "drug_A_name"] == "Naproxen"

    def test_does_not_mutate_input(self) -> None:
        # Funktio palauttaa kopion eikä muuta alkuperäistä DataFrame:a.
        df = pd.DataFrame(
            {
                "drug_A_name": ["naproxen"],
                "drug_B_name": ["indomethacin"],
                "drug_A_inchikey": ["NAP"],
                "drug_B_inchikey": ["IND"],
            }
        )
        original_a = df["drug_A_name"].copy()
        canonicalize_names_by_inchikey(df)
        pd.testing.assert_series_equal(df["drug_A_name"], original_a)

    def test_does_not_touch_inchikey_or_smiles_columns(self) -> None:
        # Auditoinnin kannalta tärkeää: kemiallinen identiteetti pysyy ennallaan.
        df = pd.DataFrame(
            {
                "drug_A_name": ["naproxen", "Naproxen"],
                "drug_B_name": ["indomethacin", "indomethacin"],
                "drug_A_inchikey": ["NAP-A", "NAP-A"],
                "drug_B_inchikey": ["IND-B", "IND-B"],
                "drug_A_smiles_canonical": ["CC1", "CC1"],
                "drug_B_smiles_canonical": ["CC2", "CC2"],
            }
        )
        out = canonicalize_names_by_inchikey(df)
        pd.testing.assert_series_equal(out["drug_A_inchikey"], df["drug_A_inchikey"])
        pd.testing.assert_series_equal(
            out["drug_A_smiles_canonical"], df["drug_A_smiles_canonical"]
        )

    def test_missing_required_column_raises(self) -> None:
        df = pd.DataFrame([{"drug_A_name": "x", "drug_A_inchikey": "AAA"}])
        with pytest.raises(KeyError, match="canonicalize_names_by_inchikey"):
            canonicalize_names_by_inchikey(df)


# =============================================================================
# snap_to_simple_ratio
# =============================================================================


class TestSnapToSimpleRatio:
    def test_snaps_2_3_three_decimals(self) -> None:
        # 0.667 on 2/3:n pyöristetty muoto -> snäppää 2/3:een (= 0.6667).
        assert snap_to_simple_ratio(0.667) == 0.6667

    def test_snaps_2_3_four_decimals(self) -> None:
        # 0.6667 on jo lähellä 2/3:a -> snäppää samaan kanoniseen muotoon.
        assert snap_to_simple_ratio(0.6667) == 0.6667

    def test_does_not_snap_massweighted_value(self) -> None:
        # 0.506 (massapainotettu 1:1, Knapik 2019) ei ole minkään yksinkertaisen
        # suhteen lähellä toleranssin 0.002 sisällä -> säilyy ennallaan.
        assert snap_to_simple_ratio(0.506) == 0.506

    def test_snaps_half(self) -> None:
        assert snap_to_simple_ratio(0.5) == 0.5

    def test_snaps_one_third(self) -> None:
        assert snap_to_simple_ratio(0.333) == 0.3333
        assert snap_to_simple_ratio(0.3333) == 0.3333

    def test_snaps_ten_eleventh(self) -> None:
        # 10:1 mol-suhde -> mole_fraction_A = 10/11 = 0.9091.
        # 0.909 on 10/11:n karkea pyöristys, snäppää tarkempaan muotoon.
        assert snap_to_simple_ratio(0.909) == 0.9091

    def test_endpoints(self) -> None:
        assert snap_to_simple_ratio(0.0) == 0.0
        assert snap_to_simple_ratio(1.0) == 1.0

    def test_none_returns_none(self) -> None:
        assert snap_to_simple_ratio(None) is None

    def test_nan_returns_nan(self) -> None:
        import math

        assert math.isnan(snap_to_simple_ratio(math.nan))

    def test_out_of_range_returned_asis(self) -> None:
        # Joukon ulkopuoliset (esim. virheellinen syöttö) palautetaan ennallaan
        # ilman snäppäystä — ei valideja mole_fraction-arvoja.
        assert snap_to_simple_ratio(-0.1) == -0.1
        assert snap_to_simple_ratio(1.5) == 1.5

    def test_tolerance_excludes_far_values(self) -> None:
        # 0.55 on 1/2:n ja 5/9:n välissä; ei kummankaan toleranssin sisällä.
        # 0.55 - 0.5 = 0.05 > 0.002, 0.55 - 5/9 = 0.0056 > 0.002 -> ei snäppää.
        assert snap_to_simple_ratio(0.55) == 0.55

    def test_simpler_denominator_wins_on_tie(self) -> None:
        # 0.5 osuu sekä 1/2:n että 2/4:n kohdalle (diff = 0). Pienempi
        # nimittäjä (q=1 -> 0/1 = 0, ei pätevä; q=2 -> 1/2 = 0.5) voittaa.
        assert snap_to_simple_ratio(0.5) == 0.5

    def test_custom_tolerance(self) -> None:
        # Tiukempi toleranssi estää 0.667:n snäppäyksen 2/3:een (diff ~0.0003).
        assert snap_to_simple_ratio(0.667, tolerance=0.0001) == 0.667
        # 0.555 on lähimpänä 5/9:ää (diff 0.0006), mutta diff > 0.0001.
        # Tiukassa toleranssissa palautuu ennallaan.
        assert snap_to_simple_ratio(0.555, tolerance=0.0001) == 0.555
        # Löysemmässä toleranssissa snäppää 5/9:ään.
        assert snap_to_simple_ratio(0.555, tolerance=0.001) == 0.5556


# =============================================================================
# harmonize_mole_fractions
# =============================================================================


class TestHarmonizeMoleFractions:
    def test_snaps_and_preserves_raw(self) -> None:
        df = pd.DataFrame(
            {
                "mole_fraction_A": [0.667, 0.6667, 0.506],
                "mole_fraction_B": [0.333, 0.3333, 0.494],
            }
        )
        out = harmonize_mole_fractions(df)
        # Snäpätyt arvot.
        assert out["mole_fraction_A"].tolist() == [0.6667, 0.6667, 0.506]
        # B = 1 - A jolloin summa varmasti 1.0.
        assert out["mole_fraction_B"].tolist() == [0.3333, 0.3333, 0.494]
        # Raw-sarakkeet säilyttävät alkuperäiset.
        assert out["mole_fraction_A_raw"].tolist() == [0.667, 0.6667, 0.506]
        assert out["mole_fraction_B_raw"].tolist() == [0.333, 0.3333, 0.494]

    def test_sum_equals_one_after_harmonization(self) -> None:
        df = pd.DataFrame(
            {
                "mole_fraction_A": [0.667, 0.5, 0.909, 0.506],
                "mole_fraction_B": [0.333, 0.5, 0.091, 0.494],
            }
        )
        out = harmonize_mole_fractions(df)
        sums = out["mole_fraction_A"] + out["mole_fraction_B"]
        # Sallitaan pieni floating-point -toleranssi.
        assert (sums - 1.0).abs().max() < 1e-9

    def test_nan_rows_left_alone(self) -> None:
        df = pd.DataFrame(
            {
                "mole_fraction_A": [0.667, None, 0.5],
                "mole_fraction_B": [0.333, None, 0.5],
            }
        )
        out = harmonize_mole_fractions(df)
        # NaN-rivin pitäisi pysyä NaN:nä molemmissa snäpätyssä ja raw:ssa.
        assert pd.isna(out.loc[1, "mole_fraction_A"])
        assert pd.isna(out.loc[1, "mole_fraction_B"])
        assert pd.isna(out.loc[1, "mole_fraction_A_raw"])

    def test_does_not_mutate_input(self) -> None:
        df = pd.DataFrame(
            {
                "mole_fraction_A": [0.667],
                "mole_fraction_B": [0.333],
            }
        )
        original = df["mole_fraction_A"].copy()
        harmonize_mole_fractions(df)
        pd.testing.assert_series_equal(df["mole_fraction_A"], original)
        # Eikä lisää sarakkeita alkuperäiseen.
        assert "mole_fraction_A_raw" not in df.columns

    def test_does_not_touch_weight_fractions(self) -> None:
        df = pd.DataFrame(
            {
                "mole_fraction_A": [0.667],
                "mole_fraction_B": [0.333],
                "weight_fraction_A": [0.687],
                "weight_fraction_B": [0.313],
            }
        )
        out = harmonize_mole_fractions(df)
        # weight_fraction-sarakkeet säilyvät ennallaan — eri konsepti (massa).
        assert out["weight_fraction_A"].iloc[0] == 0.687
        assert out["weight_fraction_B"].iloc[0] == 0.313

    def test_missing_required_column_raises(self) -> None:
        df = pd.DataFrame([{"mole_fraction_A": 0.5}])
        with pytest.raises(KeyError, match="harmonize_mole_fractions"):
            harmonize_mole_fractions(df)
