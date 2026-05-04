"""Baird-Taylor lasinmuodostumisluokitus.

MIKSI tämä moduuli on olemassa
------------------------------
Co-amorfisten lääkesysteemien stabiilisuus luokitellaan empiirisesti
kolmeen luokkaan, jotka heijastavat lasinmuodostumistaipumusta
(*Glass-Forming Ability*, GFA):

* **Class I** — heikko GFA. Aine kiteytyy nopeasti DSC-jäähdytyksellä
  tai induktioaika säilytysoloissa Tg+15 K on alle viikko. Käytännössä
  käyttökelvoton lääkevalmistuksessa.
* **Class II** — kohtalainen GFA. Stabiili viikkoja-kuukausia mutta
  alle 6 kuukautta. Mahdollinen, jos säilytys on kontrolloitua.
* **Class III** — vahva GFA. Stabiili vähintään 6 kuukautta kiihdytetyissä
  oloissa (40 °C / 75 % RH) tai vähintään vuoden tavanomaisissa oloissa
  (25 °C / 60 % RH). Lääkevalmistuksen "kultainen standardi".

Luokitus on siis induktioajan, säilytysolojen ja DSC-käyttäytymisen
yhteistulos. Kirjallisuus käyttää eri raja-arvoja; tämä toteutus seuraa
Baird & Taylor (2012) -konsensusta sellaisena kuin H1:n kohta 2.3 sen
määrittää.

Reference
---------
Baird, J. A. & Taylor, L. S. *Adv. Drug Deliv. Rev.* 64 (2012) 396–421.

Sensurointi (induction_time_censored)
-------------------------------------
Stabiilisuuskokeissa on tilastollisesti tärkeä ero kahden tilanteen välillä:

* **Sensuroimaton** (``censored=False``) — kiteytyminen *havaittiin*
  raportoitulla ajanhetkellä. Aika on siis stabiilisuusajan **yläraja**:
  näyte oli amorfinen tähän asti ja kiteytyi sitten.
* **Sensuroitu** (``censored=True``) — koe loppui ilman havaittua
  kiteytymistä. Aika on stabiilisuusajan **alaraja**: näyte saattaisi
  pysyä amorfisena vielä pidempäänkin.

Tämä erottelu vaikuttaa erityisesti Class III -kynnysten lähellä:
``censored=True`` ja ``t >= 180 vrk`` 40/75 -oloissa on suora todiste
Class III:sta, kun taas ``censored=False`` ja ``t = 180 vrk`` tarkoittaa,
että kiteytyminen havaittiin tasan rajalla — eli näyte vain hädin tuskin
*ei* täytä Class III -kriteeriä (``ei kiteytymistä >= 180 vrk``).

Toteutuksen tarkoitus
---------------------
Klassifikaatio on yksi paikka, jossa virheet kertautuvat: jos sama rivi
saa eri luokan eri ekstraktoreilta, koko ML-mallin opetus on roskaa.
Siksi luokitin on:

* puhdas, sivuvaikutukseton funktio (helppo testata),
* eksplisiittisillä kynnysarvoilla (modulivakioina, ei "taikanumeroina"),
* palauttaa luottamustason, jotta rajatapauksia voi suodattaa pois
  opetusvaiheessa.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Kynnysarvot (Baird-Taylor 2012, H1 kohta 2.3).
# Säilytetään modulivakioina, jotta:
#   - testit voivat viitata samoihin nimiin,
#   - jos kirjallisuus muuttuu (uusi konsensus), muutos tehdään yhdessä paikassa.
# -----------------------------------------------------------------------------

# Class I -> Class II raja-arvo (päivinä).
# Alle viikko (7 vrk) Tg+15 K:ssa = Class I.
CLASS_I_MAX_DAYS: float = 7.0

# Class II -> Class III raja-arvo (päivinä).
# 180 vrk = 6 kuukautta, kiihdytetty stabiilisuustesti (40 °C / 75 % RH).
CLASS_II_MAX_DAYS_ACCELERATED: float = 180.0

# Class III vaihtoehtoinen kynnys: 365 vrk normaaleissa oloissa
# (25 °C / 60 % RH).
CLASS_III_MIN_DAYS_AMBIENT: float = 365.0

# Kiihdytettyjen olojen määritelmä — tunnistaa, kumpaa kynnystä sovelletaan.
# Käytetään väljiä rajoja: artikkeleissa on usein "40 ± 2 °C" tai "75 ± 5 % RH".
ACCELERATED_T_C_MIN: float = 35.0
ACCELERATED_T_C_MAX: float = 45.0
ACCELERATED_RH_MIN: float = 70.0
ACCELERATED_RH_MAX: float = 80.0

AMBIENT_T_C_MIN: float = 20.0
AMBIENT_T_C_MAX: float = 30.0
AMBIENT_RH_MIN: float = 55.0
AMBIENT_RH_MAX: float = 65.0

# Borderline-vyöhyke: ±10 % luokkarajan ympärillä saa label_confidence='borderline'.
# Tämä toteuttaa H1 2.3:n säännön "raja-arvoilla esim. 6,5 kk".
# 180 vrk * 0.10 = 18 vrk, eli 162-198 vrk on borderline II/III.
BORDERLINE_FRACTION: float = 0.10


def _is_accelerated(storage_T_C: Optional[float], storage_RH_percent: Optional[float]) -> bool:
    """Onko säilytys "kiihdytetty" (40 °C / 75 % RH ICH Q1A:n mukaan)?

    Palauttaa False, jos jompikumpi arvo puuttuu — turvallinen oletus,
    koska tuntemattomista oloista ei voi valittaa kiihdytettyjä luokkarajoja.
    """
    if storage_T_C is None or storage_RH_percent is None:
        return False
    return (
        ACCELERATED_T_C_MIN <= storage_T_C <= ACCELERATED_T_C_MAX
        and ACCELERATED_RH_MIN <= storage_RH_percent <= ACCELERATED_RH_MAX
    )


def _is_ambient(storage_T_C: Optional[float], storage_RH_percent: Optional[float]) -> bool:
    """Onko säilytys "tavanomainen" (25 °C / 60 % RH)."""
    if storage_T_C is None or storage_RH_percent is None:
        return False
    return (
        AMBIENT_T_C_MIN <= storage_T_C <= AMBIENT_T_C_MAX
        and AMBIENT_RH_MIN <= storage_RH_percent <= AMBIENT_RH_MAX
    )


def _is_borderline(value: float, threshold: float) -> bool:
    """Onko ``value`` borderline-vyöhykkeellä (±BORDERLINE_FRACTION) kynnyksen ympärillä?"""
    margin = threshold * BORDERLINE_FRACTION
    return abs(value - threshold) <= margin


def classify_baird_taylor(
    induction_time_days: Optional[float],
    storage_T_C: Optional[float],
    storage_RH_percent: Optional[float],
    crystallizes_on_dsc_cooling: Optional[bool],
    induction_time_censored: Optional[bool] = None,
) -> tuple[Optional[int], str]:
    """Luokittele co-amorfinen pari Baird-Taylor luokkaan I, II tai III.

    Parameters
    ----------
    induction_time_days : float, optional
        Aika kiteytymisen alkamiseen (vrk). ``None`` jos ei mitattu.
    storage_T_C : float, optional
        Säilytyslämpötila celsiuksina.
    storage_RH_percent : float, optional
        Suhteellinen kosteus prosentteina.
    crystallizes_on_dsc_cooling : bool, optional
        True jos näyte kiteytyi jo DSC-jäähdytyssyklissä — tämä on suora
        merkki Class I:stä riippumatta induktioaikamittauksesta.
    induction_time_censored : bool, optional
        Right-censoring -lippu induktioajalle. Oletus ``None`` (tuntematon)
        säilyttää aiemman semantiikan: borderline-tarkistus tehdään
        konservatiivisesti molempiin suuntiin. Eksplisiittinen ``True``
        (koe loppui ilman kiteytymistä) **vahvistaa** Class III -luokituksen
        180/365 vrk:n rajalla; eksplisiittinen ``False`` (kiteytyminen
        havaittu) tarkoittaa, että tasan kynnyksellä raportoitu aika
        on **yläraja** stabiilisuusajalle, ja luokitus on Class II
        borderline (kiteytyi juuri ja juuri ennen Class III -kriteerin
        täyttymistä).

    Returns
    -------
    gfa_class : int or None
        1, 2, tai 3 — Baird-Taylor luokka. ``None`` jos dataa ei riitä
        luokituksen tekemiseen (esim. >= 180 vrk tuntemattomissa oloissa,
        jolloin Class III -kriteeriä ei voida vahvistaa).
    label_confidence : {'high', 'low', 'borderline'}
        ``'high'`` selkeissä tapauksissa, ``'borderline'`` raja-arvojen
        ±10 % vyöhykkeellä, ``'low'`` kun datan puute pakottaa arvauksen.

    Notes
    -----
    Sääntöjen järjestys on tärkeä: DSC-kiteytyminen testataan ensin, koska
    se ohittaa muut indikaattorit. Sen jälkeen induktioaika, ja viimeisenä
    pitkäaikaissäilytyksen kynnykset.

    Borderline-vyöhyke tarkistetaan **molempien** luokkarajojen ympärillä
    Class II -alueella (7 vrk ja 180/365 vrk), koska 162 vrk:n näyte 40/75
    -oloissa on yhtä lähellä Class III -kynnystä kuin 7,5 vrk:n näyte on
    Class I -kynnystä.

    Examples
    --------
    DSC-kiteytyminen pakottaa Class I:n riippumatta induktioajasta:

    >>> classify_baird_taylor(200.0, 40.0, 75.0, True)
    (1, 'high')

    Selvä Class III sensuroidulla pitkällä kokeella (≥180 vrk ilman
    havaittua kiteytymistä, 40 °C / 75 % RH):

    >>> classify_baird_taylor(180.0, 40.0, 75.0, False, induction_time_censored=True)
    (3, 'high')

    Sama ajanhetki sensuroimattomana = kiteytyminen havaittu rajalla,
    eli juuri ja juuri Class II:

    >>> classify_baird_taylor(180.0, 40.0, 75.0, False, induction_time_censored=False)
    (2, 'borderline')

    Class II:n yläraja (162 vrk = 180 - 18) → borderline lähellä Class III:a:

    >>> classify_baird_taylor(162.0, 40.0, 75.0, False)
    (2, 'borderline')

    Tuntemattomat olot ja pitkä induktioaika eivät riitä luokitukseen:

    >>> classify_baird_taylor(200.0, None, None, False)
    (None, 'low')
    """
    # --- Sääntö 1: DSC-jäähdytyksellä havaittu kiteytyminen = Class I ----------
    # MIKSI ensin: DSC-kiteytyminen on suorin testi heikolle GFA:lle, eikä
    # induktioaikadata voi kumota sitä.
    if crystallizes_on_dsc_cooling is True:
        return 1, "high"

    # Ilman induktioaikaa emme voi luokitella muiden sääntöjen perusteella.
    if induction_time_days is None:
        logger.debug(
            "classify_baird_taylor: induktioaika puuttuu, ei voi luokitella."
        )
        return None, "low"

    if induction_time_days < 0:
        raise ValueError(
            f"induction_time_days ei voi olla negatiivinen, sai: {induction_time_days}"
        )

    # --- Sääntö 2: < 7 vrk -> Class I -----------------------------------------
    # Sensuroitu < 7 vrk = koe loppui ennen 7 vrk:tä ilman kiteytymistä.
    # Stabiilisuusaika on tällöin alaraja, ei välttämättä Class I — palautetaan
    # (None, 'low'), koska todellinen luokka voi olla I, II tai III.
    if induction_time_days < CLASS_I_MAX_DAYS:
        if induction_time_censored is True:
            logger.debug(
                "classify_baird_taylor: sensuroitu alle 7 vrk, ei voi luokitella."
            )
            return None, "low"
        confidence = (
            "borderline"
            if _is_borderline(induction_time_days, CLASS_I_MAX_DAYS)
            else "high"
        )
        return 1, confidence

    # --- Sääntö 3: pitkäaikaissäilytys ----------------------------------------
    # Kiihdytetty (40 °C / 75 % RH).
    if _is_accelerated(storage_T_C, storage_RH_percent):
        threshold_iii = CLASS_II_MAX_DAYS_ACCELERATED  # 180
        if induction_time_days >= threshold_iii:
            return _classify_class_iii_zone(
                induction_time_days, threshold_iii, induction_time_censored
            )
        # 7 <= t < 180 -> Class II.
        return _classify_class_ii_zone(
            induction_time_days,
            lower=CLASS_I_MAX_DAYS,
            upper=threshold_iii,
            censored=induction_time_censored,
        )

    # Tavanomainen (25 °C / 60 % RH).
    if _is_ambient(storage_T_C, storage_RH_percent):
        threshold_iii = CLASS_III_MIN_DAYS_AMBIENT  # 365
        if induction_time_days >= threshold_iii:
            return _classify_class_iii_zone(
                induction_time_days, threshold_iii, induction_time_censored
            )
        # 7 <= t < 365 -> Class II tavanomaisessa.
        return _classify_class_ii_zone(
            induction_time_days,
            lower=CLASS_I_MAX_DAYS,
            upper=threshold_iii,
            censored=induction_time_censored,
        )

    # --- Sääntö 4: tuntemattomat säilytysolot ---------------------------------
    # Class III -kriteeri vaatii olojen vahvistuksen (40/75 tai 25/60). Jos
    # oloja ei tunneta, emme voi todeta Class III:a edes pitkällä induktioajalla.
    # Class II:n välillä 7 <= t < 180 voidaan kuitenkin sanoa "todennäköisesti
    # Class II" matalalla luottamuksella.
    if CLASS_I_MAX_DAYS <= induction_time_days < CLASS_II_MAX_DAYS_ACCELERATED:
        return 2, "low"

    # >= 180 vrk tuntemattomissa oloissa: Class III edellyttää oloja,
    # joita ei tunneta -> emme luokittele.
    return None, "low"


def _classify_class_iii_zone(
    induction_time_days: float,
    threshold_iii: float,
    censored: Optional[bool],
) -> tuple[int, str]:
    """Apufunktio: luokittelu kun induktioaika on >= Class III -kynnys.

    Sensurointi muuttaa tulkintaa rajalla:

    * ``censored=True`` — kokeen kesto vahvistettu vähintään ``threshold_iii``
      vrk ilman kiteytymistä. Tämä on suora todiste Class III -kriteeristä
      ("ei kiteytymistä >= ...") -> **Class III high**, myös tasan rajalla.
    * ``censored=False`` — kiteytyminen havaittu rajan kohdalla. Tasan
      kynnyksellä tai sen alla ±10 % vyöhykkeellä näyte ei täytä
      "ei kiteytymistä >= ..." -kriteeriä, koska kiteytyminen tapahtui.
      -> **Class II borderline**.
    * ``censored=None`` — sensurointitietoa ei ole. Säilytetään aiempi
      semantiikka: borderline-vyöhykkeellä Class III borderline,
      muuten Class III high.
    """
    in_borderline = _is_borderline(induction_time_days, threshold_iii)

    if censored is True:
        return 3, "high"

    if censored is False:
        if in_borderline:
            # Kiteytyminen havaittiin rajalla -> ei täytä Class III:a.
            return 2, "borderline"
        # Kiteytyminen havaittiin selvästi rajan yli (esim. 365 vrk 40/75):
        # näyte oli amorfinen >= threshold_iii vrk, joten Class III täyttyy.
        return 3, "high"

    # censored is None
    confidence = "borderline" if in_borderline else "high"
    return 3, confidence


def _classify_class_ii_zone(
    induction_time_days: float,
    lower: float,
    upper: float,
    censored: Optional[bool],
) -> tuple[int, str]:
    """Apufunktio: luokittelu kun induktioaika on Class II -alueella.

    Borderline-tarkistus tehdään **molempien** rajojen ympärillä:

    * lähellä ``lower`` (= 7 vrk) -> rajatapaus Class I/II
    * lähellä ``upper`` (= 180 tai 365 vrk) -> rajatapaus Class II/III

    Sensuroitu havainto tällä alueella tarkoittaa, että koe loppui
    ennen kiteytymistä; stabiilisuusaika voi olla pidempi -> Class II,
    mutta luottamus alennetaan ``'low'``:ksi.
    """
    if censored is True:
        # Sensuroitu: aika on alaraja, todellinen stabiilisuus voi olla
        # suurempikin -> luokitus epävarma.
        return 2, "low"

    is_near_lower = _is_borderline(induction_time_days, lower)
    is_near_upper = _is_borderline(induction_time_days, upper)
    confidence = "borderline" if (is_near_lower or is_near_upper) else "high"
    return 2, confidence
