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

Sensuroitu data ei-standardiprotokollissa
------------------------------------------
Erikoistapaus: kun ``censored=True``, ``t >= 180 vrk`` ja säilytysolot
**eivät** vastaa ICH Q1A:n 40/75:tä tai 25/60:tä (esim. silica gel
-desikkaattori 4 °C:ssa tai 25 °C / 0 % RH 6 kk:n ajan), näyte
luokitellaan Class III:ksi ``label_confidence='low'``-painolla. Aiempi
logiikka palautti (None, 'low') ja näin pudotti **stabiileimmat
näytteet** opetusjoukon ulkopuolelle — vaikka ne ovat ML-mallin
kannalta arvokkaimpia (vahvin todiste hyvästä GFA:sta). Matala
luottamus heijastaa, että kuiva/kylmä säilytys on lievempi olo kuin
ICH-kiihdytetty, joten 180 vrk näissä oloissa ei takaa, että näyte
selviäisi 40/75:ssä yhtä pitkään. Painottamalla tällaiset rivit
opetuksessa pienemmällä kertoimella saadaan kuitenkin signaali
hyödynnettyä, ei kokonaan hylättyä.

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

# -----------------------------------------------------------------------------
# Ei-standardiprotokollien kynnykset (kuiva lyhytaikainen säilytys).
#
# MIKSI omat vakiot:
#   Kuivasäilytys (P2O5 / silica gel, RH ≈ 0 %) on lievempi olo kuin 25/60 tai
#   40/75, joten näytteen kestäminen 14 vrk kuivassa ei ole yhtä vahva todiste
#   stabiilisuudesta kuin 180 vrk kiihdytetyssä. Käytämme heuristisia rajoja:
#
#     * >= 14 vrk sensuroituna -> "luultavasti Class III" (low confidence)
#     * 7-14 vrk sensuroituna -> "luultavasti Class II" (low confidence)
#     * < 7 vrk havaittu kiteytyminen -> Class I (low confidence)
#     * >= 7 vrk havaittu kiteytyminen -> Class II (low confidence)
#
# Nämä kynnykset perustuvat Löbmann 2011/2013 -tyyppisten kuivakokeiden
# käytäntöön: tutkimukset kestävät usein 21 vrk ja oletetaan, että 2-3 viikkoa
# kuivassa amorfisena säilyminen ennustaa pidempiaikaista stabiiliutta.
# Kaikki dry_short_term-luokitukset saavat label_confidence='low' koska
# ennustaminen ei-standardiprotokollasta on epävarmaa.
# -----------------------------------------------------------------------------
DRY_SHORT_TERM_CLASS_III_PROXY_DAYS: float = 14.0
DRY_SHORT_TERM_CLASS_II_PROXY_DAYS: float = 7.0


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
    experimental_protocol: Optional[str] = None,
    protocol_max_duration_days: Optional[float] = None,
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
    experimental_protocol : str, optional
        Stabiilisuusprotokollan tunniste. Vastaa YAML:n
        ``experimental_protocol`` -enumia. Vaikuttaa luokituslogiikkaan:

        * ``"ich_q1a_accelerated"``, ``"ich_q1a_long_term"``, ``None`` —
          käytetään standardia Baird-Taylor -logiikkaa muuttumattomana.
          (None säilyttää aiemman, taaksepäin yhteensopivan käyttäytymisen,
          jos protokollatieto ei ole saatavilla.)
        * ``"dry_short_term"`` — kuivasäilytys (RH ≈ 0 %), oma kynnyssarja,
          aina ``label_confidence='low'``.
        * ``"tg_plus_15K"`` — Class I:n alkuperäinen kineettinen testi.
          < 7 vrk -> (1, 'high'); muuten (2, 'low') koska tämä protokolla
          ei voi vahvistaa Class III:a.
        * ``"dsc_in_situ"`` — vain DSC-kiteytymistarkistus. Ei riittäviä
          tietoja muuhun kuin Class I:een (DSC-kiteytymisestä).
        * ``"non_standard"`` — käytetään standardilogiikkaa, mutta
          ``label_confidence`` pakotetaan ``'low'``:ksi. Erikoistapaus:
          jos säilytysolot eivät ole 40/75 eivätkä 25/60 (esim. silica
          gel -desikkaattori 4 °C / 0 % RH) ja ``induction_time_censored=True``
          ja ``induction_time_days >= 180``, palautetaan ``(3, 'low')``.
          Tämä säilyttää ML-mallin kannalta arvokkaimmat näytteet
          (vahvin todiste hyvästä GFA:sta) opetusjoukossa
          label_confidence-painokertoimella, sen sijaan että ne
          tippuisivat (None, 'low')-tuloksen takia kokonaan ulos.
    protocol_max_duration_days : float, optional
        Kokeilun maksimikesto vuorokausina. Käytetään johdonmukaisuus-
        tarkistukseen: jos ``induction_time_censored=True``, induktioajan
        täytyy olla ``<= protocol_max_duration_days`` (muuten ekstraktio
        on epäjohdonmukainen ja nostetaan ValueError).

    Returns
    -------
    gfa_class : int or None
        1, 2, tai 3 — Baird-Taylor luokka. ``None`` jos dataa ei riitä
        luokituksen tekemiseen (esim. >= 180 vrk tuntemattomissa oloissa,
        jolloin Class III -kriteeriä ei voida vahvistaa).
    label_confidence : {'high', 'low', 'borderline'}
        ``'high'`` selkeissä tapauksissa, ``'borderline'`` raja-arvojen
        ±10 % vyöhykkeellä, ``'low'`` kun datan puute tai ei-standardi-
        protokolla pakottaa arvauksen.

    Raises
    ------
    ValueError
        Jos ``induction_time_days`` on negatiivinen, tai jos sensuroitu
        induktioaika ylittää ``protocol_max_duration_days``:n
        (epäjohdonmukainen ekstraktio).

    Notes
    -----
    Sääntöjen järjestys on tärkeä:

    1. DSC-kiteytyminen testataan ensin (ohittaa kaiken muun).
    2. Jos ``protocol_max_duration_days`` annettu, sen ja induktioajan
       johdonmukaisuus tarkistetaan.
    3. Jos ``experimental_protocol`` viittaa ei-standardiin protokollaan
       (``dry_short_term``, ``tg_plus_15K``, ``dsc_in_situ``), käytetään
       sen omaa logiikkaa ja palautetaan aina ``label_confidence='low'``
       paitsi tg_plus_15K + < 7 vrk -tapauksessa, jossa Class I voidaan
       todeta korkealla luottamuksella.
    4. Muuten (None tai ich_q1a_*) käytetään standardia logiikkaa.
    5. ``non_standard`` käyttää standardilogiikkaa mutta pakottaa
       ``label_confidence='low'``. Tuntemattomissa säilytysoloissa
       (ei 40/75 eikä 25/60), jos ``induction_time_censored=True`` ja
       ``induction_time_days >= 180``, palautetaan ``(3, 'low')`` —
       muuten standardilogiikka palauttaisi ``(None, 'low')`` ja
       stabiileimmat näytteet tippuisivat opetusjoukosta pois.

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

    Tuntemattomat olot ja pitkä induktioaika eivät riitä luokitukseen,
    jos sensurointitietoa ei ole:

    >>> classify_baird_taylor(200.0, None, None, False)
    (None, 'low')

    Sensuroitu pitkä koe (>= 180 vrk) tuntemattomissa oloissa
    (non_standard-protokolla, esim. 4 °C / 0 % RH silica gel) — paras
    todiste GFA:sta, säilytetään opetusjoukossa low-painolla:

    >>> classify_baird_taylor(186.0, 4.0, 0.0, False, induction_time_censored=True,
    ...                       experimental_protocol="non_standard")
    (3, 'low')

    Kuivasäilytys (Löbmann 2011/2013 -tyylinen): 21 vrk sensuroituna
    P2O5-desikaattorissa antaa "todennäköisesti Class III" matalalla
    luottamuksella:

    >>> classify_baird_taylor(21.0, 25.0, 0.0, False, induction_time_censored=True,
    ...                       experimental_protocol="dry_short_term")
    (3, 'low')

    Kuivasäilytys, nopea kiteytyminen:

    >>> classify_baird_taylor(3.0, 25.0, 0.0, False, induction_time_censored=False,
    ...                       experimental_protocol="dry_short_term")
    (1, 'low')

    Tg+15 K -kineettinen testi, kiteytyminen alle viikossa = Class I:n
    suora todiste:

    >>> classify_baird_taylor(5.0, None, None, False,
    ...                       experimental_protocol="tg_plus_15K")
    (1, 'high')
    """
    # --- Sääntö 1: DSC-jäähdytyksellä havaittu kiteytyminen = Class I ----------
    # MIKSI ensin: DSC-kiteytyminen on suorin testi heikolle GFA:lle, eikä
    # induktioaikadata voi kumota sitä.
    if crystallizes_on_dsc_cooling is True:
        return 1, "high"

    # Ilman induktioaikaa emme voi luokitella muiden sääntöjen perusteella.
    # Poikkeus: dsc_in_situ-protokollalla DSC-kiteytymisen puuttuminen ei
    # yksinään riitä luokitukseen, joten palautetaan (None, 'low').
    if induction_time_days is None:
        if experimental_protocol == "dsc_in_situ":
            # crystallizes_on_dsc_cooling on jo testattu yllä; tähän tultaessa
            # se on False tai None eikä induktioaikaa ole -> ei voida luokitella.
            logger.debug(
                "classify_baird_taylor: dsc_in_situ ilman kiteytymistä eikä "
                "induktioaikaa -> ei voi luokitella."
            )
            return None, "low"
        logger.debug(
            "classify_baird_taylor: induktioaika puuttuu, ei voi luokitella."
        )
        return None, "low"

    if induction_time_days < 0:
        raise ValueError(
            f"induction_time_days ei voi olla negatiivinen, sai: {induction_time_days}"
        )

    # --- Sääntö 2: protocol_max_duration_days -johdonmukaisuus ----------------
    # MIKSI: jos koe sensuroitiin ennen kiteytymistä, induktioajan pitää olla
    # <= kokeen kokonaiskesto. Päinvastainen merkitsisi ekstraktiovirhettä.
    # Tämä tarkistus on uusi turvaverkko ekstraktorille.
    if (
        induction_time_censored is True
        and protocol_max_duration_days is not None
        and induction_time_days > protocol_max_duration_days
    ):
        raise ValueError(
            f"induction_time_days ({induction_time_days}) ylittää "
            f"protocol_max_duration_days ({protocol_max_duration_days}), "
            f"vaikka induction_time_censored=True. Sensuroitu havainto ei voi "
            f"olla pidempi kuin kokeen kokonaiskesto."
        )

    # --- Sääntö 3: ei-standardit protokollat ---------------------------------
    # MIKSI omat haarat: kuivasäilytys, Tg+15 K -testi ja dsc_in_situ käyttävät
    # eri kynnyksiä kuin ICH Q1A. Standardilogiikka soveltuu vain
    # ich_q1a_*-protokolliin tai None:lle (taaksepäin yhteensopivuus).
    if experimental_protocol == "dry_short_term":
        return _classify_dry_short_term(induction_time_days, induction_time_censored)

    if experimental_protocol == "tg_plus_15K":
        return _classify_tg_plus_15K(induction_time_days)

    if experimental_protocol == "dsc_in_situ":
        # crystallizes_on_dsc_cooling=True käsiteltiin jo Säännössä 1 (Class I).
        # Tähän tultaessa se on False tai None ja meillä on induktioaika —
        # mutta dsc_in_situ-protokolla ei oletuksena tuota induktioaika-
        # mittausta. Jos sellainen kuitenkin on (vapaakirjattu), palautetaan
        # konservatiivisesti (None, 'low') koska protokolla ei tue sitä.
        logger.debug(
            "classify_baird_taylor: dsc_in_situ-protokollalla ei DSC-kiteytymistä; "
            "ei luokitella induktioajan perusteella."
        )
        return None, "low"

    # --- Sääntö 4: standardi logiikka (ICH Q1A, None tai non_standard) -------
    # Suoritetaan standardit kynnystestit ja viimeisenä alennetaan confidence
    # 'low':ksi, jos protokolla on eksplisiittisesti non_standard.
    gfa_class, confidence = _classify_standard(
        induction_time_days,
        storage_T_C,
        storage_RH_percent,
        induction_time_censored,
    )

    if experimental_protocol == "non_standard":
        # Ei-standardi protokolla: pakota label_confidence='low' riippumatta
        # standardilogiikan tuottamasta confidence-arvosta. gfa_class itsessään
        # voi pysyä, mutta sen luotettavuus on rajoitettu.
        return gfa_class, "low"

    return gfa_class, confidence


def _classify_standard(
    induction_time_days: float,
    storage_T_C: Optional[float],
    storage_RH_percent: Optional[float],
    censored: Optional[bool],
) -> tuple[Optional[int], str]:
    """Standardi Baird-Taylor -luokituslogiikka (ICH Q1A 40/75 tai 25/60).

    Erotettu omaksi funktiokseen, jotta ``non_standard``-protokolla voi
    käyttää samaa logiikkaa, mutta override-päättää lopullisen
    label_confidence-arvon.
    """
    # < 7 vrk -> Class I (sensuroinnilla erikoistapaus).
    if induction_time_days < CLASS_I_MAX_DAYS:
        if censored is True:
            # Sensuroitu < 7 vrk = koe loppui ennen 7 vrk:tä ilman
            # kiteytymistä. Stabiilisuusaika on alaraja -> ei voi luokitella.
            logger.debug(
                "_classify_standard: sensuroitu alle 7 vrk, ei voi luokitella."
            )
            return None, "low"
        confidence = (
            "borderline"
            if _is_borderline(induction_time_days, CLASS_I_MAX_DAYS)
            else "high"
        )
        return 1, confidence

    # Pitkäaikaissäilytys, kiihdytetty (40 °C / 75 % RH).
    if _is_accelerated(storage_T_C, storage_RH_percent):
        threshold_iii = CLASS_II_MAX_DAYS_ACCELERATED  # 180
        if induction_time_days >= threshold_iii:
            return _classify_class_iii_zone(
                induction_time_days, threshold_iii, censored
            )
        return _classify_class_ii_zone(
            induction_time_days,
            lower=CLASS_I_MAX_DAYS,
            upper=threshold_iii,
            censored=censored,
        )

    # Pitkäaikaissäilytys, tavanomainen (25 °C / 60 % RH).
    if _is_ambient(storage_T_C, storage_RH_percent):
        threshold_iii = CLASS_III_MIN_DAYS_AMBIENT  # 365
        if induction_time_days >= threshold_iii:
            return _classify_class_iii_zone(
                induction_time_days, threshold_iii, censored
            )
        return _classify_class_ii_zone(
            induction_time_days,
            lower=CLASS_I_MAX_DAYS,
            upper=threshold_iii,
            censored=censored,
        )

    # Sääntö: sensuroitu data tuntemattomissa oloissa — sallitaan Class III,
    # jos koe kesti vähintään 180 vrk ilman havaittua kiteytymistä.
    #
    # MIKSI tämä haara
    # ----------------
    # Aiempi logiikka palautti (None, 'low') aina kun säilytysolot eivät
    # täsmänneet ICH Q1A:n kiihdytettyihin (40/75) tai tavanomaisiin (25/60).
    # Tämä menettää **stabiileimpien näytteiden luokituksen**: esim. Allesø 2009:n
    # 1:1 NAP-CIM säilyi amorfisena 186 vrk:n ajan 4 °C / 0 % RH:ssa (kuiva
    # silica gel -desikkaattori). Tällainen näyte on ML-mallin näkökulmasta
    # **kullanarvoinen** — paras todiste vahvasta GFA:sta — mutta se merkittiin
    # (None, 'low') ja tippui näin opetusjoukosta pois.
    #
    # Sensuroitu havainto >= 180 vrk on suora todiste "ei kiteytymistä >= 180
    # vrk" -kriteeristä riippumatta tarkasta säilytyslämpötilasta tai
    # kosteudesta. Kuiva (RH ≈ 0 %) tai kylmä (4 °C) säilytys on lievempi
    # olosuhde kuin ICH:n kiihdytetty, joten näiden olojen 180 vrk on
    # todennäköisesti yliarvio "todellisesta" 40/75-stabiilisuudesta — tästä
    # syystä label_confidence pakotetaan 'low':ksi. Näin näyte säilyy
    # opetusjoukossa label_confidence-painokertoimella, eikä tippu kokonaan
    # ulos.
    if (
        censored is True
        and induction_time_days >= CLASS_II_MAX_DAYS_ACCELERATED
    ):
        return 3, "low"

    # Tuntemattomat säilytysolot, sensuroimaton tai lyhytaikainen havainto.
    # Class II:n alueella 7 <= t < 180 voidaan sanoa "todennäköisesti Class II".
    # Pidempi induktioaika ilman sensurointitietoa ei riitä Class III:n
    # vahvistamiseen ilman olojen kontekstia.
    if CLASS_I_MAX_DAYS <= induction_time_days < CLASS_II_MAX_DAYS_ACCELERATED:
        return 2, "low"

    return None, "low"


def _classify_dry_short_term(
    induction_time_days: float,
    censored: Optional[bool],
) -> tuple[Optional[int], str]:
    """Luokitus kuivasäilytyskokeille (P2O5 / silica gel, RH ≈ 0 %).

    MIKSI oma haara
    ---------------
    Kuivasäilytys on lievempi olo kuin ICH Q1A: matala kosteus poistaa
    veden plastisaatiovaikutuksen, joten näytteen pitäminen amorfisena
    2-3 viikkoa kuivassa ei ole sama todistusvoima kuin 6 kk 40/75:ssa.
    Käytämme käytännöllisiä kynnyksiä Löbmann 2011/2013 -tyyppisille
    21 vrk:n kokeille.

    Säännöt
    -------
    Sensuroitu (koe loppui ilman kiteytymistä):
      * >= 14 vrk -> Class III (low confidence) — kestänyt vähintään
        kaksi viikkoa kuivassa, todennäköisesti hyvä GFA
      * 7-14 vrk  -> Class II  (low confidence)
      * < 7 vrk   -> ei voi luokitella ((None, 'low'))

    Sensuroimaton (kiteytyminen havaittu):
      * < 7 vrk   -> Class I  (low confidence) — nopea kiteytyminen
        kuivassakin viittaa heikkoon GFA:han
      * >= 7 vrk  -> Class II (low confidence) — kesti vähintään viikon

    Sensurointitieto puuttuu (None): käsitellään kuten sensuroimaton —
    raportoitu aika tulkitaan havaintona.

    Kaikki tulokset saavat ``label_confidence='low'``, koska
    kuivasäilytyskokeen ennustearvo standardin kiihdytetyn (40/75) tai
    tavanomaisen (25/60) säilytyksen suhteen on epävarma.
    """
    if censored is True:
        if induction_time_days >= DRY_SHORT_TERM_CLASS_III_PROXY_DAYS:
            return 3, "low"
        if induction_time_days >= DRY_SHORT_TERM_CLASS_II_PROXY_DAYS:
            return 2, "low"
        # < 7 vrk sensuroitu kuivassa: koe loppui ennen kuin Class I -kynnys
        # edes saavutettiin. Stabiilisuusaika voi olla mikä tahansa -> emme
        # luokittele.
        logger.debug(
            "_classify_dry_short_term: sensuroitu alle 7 vrk, ei voi luokitella."
        )
        return None, "low"

    # censored is False tai None: tulkitaan raportoitu aika kiteytymishavainto.
    if induction_time_days < DRY_SHORT_TERM_CLASS_II_PROXY_DAYS:
        return 1, "low"
    return 2, "low"


def _classify_tg_plus_15K(induction_time_days: float) -> tuple[int, str]:
    """Luokitus Tg+15 K -kineettiselle testille.

    MIKSI oma haara
    ---------------
    Tg+15 K -testi on Class I:n alkuperäinen määrittely (Baird-Taylor 2012):
    näyte pidetään 15 K lasittumislämpötilan yläpuolella ja katsotaan,
    kiteytyykö se viikossa. Tämä on **suora** Class I -kriteeri, eikä
    pidempi induktioaika tässä testissä todista Class III:a — testi vain
    kertoo, onko aine Class I vai ei.

    Säännöt
    -------
      * < 7 vrk   -> (1, 'high') — Class I:n suora todiste
      * >= 7 vrk  -> (2, 'low')  — ei Class I; tarkkaa luokkaa II/III ei
        voi päätellä tästä testistä yksinään
    """
    if induction_time_days < CLASS_I_MAX_DAYS:
        return 1, "high"
    return 2, "low"


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
