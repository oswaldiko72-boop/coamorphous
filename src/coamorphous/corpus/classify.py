"""GFA- ja stabiilisuusluokitus erotettuna kahteen riippumattomaan
kohdemuuttujaan.

MIKSI tämä moduuli on olemassa
------------------------------
Aiemmin yksittäinen ``classify_baird_taylor``-funktio tuotti kahta eri
ilmiötä sekoittavan luokan:

* **Lasinmuodostumiskyky (GFA)** — molekyylin sisäinen ominaisuus, joka
  havaitaan DSC-syklissä (heating-cooling-reheating). Baird et al.
  *J. Pharm. Sci.* **99** (2010) 3787-3806 (DOI 10.1002/jps.22197)
  määrittelee:

    * Class I  — kiteytyy DSC-jäähdytyksellä (huono lasinmuodostaja).
    * Class II — ei kiteydy jäähdytyksellä, mutta uudelleenkiteytyy
      uudelleenlämmityksessä (keskinkertainen).
    * Class III — ei kiteydy jäähdytyksellä eikä uudelleenlämmityksessä
      (hyvä lasinmuodostaja).

* **Stabiilisuus säilytysoloissa** — kontekstiriippuvainen mittaus,
  jossa amorfisesta näytteestä mitataan induktioaika (kiteytymisen
  alkamishetki) ICH Q1A -tyyppisissä oloissa. Tämä on aikasarja, ei
  kiinteä molekyylin ominaisuus.

Sekoittaminen oli haitallista, koska:

* Sama lääkepari voi olla GFA Class III (intrinsisesti hyvä lasinmuodostaja)
  mutta silti epästabiili 40/75-oloissa kosteuden vaikutuksen takia.
* ML-malli oppi sekoitelman, jolla on huono yleistettävyys uusiin
  säilytysoloihin: lopulta ei tiedetty, ennustaako malli kemiaa vai
  protokollaa.
* Luokitusrajojen muuttaminen (esim. uusi konsensus) vaati molempien
  ilmiöiden uudelleenajon yhdessä.

Ratkaisu: kaksi erillistä luokituspolkua, jotka voi yhdistää myöhemmin
hybridimallissa (H4):

    classify_gfa_dsc(...)          -> (gfa_class, gfa_label_confidence, gfa_dsc_evidence)
    classify_stability_week_bin(...) -> str
    classify_stability_protocol(...) -> str
    classify_stability_label_confidence(...) -> str

Periaatteet
-----------
* Puhdas, sivuvaikutukseton funktio (helppo testata).
* Eksplisiittiset kynnysarvot modulivakioina (ei "taikanumeroita").
* Rajatapauksissa palautetaan matala luottamus eikä keksitä luokituksia.

Reference
---------
* Baird, J. A.; Van Eerdenbrugh, B.; Taylor, L. S. *J. Pharm. Sci.*
  **99** (2010) 3787-3806 (GFA-luokitus, DOI 10.1002/jps.22197).
* Baird, J. A. & Taylor, L. S. *Adv. Drug Deliv. Rev.* **64** (2012)
  396-421 (stabiilisuusprotokollat, ICH Q1A -konteksti).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Stabiilisuusprotokollien tunnistuksen kynnykset.
#
# Käytämme väljiä rajoja, koska artikkeleissa on usein "40 ± 2 °C" tai
# "75 ± 5 % RH". Tarkat ICH Q1A:n nominaaliarvot ovat 40/75 (accelerated)
# ja 25/60 (long-term).
# -----------------------------------------------------------------------------
ACCELERATED_T_C_MIN: float = 35.0
ACCELERATED_T_C_MAX: float = 45.0
ACCELERATED_RH_MIN: float = 70.0
ACCELERATED_RH_MAX: float = 80.0

AMBIENT_T_C_MIN: float = 20.0
AMBIENT_T_C_MAX: float = 30.0
AMBIENT_RH_MIN: float = 55.0
AMBIENT_RH_MAX: float = 65.0

# Kuiva säilytys: silica gel / P2O5 -desikkaattori, RH lähellä nollaa.
# Käytetään erottamaan ICH-protokollista, joissa RH on 60-75 %.
DRY_RH_MAX: float = 10.0

# Sallittujen experimental_protocol-arvojen joukko, peilaa
# configs/corpus_schema.yaml:n experimental_protocol-enumia.
ALLOWED_PROTOCOLS: frozenset[str] = frozenset(
    {
        "ich_q1a_accelerated",
        "ich_q1a_long_term",
        "dry_short_term",
        "tg_plus_15K",
        "dsc_in_situ",
        "above_tg_kinetic",
        "non_standard",
    }
)

# Kynnys above_tg_kinetic-tunnistukselle: jos säilytys-T on yli Tg + 15 K,
# kyseessä on kineettinen kiteytymiskoe (BDS, isothermal DSC) eikä
# pitkäaikainen säilyvyyskoe — Baird-Taylor (2012) ICH Q1A -konsensus ei
# kata tällaisia oloja. Raja on EKSKLUUSIIVINEN: tasan Tg+15 K -mittaus
# on yhä tg_plus_15K-rajatapaus, vasta yli sen menee above_tg_kinetic:ksi.
ABOVE_TG_DELTA_K: float = 15.0

# -----------------------------------------------------------------------------
# Stabiilisuusajan binitys (induction_time_days -> stability_week_bin).
#
# MIKSI binitys: induction_time_days on hyvin vinoutunut jakauma
# (muutamasta päivästä useaan vuoteen) ja ML-mallit (erityisesti
# satunnaismetsä ja XGBoost) hyötyvät jaettujen kohdemuuttujien
# päättelystä. Lisäksi binit ovat ihmisluettavia raporteissa.
#
# Bin-rajat ovat **inklusiiviset alarajalla, eksklusiiviset ylärajalla**:
# (lower, upper, label) — t kuuluu biniin jos lower <= t < upper.
# Viimeinen ">=12m" -bin on inklusiivinen kummastakin päästä.
# -----------------------------------------------------------------------------
WEEK_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 7.0, "<1w"),
    (7.0, 14.0, "1-2w"),
    (14.0, 21.0, "2-3w"),
    (21.0, 28.0, "3-4w"),
    (28.0, 60.0, "1-2m"),
    (60.0, 90.0, "2-3m"),
    (90.0, 120.0, "3-4m"),
    (120.0, 150.0, "4-5m"),
    (150.0, 180.0, "5-6m"),
    (180.0, 210.0, "6-7m"),
    (210.0, 240.0, "7-8m"),
    (240.0, 270.0, "8-9m"),
    (270.0, 300.0, "9-10m"),
    (300.0, 335.0, "10-11m"),
    (335.0, 365.0, "11-12m"),
)
WEEK_BIN_OVER_YEAR: str = ">=12m"
WEEK_BIN_UNKNOWN: str = "unknown"


# =============================================================================
# GFA-luokitus (DSC-pohjainen, intrinsinen molekyylin ominaisuus)
# =============================================================================


def classify_gfa_dsc(
    crystallizes_on_cooling: Optional[bool] = None,
    crystallizes_on_reheating: Optional[bool] = None,
    paper_states_class: Optional[int] = None,
) -> tuple[Optional[int], str, str]:
    """Luokittele lasinmuodostumiskyky (GFA) DSC-syklin perusteella.

    Logiikka noudattaa Baird et al. *J. Pharm. Sci.* **99** (2010)
    3787-3806 -määrittelyä:

    * Class I  — kiteytyy DSC-jäähdytyksellä.
    * Class II — ei kiteydy jäähdytyksellä, kiteytyy uudelleenlämmityksessä.
    * Class III — ei kiteydy jäähdytyksellä eikä uudelleenlämmityksessä.

    Tämä on **molekyylin sisäinen** ominaisuus, ei säilytysolojen
    stabiilisuusmittaus. Erityisesti: Class III -aine voi silti olla
    epästabiili kosteissa oloissa, ja Class I -aine voi olla "stabiili"
    kuivassa kylmässä — GFA ei suoraan ennusta säilytysstabiiliutta,
    mutta molemmat on syytä tietää.

    Parameters
    ----------
    crystallizes_on_cooling : bool, optional
        Kiteytyikö näyte DSC-jäähdytyssyklissä? ``True`` -> suora
        Class I -todiste. ``None`` jos ei raportoitu.
    crystallizes_on_reheating : bool, optional
        Kiteytyikö näyte DSC-uudelleenlämmityksessä? Käytetään vain jos
        ``crystallizes_on_cooling=False`` (Class II vs III erotus).
        ``None`` jos uudelleenlämmityssykliä ei raportoitu.
    paper_states_class : int, optional
        Jos artikkeli ilmoittaa luokan (1, 2, tai 3) ilman DSC-syklin
        yksityiskohtia, tämä arvo käytetään tietolähteenä.

    Returns
    -------
    gfa_class : int or None
        1, 2, 3 tai ``None`` jos dataa ei riitä.
    gfa_label_confidence : str
        ``"high"`` kun täysi DSC-sykli tai eksplisiittinen lähteen maininta,
        ``"low"`` kun päätelty epäsuorasti (osittainen DSC + paper-täydennys),
        ``"unknown"`` kun gfa_class ei voitu määrittää.
    gfa_dsc_evidence : str
        Lähde, josta luokitus johdettiin: ``"dsc_cycle_full_reported"`` (paras),
        ``"dsc_thermogram_inferred"``, ``"stated_explicitly"`` tai
        ``"not_reported"``.

    Notes
    -----
    Class II vs III erotus vaatii uudelleenlämmityssyklin havainnon. Jos
    ``crystallizes_on_cooling=False`` mutta ``crystallizes_on_reheating=None``,
    luokitus voi olla joko Class II tai Class III — tällöin palautetaan
    ``paper_states_class`` jos se on annettu (``low`` confidence,
    ``dsc_thermogram_inferred``-evidence), muuten ``None`` /
    ``unknown``-confidence.

    Examples
    --------
    Täysi DSC-sykli, kiteytyy jäähdytyksellä = Class I:

    >>> classify_gfa_dsc(crystallizes_on_cooling=True,
    ...                  crystallizes_on_reheating=False)
    (1, 'high', 'dsc_cycle_full_reported')

    Täysi sykli, ei kiteydy ollenkaan = Class III:

    >>> classify_gfa_dsc(crystallizes_on_cooling=False,
    ...                  crystallizes_on_reheating=False)
    (3, 'high', 'dsc_cycle_full_reported')

    Vain artikkelin eksplisiittinen ilmoitus = korkea luottamus:

    >>> classify_gfa_dsc(paper_states_class=3)
    (3, 'high', 'stated_explicitly')

    Ei mitään tietoa:

    >>> classify_gfa_dsc()
    (None, 'unknown', 'not_reported')
    """
    # --- Sääntö 1: täysi DSC-sykli antaa varmimman luokituksen --------------
    if crystallizes_on_cooling is True:
        # Kiteytyy jäähdytyksellä = Class I, riippumatta uudelleenlämmityksestä.
        # Evidence on dsc_cycle_full_reported jos uudelleenlämmityskin raportoitu;
        # muutoin dsc_thermogram_inferred (yksittäisen termogrammin piirre).
        if crystallizes_on_reheating is not None:
            return 1, "high", "dsc_cycle_full_reported"
        return 1, "high", "dsc_thermogram_inferred"

    if crystallizes_on_cooling is False:
        if crystallizes_on_reheating is True:
            return 2, "high", "dsc_cycle_full_reported"
        if crystallizes_on_reheating is False:
            return 3, "high", "dsc_cycle_full_reported"
        # cooling=False mutta reheating=None: voi olla joko Class II tai III.
        # Käytetään paper_states_class jos saatavilla (matala luottamus, koska
        # paperin maininta täydentää osittaista DSC-tietoa). Muuten ei voi luokitella.
        if paper_states_class in (2, 3):
            return paper_states_class, "low", "dsc_thermogram_inferred"
        logger.debug(
            "classify_gfa_dsc: crystallizes_on_cooling=False mutta "
            "uudelleenlämmityksen tieto puuttuu eikä artikkelin luokkaa "
            "ole annettu -> ei voi erottaa Class II/III."
        )
        return None, "unknown", "dsc_thermogram_inferred"

    # --- Sääntö 2: cooling=None, käytetään artikkelin ilmoitusta ------------
    # Eksplisiittinen lähteen maininta ("Class III glass former" tms.) on
    # luotettava signaali — luotetaan siihen korkealla confidencellä.
    if paper_states_class in (1, 2, 3):
        return paper_states_class, "high", "stated_explicitly"

    # --- Sääntö 3: ei dataa ollenkaan ---------------------------------------
    return None, "unknown", "not_reported"


# =============================================================================
# Stabiilisuusajan binitys
# =============================================================================


def classify_stability_week_bin(
    induction_time_days: Optional[float],
    induction_time_censored: Optional[bool] = False,
) -> str:
    """Bini induction_time_days -arvo diskreettiin viikko-/kuukausikoteloon.

    Käytetään stability-mallin kohdemuuttujana. Censored-tieto pidetään
    erillisenä (``induction_time_censored``-sarakkeessa); tämä funktio
    palauttaa vain bin-merkin.

    Parameters
    ----------
    induction_time_days : float, optional
        Aika kiteytymisen alkuun vuorokausina. ``None`` -> ``"unknown"``.
    induction_time_censored : bool, optional
        Right-censoring -lippu. **Ei vaikuta bin-arvoon** — bin perustuu
        raakaan päiväarvoon. Censored-status säilytetään kokonaan
        erillään, jotta ML-malli voi käyttää sitä piirteenä.
        Parametri otetaan vastaan johdonmukaisuussyistä, jotta kutsuvat
        funktiot voivat välittää sen ilman tarkistusta.

    Returns
    -------
    str
        Bin-merkki ``configs/corpus_schema.yaml``:n
        ``stability_week_bin``-enumista, esim. ``"6-7m"`` tai ``">=12m"``.

    Raises
    ------
    ValueError
        Jos ``induction_time_days`` on negatiivinen.

    Examples
    --------
    >>> classify_stability_week_bin(3.0)
    '<1w'
    >>> classify_stability_week_bin(60.0)
    '2-3m'
    >>> classify_stability_week_bin(186.0)
    '6-7m'
    >>> classify_stability_week_bin(400.0)
    '>=12m'
    >>> classify_stability_week_bin(None)
    'unknown'
    """
    # Censored-parametri on signature-tasolla mukana, jotta kutsuva koodi
    # voi yhtenäisesti välittää saman arvon useille classify_*-funktioille.
    # Itse binitys ei kuitenkaan riipu siitä — censored-status näkyy CSV:ssä
    # erillisessä induction_time_censored-sarakkeessa.
    del induction_time_censored

    if induction_time_days is None:
        return WEEK_BIN_UNKNOWN

    if induction_time_days < 0:
        raise ValueError(
            f"induction_time_days ei voi olla negatiivinen, sai: {induction_time_days}"
        )

    for lower, upper, label in WEEK_BINS:
        if lower <= induction_time_days < upper:
            return label

    # Kaikki yli 365 vrk:n havainnot menevät yhteen >=12m-koteloon.
    return WEEK_BIN_OVER_YEAR


# =============================================================================
# Stabiilisuusprotokollan tunnistus
# =============================================================================


def classify_stability_protocol(
    storage_T_C: Optional[float],
    storage_RH_percent: Optional[float],
    Tg_K: Optional[float] = None,
    experimental_protocol: Optional[str] = None,
) -> str:
    """Tunnista, mitä standardiprotokollatyyppiä säilytysolot vastaavat.

    Prioriteetti:

    1. Jos ``storage_T_C`` ja ``Tg_K`` molemmat saatavilla ja
       T > Tg + 15 K, palautetaan ``"above_tg_kinetic"`` riippumatta
       ekstraktoijan ilmoituksesta. Tämä estää, että BDS:ssä tai
       isotermisessä DSC:ssä mitattu kineettinen kiteytymiskoe
       sekoitetaan ICH Q1A -tyyppiseen säilyvyyskokeeseen ML-mallin
       opetuksessa. Olot Tg:n yläpuolella eivät ennusta käytännön
       varastointistabiliteettia.
    2. Jos ``experimental_protocol`` on annettu eksplisiittisesti ja se on
       sallittu enum-arvo, palautetaan se sellaisenaan (kutsuva ekstraktoija
       on tehnyt päätöksen).
    3. Muuten päätellään storage_T_C/storage_RH-arvoista.

    Parameters
    ----------
    storage_T_C : float, optional
        Säilytyslämpötila celsiuksina.
    storage_RH_percent : float, optional
        Suhteellinen kosteus prosentteina.
    Tg_K : float, optional
        Lasittumislämpötila kelvineinä. Jos saatavilla, käytetään
        above_tg_kinetic-tunnistukseen vertailemalla
        ``storage_T_K = storage_T_C + 273.15`` arvoon ``Tg_K + 15.0``.
        Jos ``None``, ehto ohitetaan ja luokitus etenee normaalisti.
    experimental_protocol : str, optional
        Ekstraktoijan ilmoittama protokolla (``configs/corpus_schema.yaml``:n
        ``experimental_protocol``-enum). Jos annettu, käytetään sellaisenaan
        (paitsi above_tg_kinetic-ehto ohittaa myös tämän — datan eheys
        prioriteetiltaan ekstraktoijan päätöksen yli).

    Returns
    -------
    str
        Yksi seuraavista: ``"ich_q1a_accelerated"``, ``"ich_q1a_long_term"``,
        ``"dry_short_term"``, ``"tg_plus_15K"``, ``"dsc_in_situ"``,
        ``"above_tg_kinetic"``, ``"non_standard"``.

    Notes
    -----
    Pääsääntö (Tg-ehdon ulkopuolella): jos olot eivät täsmää ICH Q1A:n
    40/75:ään tai 25/60:een eikä eksplisiittistä protokollaa ole,
    palautetaan ``"non_standard"``. Kuivasäilytys (RH <= 10 %) ilman
    eksplisiittistä protokollaa luokitellaan ``"dry_short_term"``-
    ehdokkaaksi vain jos ekstraktoija on niin merkinnyt.

    Examples
    --------
    >>> classify_stability_protocol(40.0, 75.0)
    'ich_q1a_accelerated'
    >>> classify_stability_protocol(25.0, 60.0)
    'ich_q1a_long_term'
    >>> classify_stability_protocol(4.0, 0.0)
    'non_standard'
    >>> classify_stability_protocol(None, None, experimental_protocol="dry_short_term")
    'dry_short_term'
    >>> classify_stability_protocol(99.85, None, Tg_K=323.0)
    'above_tg_kinetic'
    """
    # 1) above_tg_kinetic ohittaa kaiken muun: kineettinen koe Tg:n yläpuolella
    # ei ole säilyvyyskoe, vaikka ekstraktoija olisi sen sellaiseksi merkinnyt.
    if storage_T_C is not None and Tg_K is not None:
        storage_T_K = storage_T_C + 273.15
        if storage_T_K > Tg_K + ABOVE_TG_DELTA_K:
            return "above_tg_kinetic"

    # 2) Eksplisiittinen protokolla ohittaa olojen päättelyn, kunhan se on
    # sallittu enum-arvo.
    if experimental_protocol in ALLOWED_PROTOCOLS:
        return experimental_protocol

    # 3) Päättele oloista.
    if storage_T_C is None or storage_RH_percent is None:
        return "non_standard"

    if (
        ACCELERATED_T_C_MIN <= storage_T_C <= ACCELERATED_T_C_MAX
        and ACCELERATED_RH_MIN <= storage_RH_percent <= ACCELERATED_RH_MAX
    ):
        return "ich_q1a_accelerated"

    if (
        AMBIENT_T_C_MIN <= storage_T_C <= AMBIENT_T_C_MAX
        and AMBIENT_RH_MIN <= storage_RH_percent <= AMBIENT_RH_MAX
    ):
        return "ich_q1a_long_term"

    return "non_standard"


# =============================================================================
# Stabiilisuusluokituksen luotettavuus
# =============================================================================


def classify_stability_label_confidence(
    protocol_match: str,
    experimental_protocol: Optional[str] = None,
) -> str:
    """Pääsuhteena ICH Q1A -protokollat saavat 'high', muut 'low'.

    Parameters
    ----------
    protocol_match : str
        ``classify_stability_protocol``-funktion palauttama arvo.
    experimental_protocol : str, optional
        Alkuperäinen protokollatieto. Hyväksytään parametrina
        johdonmukaisuussyistä, mutta logiikka katsoo ensisijaisesti
        ``protocol_match``-arvoa, koska se on jo normalisoitu.

    Returns
    -------
    str
        ``"high"`` jos ``protocol_match`` on ``"ich_q1a_accelerated"`` tai
        ``"ich_q1a_long_term"``, muuten ``"low"``.

    Notes
    -----
    MIKSI vain ICH-protokollat saavat 'high':
    Baird-Taylor (2012) -konsensus käyttää ICH Q1A:n 40/75 ja 25/60 -oloja
    standardina. Muut protokollat (kuivasäilytys, Tg+15 K, DSC in situ,
    non_standard) ovat tutkimusryhmäkohtaisia eivätkä takaa, että näytteen
    käyttäytyminen yleistyy normaaleihin lääkesäilytysoloihin.
    """
    del experimental_protocol  # ei käytetä; kutsuja saa välittää sen vapaasti
    if protocol_match in ("ich_q1a_accelerated", "ich_q1a_long_term"):
        return "high"
    return "low"
