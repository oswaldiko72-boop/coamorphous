"""Pydantic-skeema LLM-ekstraktion tuottamille raakatiedoille.

MIKSI tämä on rajatumpi kuin master-CSV
---------------------------------------
LLM (Claude Code) tuottaa lähdeartikkelista vain *raakatiedot*: lääkkeiden
nimet sellaisena kuin lähde ne kirjoittaa, suhteet, prosessikuvaus,
stabiiliusmittauksen tulokset ja säilytysolot. Kaikki *johdetut* arvot
— kanonisoitu SMILES, InChIKey, ``gfa_class``, ``label_confidence``,
``delta_MW``, ``tanimoto_*`` — lasketaan ohjelmallisesti
(``coamorphous.extraction.enrich`` ja ``coamorphous.descriptors``).

Tämä erottelu on kriittinen luotettavuuden kannalta: jos LLM "muistaisi"
SMILES-merkkijonon väärin, se voisi olla muuten validi mutta viitata
toiseen molekyyliin. Sama koskee gfa_class-luokitusta, jonka raja-arvot
on määritelty Baird-Taylor (2012):ssa eikä mallin kannata päättää siitä.

Pydantic v2:n etu vs. tavallinen dict
--------------------------------------
* Tyypit pakotetaan jo Pythonin puolella (esim. ``mole_fraction``
  on float välillä [0, 1]).
* Literal-tyypit varmistavat, että enum-arvot ovat sallittuja
  jo ekstraktiovaiheessa, ei vasta Panderan validointihetkellä.
* JSON-deserialisointi (``RawPair(**dict)``) tuottaa selkeän virheen
  väärälle muodolle, mikä helpottaa LLM-output-debuggausta.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Literal-tyypit pidetään modulivakioina, jotta niitä voi myös
# importoida testeissä ja ne pysyvät yhdessä paikassa, joka peilaa
# YAML-skeeman enumia.
DrugRole = Literal["api", "coformer", "excipient"]
RatioReportedAs = Literal[
    "mole_fraction", "weight_fraction", "molar_ratio", "weight_ratio"
]
ProcessMethod = Literal[
    "melt_quench",
    "cryo_milling",
    "spray_drying",
    "solvent_evaporation",
    "ball_milling",
    "liquid_assisted_grinding",
    "co_precipitation",
    "freeze_drying",
    "hot_melt_extrusion",
    "quench_cooling",
    "other",
]
ExperimentalProtocol = Literal[
    "ich_q1a_accelerated",
    "ich_q1a_long_term",
    "dry_short_term",
    "tg_plus_15K",
    "dsc_in_situ",
    "non_standard",
]


class RawPair(BaseModel):
    """Yksi LLM:n ekstraktoima ko-amorfinen pari.

    Tämä on **ainoa** tietorakenne, jonka LLM kirjoittaa. Kaikki muut
    master-CSV:n sarakkeet (kanoniset SMILES, gfa_class, descriptorit)
    täytetään ohjelmallisesti seuraavissa vaiheissa.

    Notes
    -----
    Ei ``pair_id``- tai ``entry_id``-kenttää: ``pair_id`` generoidaan
    juoksevasti ``enrich.assign_pair_id``-funktiolla ja ``entry_id`` on
    täällä ``source_table_or_figure`` — vapaasti formatoitu viittaus,
    joka auttaa palaamaan alkuperäiseen julkaisuun.
    """

    # Pydantic-konfiguraatio: kielletään tuntemattomat kentät, jotta
    # LLM:n ylimääräiset avaimet havaitaan välittömästi sen sijaan, että
    # ne hiljaa katoaisivat.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # --- Tunnisteet (lähteen omat termit) ------------------------------------
    # _raw-pääte: nämä ovat artikkelin omat nimet, ei kanoniset. Kanoninen
    # nimi tulee PubChem-haun jälkeen ja tallennetaan eri kenttään master-CSV:ssä.
    drug_A_name_raw: str = Field(..., min_length=1)
    drug_B_name_raw: str = Field(..., min_length=1)
    drug_A_role: DrugRole
    drug_B_role: DrugRole

    # --- Suhde -----------------------------------------------------------------
    # MIKSI sekä mooli- että painoosuudet sallittuja: lähteet raportoivat
    # suhteen eri muodoissa, ja molempien tallentaminen yhden rivin sisällä
    # estää tiedon häviämisen muunnoksessa. Vähintään yksi pari pitää olla.
    mole_fraction_A: Optional[float] = Field(None, ge=0, le=1)
    mole_fraction_B: Optional[float] = Field(None, ge=0, le=1)
    weight_fraction_A: Optional[float] = Field(None, ge=0, le=1)
    weight_fraction_B: Optional[float] = Field(None, ge=0, le=1)
    ratio_reported_as: Optional[RatioReportedAs] = None
    ratio_source_quote: str = Field(
        ...,
        min_length=1,
        description=(
            "Eksakti lainaus lähteestä, joka kertoo suhteen "
            "(esim. '1:1 mol' tai '70:30 w/w'). Audit-kenttä."
        ),
    )

    # --- Prosessi -------------------------------------------------------------
    process_method: Optional[ProcessMethod] = None
    process_details: Optional[str] = None

    # --- Stabiilisuus ---------------------------------------------------------
    induction_time_days: Optional[float] = Field(None, ge=0)
    induction_time_censored: Optional[bool] = None
    storage_T_C: Optional[float] = None
    storage_RH_percent: Optional[float] = Field(None, ge=0, le=100)
    crystallizes_on_dsc_cooling: Optional[bool] = None
    experimental_protocol: Optional[ExperimentalProtocol] = None
    protocol_max_duration_days: Optional[float] = Field(None, ge=0)

    # --- Termodynamiikka ------------------------------------------------------
    Tg_K: Optional[float] = None
    Tg_uncertainty_K: Optional[float] = Field(None, ge=0)
    Tg_heating_rate_K_min: Optional[float] = None
    Tm_A_K: Optional[float] = None
    Tm_B_K: Optional[float] = None
    pxrd_amorphous: Optional[bool] = None
    detection_methods: Optional[str] = None

    # --- Lähteen audit-tiedot -------------------------------------------------
    # source_table_or_figure on sama kenttä kuin master-CSV:ssä (esim. "Table 2"
    # tai "Figure 4a"). source_quote on lyhyt lainaus, joka *tukee* tätä riviä —
    # auditointia varten, jos joku rivi vaikuttaa väärältä myöhemmin.
    source_table_or_figure: str = Field(..., min_length=1)
    source_quote: str = Field(..., min_length=1)
    notes: Optional[str] = None

    # --- Ihmisen tarkastus ----------------------------------------------------
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_at_least_one_ratio(self) -> "RawPair":
        """Vähintään yksi (mole_fraction- tai weight_fraction) -pari on annettava.

        MIKSI: kokonaan tuntematon koostumus tekisi rivistä käyttökelvottoman
        ML-mallille. Jos lähde ei raportoi suhdetta lainkaan, ekstraktoijan
        (LLM) pitäisi merkitä ``needs_review=True`` — silloin voimme sallia
        tyhjät kentät, koska niitä on tarkoitus käsitellä manuaalisesti.
        """
        has_mole = (
            self.mole_fraction_A is not None and self.mole_fraction_B is not None
        )
        has_weight = (
            self.weight_fraction_A is not None and self.weight_fraction_B is not None
        )
        if not (has_mole or has_weight or self.needs_review):
            raise ValueError(
                "RawPair: vähintään yksi mole_fraction- tai weight_fraction -pari "
                "on annettava, tai needs_review=True jos suhde on todella tuntematon."
            )
        return self
