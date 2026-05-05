"""``RawPair``-objektien rikastus master-CSV:n riveiksi.

MIKSI tämä moduuli on olemassa
------------------------------
``RawPair`` (Pydantic-skeema) sisältää vain raakatiedot lähteestä. Master-CSV
vaatii lisäksi:

* kanoniset SMILES ja InChIKey (lasketaan PubChem-hausta + RDKit:llä),
* ``gfa_class_dsc``, ``gfa_dsc_evidence`` ja ``gfa_label_confidence``
  (``classify_gfa_dsc``),
* ``stability_week_bin``, ``stability_protocol_match`` ja
  ``stability_label_confidence`` (vastaavat ``classify_stability_*``-funktiot),
* lähdemetadata (DOI, kirjailija, vuosi, ekstraktiopäivä),
* pari-ID juoksevasti.

Tämä moduuli kokoaa kaikki vaiheet yhteen funktioon
(``raw_pair_to_master_row``), mutta yksittäiset askeleet ovat eksportoitu
erikseen, jotta niitä voi testata ja debugata erikseen.

Suunnitteluperiaate
-------------------
Ei sivuvaikutuksia rikastusvaiheen *sisällä*: kaikki funktiot palauttavat
uuden dictin tai modifioivat *RawPair*-objektin ``review_reasons``-listaa
nimenomaisesti. Tämä helpottaa kunkin riviä-kohti -muutoksen jäljitystä.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from coamorphous.corpus.canonicalize import inchikey_from_smiles, safe_canonical
from coamorphous.corpus.classify import (
    classify_gfa_dsc,
    classify_stability_label_confidence,
    classify_stability_protocol,
    classify_stability_week_bin,
)
from coamorphous.extraction.extraction_schema import RawPair
from coamorphous.extraction.pubchem_lookup import (
    PubChemError,
    search_by_name,
    with_retry,
)

logger = logging.getLogger(__name__)


def gfa_label_from_class(gfa_class: Optional[int]) -> Optional[str]:
    """Muunna numeerinen GFA-luokka 1/2/3 ihmisluettavaksi merkinnäksi.

    "Class I", "Class II", "Class III". Käytetään master-CSV:n
    ``gfa_class_label``-sarakkeeseen, joka on redundantti mutta hyödyllinen
    visualisoinneissa ja raportoinnissa.
    """
    if gfa_class is None:
        return None
    return f"Class {'I' * gfa_class}"


def pubchem_lookup_pair(raw: RawPair) -> dict:
    """Hae molempien lääkkeiden PubChem-tiedot ja palauta kootut osumat.

    Ei modifioi ``raw``-objektia paitsi ``review_reasons``-listaa silloin,
    kun haku epäonnistuu — kutsuva koodi merkitsee rivin ihmisen
    tarkastusta varten.

    Returns
    -------
    dict
        ``{"A": {smiles, inchikey, mw, cid} | None, "B": ...}``.
        Kummatkin avaimet voivat olla ``None``, jos PubChem ei tunnistanut
        kyseistä lääkettä — tällöin riville on jo lisätty review_reason.
    """
    result: dict[str, Optional[dict]] = {"A": None, "B": None}
    for key, name in (("A", raw.drug_A_name_raw), ("B", raw.drug_B_name_raw)):
        try:
            hit = with_retry(search_by_name, name)
        except PubChemError as exc:
            logger.warning("PubChem-haku epäonnistui %r:lle: %s", name, exc)
            raw.review_reasons.append(
                f"PubChem-haku epäonnistui drug_{key}={name!r}: {exc}"
            )
            raw.needs_review = True
            continue

        if hit is None:
            logger.info("PubChem ei tunnistanut nimeä %r", name)
            raw.review_reasons.append(
                f"PubChem ei tunnistanut nimeä drug_{key}={name!r}"
            )
            raw.needs_review = True
            continue

        result[key] = hit
    return result


def canonicalize_pair(raw_dict: dict) -> dict:
    """Sovella RDKit-kanonisointia ``raw_dict``:n SMILES-kenttiin.

    Päivittää ``drug_A_smiles_canonical`` ja ``drug_B_smiles_canonical`` sekä
    ``drug_A_inchikey``/``drug_B_inchikey`` jos kanonisointi onnistui.
    Ei muuta muita kenttiä.

    Parameters
    ----------
    raw_dict : dict
        Rivi master-CSV:n muodossa, sisältäen
        ``drug_A_smiles_original``/``drug_B_smiles_original``-kentät
        (yleensä PubChem:istä saadut). ``safe_canonical`` palauttaa
        ``None`` virheellisille SMILES:eille, joten tämä funktio ei heitä.
    """
    out = dict(raw_dict)
    for letter in ("A", "B"):
        original = out.get(f"drug_{letter}_smiles_original")
        canonical = safe_canonical(original)
        out[f"drug_{letter}_smiles_canonical"] = canonical

        # InChIKey vain, jos kanonisointi onnistui — muutoin on parempi
        # jättää InChIKey tyhjäksi kuin laskea se virheellisestä syötteestä.
        if canonical is not None:
            try:
                out[f"drug_{letter}_inchikey"] = inchikey_from_smiles(canonical)
            except Exception as exc:  # noqa: BLE001 — laaja koppaus tarkoituksellinen
                logger.warning(
                    "InChIKey-laskenta epäonnistui drug_%s:lle: %s", letter, exc
                )
                out[f"drug_{letter}_inchikey"] = None
        else:
            out[f"drug_{letter}_inchikey"] = None
    return out


def compute_classification(raw: RawPair) -> dict:
    """Aja kaikki neljä luokitusta ``RawPair``-objektin tiedoilla.

    Yhdistää GFA-luokituksen (DSC-pohjainen) ja stabiilisuusluokituksen
    (säilytysprotokollan match + viikkobini + label_confidence) yhteen
    sanakirjaan. Erotettu omaksi funktioksi, jotta:

    * testit voivat varmistaa, että luokitukset kutsutaan oikeilla parametreilla,
    * ekstraktiopipelinen "luokitus" -vaihe on yksi nimetty toiminto.

    Älä koskaan kovakoodaa luokituksia ekstraktiossa — kutsu aina tätä
    funktiota, jolloin uudelleenajot tuottavat yhtenäiset tulokset
    luokituslogiikan päivitysten kanssa.

    Returns
    -------
    dict
        Avaimet: ``gfa_class_dsc``, ``gfa_label_confidence``,
        ``gfa_dsc_evidence``, ``stability_week_bin``,
        ``stability_protocol_match``, ``stability_label_confidence``.
    """
    # classify_gfa_dsc johtaa evidence-arvon itse syöttötiedoista —
    # ekstraktiokoodi välittää vain raakatiedot eikä yritä päätellä
    # evidence-merkkiä etukäteen.
    gfa_class, gfa_conf, gfa_evidence = classify_gfa_dsc(
        crystallizes_on_cooling=raw.crystallizes_on_dsc_cooling,
        crystallizes_on_reheating=raw.crystallizes_on_dsc_reheating,
        paper_states_class=raw.paper_states_gfa_class,
    )

    week_bin = classify_stability_week_bin(
        induction_time_days=raw.induction_time_days,
        induction_time_censored=raw.induction_time_censored,
    )
    protocol_match = classify_stability_protocol(
        storage_T_C=raw.storage_T_C,
        storage_RH_percent=raw.storage_RH_percent,
        experimental_protocol=raw.experimental_protocol,
    )
    stability_conf = classify_stability_label_confidence(
        protocol_match=protocol_match,
        experimental_protocol=raw.experimental_protocol,
    )

    return {
        "gfa_class_dsc": gfa_class,
        "gfa_label_confidence": gfa_conf,
        "gfa_dsc_evidence": gfa_evidence,
        "stability_week_bin": week_bin,
        "stability_protocol_match": protocol_match,
        "stability_label_confidence": stability_conf,
    }


def raw_pair_to_master_row(
    raw: RawPair,
    source_metadata: dict,
    extracted_by: str = "claude_code",
) -> dict:
    """Yhdistä raakatiedot, PubChem-osumat ja luokitus yhdeksi master-CSV-riviksi.

    Ei aseta ``pair_id``-kenttää — se annetaan kollektiivisesti
    ``assign_pair_id``-funktiolla, jolloin numerointi on juokseva
    yhdellä lähteellä.

    Parameters
    ----------
    raw : RawPair
        LLM-ekstraktion tuottama validoitu raaka-objekti.
    source_metadata : dict
        Sisältää vähintään ``source_doi``, ``source_first_author``,
        ``source_year``. Voi sisältää myös vapaaehtoisia avaimia.
    extracted_by : str, default "claude_code"
        Audit-merkki: kuka/mikä ekstraktoi rivin. Helpottaa myöhempiä
        revisiokierroksia.

    Returns
    -------
    dict
        Master-CSV:n rivi (avaimet vastaavat YAML-skeemaa). Pari-tason
        deskriptorit (``delta_*``, ``tanimoto_*``) jätetään ``None``:ksi
        — ne lasketaan H1-kohdan 2.4 erillisessä vaiheessa.
    """
    # 1) PubChem-haku molemmille lääkkeille.
    pubchem_hits = pubchem_lookup_pair(raw)

    smiles_A = pubchem_hits["A"]["smiles"] if pubchem_hits["A"] else None
    smiles_B = pubchem_hits["B"]["smiles"] if pubchem_hits["B"] else None

    # 2) Kanoniset SMILES + InChIKey RDKit:llä.
    enriched = canonicalize_pair(
        {
            "drug_A_smiles_original": smiles_A,
            "drug_B_smiles_original": smiles_B,
        }
    )

    # 3) Luokitus — kahdeksi erilliseksi kohdemuuttujaksi (GFA, stability).
    classifications = compute_classification(raw)

    # 4) Notes-kenttä: yhdistetään LLM:n notes ja review_reasons,
    # jotta auditoinnissa kaikki epävarmuudet ovat yhdellä rivillä näkyvissä.
    notes_parts: list[str] = []
    if raw.notes:
        notes_parts.append(raw.notes)
    if raw.review_reasons:
        notes_parts.append("[review] " + "; ".join(raw.review_reasons))
    notes_combined = " | ".join(notes_parts) if notes_parts else None

    row: dict = {
        # A. IDENTITEETTI — pair_id puuttuu, asetetaan myöhemmin.
        "pair_id": None,
        "entry_id": raw.source_table_or_figure,
        "drug_A_name": raw.drug_A_name_raw,
        "drug_B_name": raw.drug_B_name_raw,
        "drug_A_role": raw.drug_A_role,
        "drug_B_role": raw.drug_B_role,
        "drug_A_smiles_original": smiles_A,
        "drug_B_smiles_original": smiles_B,
        "drug_A_smiles_canonical": enriched.get("drug_A_smiles_canonical"),
        "drug_B_smiles_canonical": enriched.get("drug_B_smiles_canonical"),
        "drug_A_inchikey": enriched.get("drug_A_inchikey"),
        "drug_B_inchikey": enriched.get("drug_B_inchikey"),
        "drug_A_cas": pubchem_hits["A"]["cas"] if pubchem_hits["A"] else None,
        "drug_B_cas": pubchem_hits["B"]["cas"] if pubchem_hits["B"] else None,
        # B. KOOSTUMUS JA PROSESSI
        "mole_fraction_A": raw.mole_fraction_A,
        "mole_fraction_B": raw.mole_fraction_B,
        "weight_fraction_A": raw.weight_fraction_A,
        "weight_fraction_B": raw.weight_fraction_B,
        "ratio_reported_as": raw.ratio_reported_as,
        "process_method": raw.process_method,
        "process_details": raw.process_details,
        # C. KOHDEMUUTTUJAT
        "gfa_class_dsc": classifications["gfa_class_dsc"],
        "gfa_class_label": gfa_label_from_class(classifications["gfa_class_dsc"]),
        "gfa_dsc_evidence": classifications["gfa_dsc_evidence"],
        "gfa_label_confidence": classifications["gfa_label_confidence"],
        "stability_week_bin": classifications["stability_week_bin"],
        "stability_protocol_match": classifications["stability_protocol_match"],
        "stability_label_confidence": classifications["stability_label_confidence"],
        "induction_time_days": raw.induction_time_days,
        "induction_time_censored": raw.induction_time_censored,
        "storage_T_C": raw.storage_T_C,
        "storage_RH_percent": raw.storage_RH_percent,
        # D. TERMODYNAMIIKKA
        "Tg_K": raw.Tg_K,
        "Tg_uncertainty_K": raw.Tg_uncertainty_K,
        "Tg_heating_rate_K_min": raw.Tg_heating_rate_K_min,
        "Tm_A_K": raw.Tm_A_K,
        "Tm_B_K": raw.Tm_B_K,
        "pxrd_amorphous": raw.pxrd_amorphous,
        "detection_methods": raw.detection_methods,
        # E. LÄHDEMETADATA
        "source_doi": source_metadata.get("source_doi"),
        "source_first_author": source_metadata.get("source_first_author"),
        "source_year": source_metadata.get("source_year"),
        "source_table_or_figure": raw.source_table_or_figure,
        "extraction_date": source_metadata.get(
            "extraction_date", date.today().isoformat()
        ),
        "extracted_by": extracted_by,
        "notes": notes_combined,
        "experimental_protocol": raw.experimental_protocol,
        "protocol_max_duration_days": raw.protocol_max_duration_days,
        "storage_T_K": (
            raw.storage_T_C + 273.15 if raw.storage_T_C is not None else None
        ),
        # F. PARI-TASON JOHDETUT PIIRTEET — lasketaan H1 2.4:ssa erikseen.
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
    return row


def assign_pair_id(
    rows: list[dict], first_author: str, year: int
) -> list[dict]:
    """Generoi juokseva ``pair_id`` muodossa ``{first_author}{year}_{nn}``.

    Esim. ``lobmann2011_03``. Numerointi alkaa 1:stä, leveys 2 (zero-pad).

    Parameters
    ----------
    rows : list of dict
        Master-CSV-rivit ``raw_pair_to_master_row``:n palautusarvosta.
        Modifioidaan paikallaan ``pair_id``-kentän osalta.
    first_author : str
        Lähteen ensimmäisen kirjailijan sukunimi pienillä kirjaimilla,
        ilman välilyöntejä (esim. ``"lobmann"``, ``"fink"``).
    year : int
        Julkaisuvuosi.

    Returns
    -------
    list of dict
        Sama lista (sama referenssi), helpottaa chainausta.
    """
    fa = first_author.strip().lower().replace(" ", "")
    for n, row in enumerate(rows, start=1):
        row["pair_id"] = f"{fa}{year}_{n:02d}"
    return rows
