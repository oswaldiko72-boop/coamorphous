"""
Päivittää extract_template.ipynb:n promptin kahdella säännöllä:
  - Sääntö 7: parannettu needs_review-flagging (SD>mean, Tg-yhteneväisyys, jne.)
  - Sääntö 9: drug_role-tunnistus drug-drug-systeemeille (api/api)

Käyttö: python tools/update_prompt.py

Idempotentti: tarkistaa onko muutos jo tehty ja exitoituu turvallisesti
jos säännöt 7 ja 9 on jo päivitetty.
"""
import sys
from pathlib import Path

try:
    import nbformat
except ImportError:
    print("VIRHE: nbformat ei ole asennettu. Aja: pip install nbformat")
    sys.exit(1)

NB_PATH = Path("notebooks/01_corpus_extraction/extract_template.ipynb")

# --- Vanhat säännöt 7 + 8 (näiden tulee löytyä tarkalleen tästä muodosta) ---
OLD_RULES_7_8 = """7. Aseta needs_review=True ja review_reasons-listalle, jos:
   - Lähde on epäjohdonmukainen (esim. taulukon otsikko vs. tekstin numerot)
   - Sensurointi on epäselvä
   - Protokolla ei sovi mihinkään selvästi
   - Stabiiliuskoetta ei raportoitu lainkaan (vain DSC tai PXRD)
8. source_quote on 1-2 lauseen lainaus lähteestä, joka tukee tämän parin tietoja."""

# --- Uudet säännöt 7 + 8 + 9 (sääntö 8 säilyy ennallaan) ---
NEW_RULES_7_8_9 = """7. Aseta needs_review=True ja review_reasons-listalle, jos JOKIN seuraavista
   ehdoista täyttyy. Anna SPESIFINEN selitys, ei geneeristä tekstiä:

   A) TILASTOLLINEN EPÄVARMUUS YLITTÄÄ MITTAUKSEN:
      - induction_time SD >= mean (esim. "tcryst = 0.3 +- 0.6 vrk" -> SD > mean)
      - Tg_uncertainty_K >= 5 K (tyypillinen on 0.1-2 K)

   B) SISÄINEN EPÄJOHDONMUKAISUUS:
      - Pari-Tg <= 2 K päässä joko puhtaan A:n tai B:n Tg:stä
        (viittaa faasierotukseen tai kiteytymiseen, ei co-amorfiseen tilaan)
      - Raportoitu storage_T_C poikkeaa >5 K paperin omasta laskukaavasta
        (esim. T_s = T_g + 0.3*(T_m50:50 - T_g))
      - mole_fraction_A + mole_fraction_B != 1.00 (toleranssi 0.01)

   C) LÄHTEEN EPÄJOHDONMUKAISUUS:
      - Taulukon otsikko vs. tekstin numerot ristiriidassa
      - ratio_source_quote ei suoraan löydy paperin tekstistä
      - Stabiiliuskoetta ei raportoitu lainkaan (vain DSC tai PXRD)

   D) PROTOKOLLAN RAJATAPAUKSET:
      - Sensurointi on epäselvä (kokeen kesto vs. havaintoaika ei selvä)
      - storage_T_C on 2 K päässä Tg:stä (rajatapaus above_tg_kinetic)
      - Protokolla ei sovi mihinkään enum-arvoon selvästi

   review_reasons[] tulee sisältää KONKREETTINEN havainto, ei geneerinen teksti.
   Hyvä esimerkki: "PLM induction_time SD (0.6 vrk) ylittää keskiarvon (0.3 vrk),
                   indikoi suurta epävarmuutta kiteytymisen havaitsemisessa"
   Huono esimerkki: "data quality issue"
8. source_quote on 1-2 lauseen lainaus lähteestä, joka tukee tämän parin tietoja.
9. Aseta drug_A_role ja drug_B_role:

   DRUG-DRUG-SYSTEEMIT (molemmat farmaseuttisia vaikuttavia aineita):
      - drug_A_role = "api"
      - drug_B_role = "api"

   Käytä "coformer"-arvoa VAIN kun komponentti B on ei-vaikuttava
   stabilointiaine, esim:
      - aminohapot (arginiini, tryptofaani, asparagiinihappo, jne.)
      - sokerit/sokerialkoholit (trehaloosi, mannitoli, jne.)
      - polymeerit (PVP, HPMC, Soluplus, jne.)
      - urea, sakkariini, sitruunahappo tai muut pien-molekyyliexcipientit

   Esimerkkejä DRUG-DRUG-systeemeistä (api/api):
      - Pajula 2014: terfenadiini + parasetamoli, indometaiini + ASA, jne.
      - Lobmann 2011: naprokseni + indometaiini
      - Lobmann 2012: simvastatiini + glipitsidi
      - Alleso 2009: naprokseni + simetidiini
      - Knapik 2015: etsetimibi + indapamidi
      - Knapik 2019: etsetimibi + simvastatiini

   Jos paperi käyttää sanoja "drug-drug mixture", "drug-drug system", tai
   "binary drug pair" -> api/api riippumatta kumpi on ensisijainen
   terapeuttinen kohde."""


def main() -> int:
    if not NB_PATH.exists():
        print(f"VIRHE: Tiedostoa ei loydy: {NB_PATH}")
        return 1

    nb = nbformat.read(str(NB_PATH), as_version=4)

    target = None
    for cell in nb.cells:
        if (
            cell.cell_type == "code"
            and 'prompt_body = f"""' in cell.source
            and "KRIITTISET" in cell.source
        ):
            target = cell
            break

    if target is None:
        print("VIRHE: Code-solua, jossa on prompt_body f-string, ei loydy.")
        return 1

    # Idempotency-tarkistus
    if (
        '9. Aseta drug_A_role' in target.source
        and 'drug_A_role = "api"' in target.source
    ):
        print("INFO: Saannot 7 ja 9 on jo paivitetty. Ei muutoksia.")
        return 0

    # Tarkista etta vanha sisalto on loydettavissa
    if OLD_RULES_7_8 not in target.source:
        print("VIRHE: Vanhaa saanto 7+8 -tekstia ei loytynyt.")
        print("Promptin rakenne voi olla muuttunut viime ajosta.")
        return 1

    orig_len = len(target.source)
    new_source = target.source.replace(OLD_RULES_7_8, NEW_RULES_7_8_9)

    if new_source == target.source:
        print("VIRHE: Korvaus epaonnistui (sisalto ei muuttunut).")
        return 1

    target.source = new_source
    nbformat.write(nb, str(NB_PATH))

    print(f"OK: Paivitetty {NB_PATH}")
    print(f"  - Saanto 7 (needs_review) parannettu (4 ryhmaa: A, B, C, D)")
    print(f"  - Saanto 9 (drug_role: api/api) lisatty")
    print(f"  - Source kasvoi {orig_len} -> {len(new_source)} merkkia")
    print()
    print("Seuraava vaihe: aja Pajula 2014 -ekstraktio uudelleen Opus 4.7:lla")
    print("ja tarkista etta 'above_tg_kinetic' loytyy seka 'drug_B_role: api'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())