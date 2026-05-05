# coamorphous

Tutkimusprojekti, joka ennustaa **co-amorfisten lääkesysteemien stabiilisuutta**
yhdistämällä molekyylidynamiikkaa (MD) ja koneoppimista (ML).

## Tausta

Co-amorfinen järjestelmä = kaksi (tai useampi) lääkemolekyyliä, jotka
muodostavat yhteisen amorfisen faasin. Tällaiset järjestelmät voivat parantaa
heikosti liukenevien lääkeaineiden biosaatavuutta, mutta ne ovat
termodynaamisesti epästabiileja ja taipuvaisia rekristallisaatioon. Tavoitteena
on rakentaa malli, joka ennustaa parin **Baird-Taylor lasinmuodostumisluokan**
(Class I/II/III) kemiallisista deskriptoreista ja MD-piirteistä.

## Projektin rakenne (työpaketit)

| | |
|---|---|
| H1 | Datakorpus + baseline-luokitin (**käynnissä**) |
| H2 | MD-simulaatiopipeline |
| H3 | MD-johdetut piirteet |
| H4 | Hybridimallit (kemia + MD) |
| H5 | Validointi ulkoisilla pareilla |
| H6 | Analyysi (SHAP, kuvat) |

## Asennus

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -e .[dev]
```

Tarkista, että kaikki toimii:

```bash
pytest
```

## Hakemistorakenne

```
data/
  raw/         alkuperäiset PDF:t ja lähde-CSV:t (gitignore)
  interim/     ekstraktoidut välivaiheen CSV:t per lähde
  processed/   master-korpus (CSV committoidaan, Parquet ei)
  external/    PubChem/DrugBank-haut välimuistissa
notebooks/
  01_corpus_extraction/   per-lähde-ekstraktion notebookit
  02_descriptor_calc/     RDKit/ECFP-deskriptorilaskenta
  03_baseline_models/     RandomForest/XGBoost
  04_md_validation/       MD-johdetut piirteet (H2/H3)
  05_analysis/            SHAP, kuvat (H6)
src/coamorphous/
  corpus/        skeema, kanonisointi, luokitus, yhdistäminen
  descriptors/   molekyyli- ja pari-tason deskriptorit
  utils/         I/O, lokitus
tests/           pytest-yksikkötestit
configs/
  corpus_schema.yaml   master-CSV:n virallinen sarakeskeema
docs/            laajempi dokumentaatio
```

## Uuden lähteen lisääminen

1. **Kopioi pohja:**

   ```bash
   cp notebooks/01_corpus_extraction/00_template.ipynb \
      notebooks/01_corpus_extraction/{author}_{year}.ipynb
   ```

2. **Täytä bibliografia** (DOI, kirjoittaja, vuosi).

3. **Syötä rivit** `entries`-listaan dictinä per pari. Jätä puuttuva arvo
   `None`:ksi — älä keksi.

4. **Aja kaikki solut.** Notebook ajaa skeemavalidoinnin ja SMILES-kanonisoinnin
   automaattisesti. Jos validointi epäonnistuu, korjaa rivit ja aja uudelleen.

5. **Tarkista** `data/interim/{author}_{year}.csv`.

6. **Yhdistä master-korpukseen** (tehdään H1:n päätyttyä erillisenä ajona):

   ```python
   from coamorphous.corpus.merge import merge_interim
   from coamorphous.utils.io import write_corpus
   from pathlib import Path

   df = merge_interim(
       interim_paths=list(Path("data/interim").glob("*.csv")),
       schema_yaml=Path("configs/corpus_schema.yaml"),
   )
   write_corpus(df, Path("data/processed/coamorphous_corpus_v1.csv"))
   ```

## Korpuksen versionnumerointi

Master-CSV:n nimi sisältää version: `coamorphous_corpus_v{N}.csv`.

* **v1** — alkuperäinen H1-vaiheen ekstraktio (~250 paria, kohde).
* **v2** — kun lisätään uusia lähteitä tai korjataan rakenteellisia
  virheitä (esim. uudet sarakkeet, eri luokitusperusteet).
* **vN.M** — pieni patch-päivitys (esim. yksittäisten rivien korjaus).

Kasvata pääversiota (N) vain, jos skeema (`configs/corpus_schema.yaml`)
muuttuu epäyhteensopivasti.

## Tieteellinen referenssi

Baird, J. A. & Taylor, L. S. *Adv. Drug Deliv. Rev.* **64** (2012) 396–421.

## Lisenssi

MIT.
# coamorphous
