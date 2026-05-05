# Notebooks

Tutkimusvaiheittain organisoidut Jupyter-notebookit. Ekstraktiopipelinen
työpöytä on `01_corpus_extraction/`.

## Hakemistot

- `01_corpus_extraction/` — yksi notebook per lähde (Löbmann 2011, Fink 2023, ...).
  Kukin tuottaa `data/interim/{author}_{year}.csv`-tiedoston.
- `02_descriptor_calc/` — H1 2.4 pari-tason RDKit/ECFP-deskriptorit.
- `03_baseline_models/` — Random Forest / XGBoost -baseline.
- `04_md_validation/` — H2/H3 MD-pohjaiset piirteet.
- `05_analysis/` — H6 SHAP, kuvat, käsikirjoitusvalmiit visualisoinnit.

## Lähde-ekstraktio: extract_template.ipynb

Pipeline (LLM raakaekstraktio + ohjelmallinen rikastus + luokitus + validointi)
on parametroitu `extract_template.ipynb`-notebookiin. Käyttöohje:

1. Kopioi pohja:
   ```
   cp notebooks/01_corpus_extraction/extract_template.ipynb \
      notebooks/01_corpus_extraction/{author}_{year}.ipynb
   ```
2. Editoi konfiguraatiosolu (`PDF_PATH`, `SOURCE_DOI`, `SOURCE_FIRST_AUTHOR`,
   `SOURCE_YEAR`).
3. Aja Solu 2 (imports + polut) → tarkista että polut ovat oikein.
4. **Vaihe 1** — kopioi promptin Vaiheen 1 markdown-solusta Claude Code
   -sivupaneeliin. Anna sen ekstraktoida ja tallentaa
   `data/interim/{author}_{year}_raw.json`.
5. Aja loput solut järjestyksessä. Pipeline:
   - validoi raakatiedot Pydanticilla (`RawPair`),
   - rikastaa SMILES:t PubChem:istä,
   - kanonisoi RDKit:llä,
   - laskee `gfa_class` ja `label_confidence` `classify_baird_taylor`-funktiolla,
   - validoi mole/weight-summat ja säilytysolojen yhteensopivuuden protokollaan,
   - tallentaa `data/interim/{author}_{year}.csv`.

## Tärkeä periaate: LLM ei päätä luokituksia

LLM (sinä Claude Code) ekstraktoi vain raakatiedot. **Älä koskaan editoi**
`gfa_class`- tai `label_confidence`-kenttiä käsin. Jos rivi näyttää väärin
luokitellulta, korjaa sen *syöttötiedot* (esim. `induction_time_censored`,
`storage_T_C`, `experimental_protocol`) `*_raw.json`-tiedostossa ja aja
notebook uudelleen — luokitus päivittyy automaattisesti.

Sama koskee SMILES-merkkijonoja: ne tulevat aina PubChem:istä RDKit-kanonisoinnin
kautta, ei LLM:n muistista.

## Nykyiset lähteet

- `lobmann_2011.ipynb` — Löbmann et al. 2011 (DOI: 10.1021/mp2002973),
  6 paria. Käytti vanhaa ekstraktiotapaa; siirrettiin uuteen skeemaan
  `scripts/fix_lobmann_2011.py`-skriptillä.
