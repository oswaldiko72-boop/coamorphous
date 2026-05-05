"""Lähdeartikkelien ekstraktion työkalut.

Tämä alipakettiin (``coamorphous.extraction``) kuuluvat moduulit muodostavat
sen pipelinen, jolla yksittäisen lähdejulkaisun PDF:stä tuotetaan master-CSV:n
rivit. Pipeline pitää tarkasti erillään:

* **LLM-vastuu** (``RawPair`` Pydantic-skeema): vain raakatiedot lähteestä,
  ei johdettuja arvoja kuten ``gfa_class`` tai kanonisia SMILES:eja.
* **Koodin vastuu** (``enrich``, ``validate``): kaikki ohjelmalliset päätökset
  — PubChem-haut, kanonisointi, luokitus ``classify_baird_taylor``-funktiolla,
  validointitarkistukset.

Tämä erottelu varmistaa, että luokitus ja kanonisointi pysyvät yhtenäisinä
lähteestä riippumatta — emme luota LLM:n laskentaan asioista, jotka koodi
osaa tehdä deterministisesti.
"""
