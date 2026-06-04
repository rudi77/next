---
feature: dataset-registry
status: shipped
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr:
---

# Dataset-Registry — Upload, Validierung und `ds:`-Referenzen

Statt rohe Dateipfade durch die API zu reichen, lädt der Nutzer ein Dataset
einmal hoch; trainpipe prüft das Format, speichert es deduppliziert auf
Platte und vergibt eine Id. In einer `ExperimentSpec` referenziert man es
dann als `ds:<id>` (optional mit Sub-Sample-Suffix `#500`). Diese Referenzen
werden am Submit zu echten Pfaden aufgelöst, sodass Scheduler und Trainer nie
die Registry brauchen.

## Capabilities (was der Nutzer tun kann)

- Ein Dataset hochladen (jsonl / json / csv / tsv / parquet), benannt + beschrieben
- Datasets auflisten und Details (Format, Zeilenzahl, Größe, sha256) abrufen
- Die ersten N Zeilen eines Text-Datasets als Vorschau abrufen
- Ein Dataset per `ds:<id>` (optional `#N` für Teilmenge) in Specs referenzieren
- Ein Dataset löschen, geschützt gegen das Löschen aktiv genutzter Dateien

## Invariants (was immer gelten muss)

- Upload dedupliziert per sha256: identischer Inhalt liefert den bestehenden
  Eintrag mit **200** (statt 201) zurück, ohne die Datei zu duplizieren
- Das Format wird beim Upload validiert (Stichprobe der ersten ~100 Records);
  offensichtlich kaputte/leere Dateien werden mit 422 abgelehnt
- Uploads über das Größenlimit werden mit 413 abgewiesen; Teil-/Fehl-Dateien
  werden von der Platte aufgeräumt (kein geleakter Speicher)
- `DELETE` ist mit **409** geschützt, solange queued/running Experimente die
  Datei referenzieren; `?force=true` übergeht das
- Ein wohlgeformtes `ds:<hex>(#suffix)?` wird zum registrierten Pfad aufgelöst;
  ein `ds:`-String, der die Grammatik verletzt, ist 422 (`malformed_dataset_ref`);
  eine unbekannte Id ist 422 (`unknown_dataset_ref`)
- Auflösung passiert am Submit-Zeitpunkt; der persistierte Spec trägt nur Pfade
- Nicht-`ds:`-Strings (HF-Ids, lokale Pfade) werden unverändert durchgereicht

## API surface (der Vertrag für Clients)

- POST /datasets → 201 (neu) · 200 (sha256-Dedup-Treffer) · 422 (`invalid_dataset_format`) · 413 (zu groß)
- GET /datasets → 200 (alle, neueste zuerst)
- GET /datasets/{id} → 200 · 404
- GET /datasets/{id}/preview?n= → 200 (Plaintext) · 404 · 410 (Datei fehlt auf Platte)
- DELETE /datasets/{id} → 200 `{deleted}` · 404 · 409 (`dataset_in_use`, ohne `?force=true`)

Weitere Routen hängen am selben `/datasets`-Router, gehören aber zu Folge-Features
und sind dort spezifiziert (sie durchlaufen alle dieselbe Registrierung — sha256-Dedup,
Format-Validierung):

- `POST /datasets/bundle`, `GET /datasets/{id}/media` → [multimodal-training](multimodal-training.md)
- `POST /datasets/{id}/split`, `POST /mixes` → [dataset-versioning](dataset-versioning.md)
- `POST /datasets/{id}/redact`, `GET /datasets/{id}/models` → [pii-redaction](pii-redaction.md)
- `POST /datasets/from-labelstudio` → [labelstudio-import](labelstudio-import.md)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `TRAINPIPE_MAX_DATASET_UPLOAD_BYTES: int` (default 5 GiB) — Upload-Limit
- `TRAINPIPE_DATA_DIR: path` — unter `datasets/<id>/` liegen die Dateien

## Extension points (für Plugins / externe Nutzung)

- `training/dataset_formats.py` (`detect_and_validate`) — Format-Erkennung/Validierung;
  hier kommen neue unterstützte Formate rein
- `training/dataset_refs.py` (`resolve_spec` / `parse_ref`) — `ds:`-Grammatik + Auflösung

## Tests (müssen existieren und grün sein)

- `tests/test_dataset_formats.py` — Validierung pro Format, leere/kaputte Dateien
- `tests/test_dataset_refs.py` — `ds:`-Grammatik, Auflösung, malformed/unknown
- `tests/test_dataset_paths.py`, `tests/test_api.py` — Pfad-Existenz, Upload-Dedup, Delete-Schutz

## Known gaps

- Die Validierung prüft nur Wohlgeformtheit, **kein Schema** — fehlende Felder
  fallen erst beim Training auf.
- Parquet hat keine Plaintext-Vorschau.
- Sub-Sample-Suffix `#N` wird syntaktisch erhalten, aber von der Registry nicht
  interpretiert — die Semantik liegt beim Trainer/Eval-Runner.

## Cross-references

- related_spec: [training-experiments](training-experiments.md) — Konsument der `ds:`-Referenzen
- related_spec: [eval-framework](eval-framework.md) — Eval-Suites referenzieren Datasets gleich
- related_spec: [multimodal-training](multimodal-training.md), [dataset-versioning](dataset-versioning.md),
  [pii-redaction](pii-redaction.md), [labelstudio-import](labelstudio-import.md) — erweitern den `/datasets`-Router
