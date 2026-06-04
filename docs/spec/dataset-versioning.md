---
feature: dataset-versioning
status: shipped
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-16
---

# Dataset-Versionierung, Splits & Mixing

Datasets werden versioniert (immutable), Splits deklarativ, und Trainings-Mixes
(z.B. 30% Domäne + 70% Chat) sind first-class. Ziel: `dataset@v2`-Syntax in der
`ds:`-Referenz, `POST /datasets/{id}/split` und `POST /mixes` für gewichtete
Kombinationen.

## Capabilities (was der Nutzer tun kann)

- Eine immutable Version pro Dataset führen
- Ein Dataset deklarativ in train/val splitten (`?ratio=90:10`)
- Mehrere Datasets gewichtet zu einem composed Dataset mischen
- In Specs eine konkrete Version referenzieren: `ds:<id>@v2#500`
- Im UI Version-Badge, Split-Button und Mix-Editor nutzen

## Invariants (was immer gelten muss)

- Eine Dataset-Version ist nach dem Erstellen unveränderlich
- Ein Split erzeugt zwei neue Datasets (train + val) aus einer Quelle
- Ein Mix ist eine gewichtete Liste `dataset_id+weight` → ein composed Dataset
- Die `ds:`-Grammatik wird um `@vN` erweitert, abwärtskompatibel zur heutigen
  `ds:<hex>(#suffix)?`-Form (siehe [dataset-registry](dataset-registry.md))

## API surface (der Vertrag für Clients)

- POST /datasets/{id}/split → 201 (erzeugt zwei abgeleitete Datasets train + val)
- POST /mixes → 201 (Liste aus `dataset_id` + `weight` → ein composed Dataset)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `Dataset.version` (immutable, 1-basiert) + `Dataset.derived_from` (Ableitungs-Link);
  erweiterte `ds:`-Grammatik um `@vN`

## Extension points (für Plugins / externe Nutzung)

- `training/dataset_refs.py` — `@vN`-Erweiterung der Ref-Grammatik (`parse_ref_with_version`)

## Tests (müssen existieren und grün sein)

- `tests/test_phase16_versioning.py` — `ds:<id>@vN`-Parsing + Auflösung,
  Split-Ratio → korrekte train/val-Größen, gewichtete Mix-Komposition

## Known gaps

- Splits/Mixes erzeugen neue, abgeleitete Datasets (über `derived_from` verkettet);
  es gibt kein Editieren bestehender Versionen — Immutabilität ist gewollt.

## Cross-references

- related_spec: [dataset-registry](dataset-registry.md) — erweitert Registry + `ds:`-Grammatik
- adr: ROADMAP.md — Phase 16 „Dataset-Versionierung, Splits, Mixing"
