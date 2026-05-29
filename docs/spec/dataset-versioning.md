---
feature: dataset-versioning
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-16
---

# Dataset-Versionierung, Splits & Mixing

**Geplant (ROADMAP Phase 16) — noch nicht implementiert.**

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

## API surface (geplant — der angestrebte Vertrag)

- POST /datasets/{id}/split?ratio=90:10 → erzeugt train + val
- POST /mixes (Liste aus dataset_id + weight) → composed Dataset

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- neues immutables Version-Feld am Dataset; erweiterte `ds:`-Grammatik

## Extension points (für Plugins / externe Nutzung)

- `training/dataset_refs.py` — `@vN`-Erweiterung der Ref-Grammatik

## Tests (müssen existieren und grün sein)

- (geplant) `@vN`-Parsing + Auflösung
- (geplant) Split-Ratio erzeugt korrekte train/val-Größen; Mix-Gewichtung

## Known gaps

- Gesamtes Feature noch nicht gebaut: kein Version-Feld, keine Split-/Mix-Routen,
  keine `@vN`-Grammatik, keine UI-Erweiterungen.

## Cross-references

- related_spec: [dataset-registry](dataset-registry.md) — erweitert Registry + `ds:`-Grammatik
- adr: ROADMAP.md — Phase 16 „Dataset-Versionierung, Splits, Mixing"
