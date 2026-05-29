---
feature: pii-redaction
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-15
---

# PII-Redaction & Audit-Trail — GDPR-tauglich für DACH/Company-Daten

**Geplant (ROADMAP Phase 15) — noch nicht implementiert.**

Vor jedem Training wird der Datensatz durch eine PII-Detection geschickt; ein
auditierbarer Trail hält fest, welches Modell mit welchen Datasets (Hashes)
trainiert wurde; „Recht auf Löschung" wird verfolgbar. Ziel: Upload mit
`--auto-redact` erzeugt einen redacted-Twin; das Modell-Detail zeigt seine
Trainings-Datasets; ein Tool listet alle Modelle, die ein Original-Dataset sahen.

## Capabilities (was der Nutzer tun kann)

- Beim Upload automatisch redigieren lassen (`--auto-redact`) → redacted-Twin
- Ein bestehendes Dataset redigieren (Entity-Typen + Ersetzungsstrategie wählen)
- Im Modell-Detail sehen, aus welchen Datasets (Versionen) ein Modell trainiert wurde
- Suchen: „welche Modelle haben Dataset X benutzt?"
- „Forget user Y": betroffene Datasets identifizieren und abhängige Modelle zum
  Retraining markieren

## Invariants (was immer gelten muss)

- Ein redacted-Dataset ist ein **neues** Dataset mit Provenienz-Link zum Original
- Lineage verknüpft Modell ↔ Dataset (mit Zeitpunkt) und ist auditierbar
- PII-Erkennung (presidio / spaCy-NER) ist eine optionale Dependency; ohne sie
  bleibt der Rest der API lauffähig (graceful degrade)

## API surface (geplant — der angestrebte Vertrag)

- POST /datasets/{id}/redact (entities: list, replacement_strategy) → redacted-Twin

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- Schema: Migration v6 (`model_lineage`: model_id, dataset_id, used_at)
- presidio/spaCy als optionale Installations-Extra

## Extension points (für Plugins / externe Nutzung)

- PII-Backend (presidio vs spaCy-NER) — austauschbar
- Ersetzungsstrategien (Maskierung / Pseudonymisierung / Entfernung)

## Tests (müssen existieren und grün sein)

- (geplant) Redaction erzeugt neues Dataset mit Provenienz-Link
- (geplant) Lineage-Query findet alle Modelle, die ein Dataset gesehen haben

## Known gaps

- Gesamtes Feature noch nicht gebaut: keine Redact-Route, keine `model_lineage`-
  Tabelle, kein PII-Backend, keine Lineage-UI, kein „Forget"-Workflow.

## Cross-references

- related_spec: [dataset-registry](dataset-registry.md) — Redaction = neues Dataset + Provenienz
- related_spec: [model-registry](model-registry.md) — Lineage hängt am registrierten Modell
- adr: ROADMAP.md — Phase 15 „PII Redaction & Audit-Trail"
- docs: https://github.com/microsoft/presidio
