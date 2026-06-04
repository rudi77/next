---
feature: pii-redaction
status: shipped
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-15
---

# PII-Redaction & Audit-Trail — GDPR-tauglich für DACH/Company-Daten

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
- Lineage verknüpft Modell ↔ Dataset (mit Zeitpunkt) und ist auditierbar;
  die Lineage-Abfrage folgt Mix-/Ableitungs-Ketten (rekursiv)
- PII-Erkennung ist ein eingebauter, konservativer Regex-Redactor (email, phone,
  iban mit ISO-13616-Prüfsumme, credit_card, de_tax_id) — keine externe Dependency;
  Entity-Typen sind pro Aufruf abwählbar
- Redaction läuft nur über JSONL-Datasets; andere Formate werden abgelehnt

## API surface (der Vertrag für Clients)

- POST /datasets/{id}/redact → 201 (`entities`-Auswahl; erzeugt einen redacted-Twin
  mit Provenienz-Link) · 422 (Nicht-JSONL-Dataset)
- GET /datasets/{id}/models → 200 (welche Modelle dieses Dataset sahen — folgt Mix-/Ableitungs-Ketten)
- GET /models/{id}/datasets → 200 (Trainings-Lineage einer Modellversion)
- POST /compliance/forget-scan → 200 (Report: welche Datasets/Modelle einen Term/Regex
  enthalten) · 422 (ungültiger Regex)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- Schema: `model_lineage`-Tabelle (model_id, dataset_id, used_at) — Migration in `core/db.py`
- Keine externen PII-Dependencies; der Regex-Redactor ist eingebaut

## Extension points (für Plugins / externe Nutzung)

- `redaction/redactor.py` — Entity-Pattern + Ersetzungsmarker; höhere Recall-Backends
  (z.B. Presidio) lassen sich hier einhängen
- `compliance/forget.py` — Term-/Regex-Scan über Datasets

## Tests (müssen existieren und grün sein)

- `tests/test_phase15_pii.py` — Entity-Redaction (email/iban-Prüfsumme/phone/credit_card),
  Redact-Route erzeugt neues Dataset + 422 bei Nicht-JSONL, Lineage-Aufzeichnung beim
  Registrieren, rekursive „welche Modelle nutzten Dataset X?"-Abfrage

## Known gaps

- Der Regex-Redactor ist bewusst konservativ (Precision vor Recall); für höheren
  Recall ist ein Presidio-/NER-Backend vorgesehen, aber nicht implementiert.
- „Forget user Y" identifiziert betroffene Datasets/Modelle; das automatische
  Markieren abhängiger Modelle zum Retraining ist ein Report, kein Auto-Trigger.

## Cross-references

- related_spec: [dataset-registry](dataset-registry.md) — Redaction = neues Dataset + Provenienz
- related_spec: [model-registry](model-registry.md) — Lineage hängt am registrierten Modell
- adr: ROADMAP.md — Phase 15 „PII Redaction & Audit-Trail"
- docs: https://github.com/microsoft/presidio
