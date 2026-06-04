---
feature: continuous-training
status: shipped
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-17
---

# Continuous Training / Drift-Detection

Sinkt die Eval-Metrik unter einen Threshold, wird automatisch ein Retraining
getriggert — oder zeitgesteuert „alle N Sekunden neu trainieren". Eine
Watch-Config triggert auf ein festes Intervall **oder** einen Metrik-Threshold
und startet eine gespeicherte Pipeline.

## Capabilities (was der Nutzer tun kann)

- Eine Watch anlegen, die auf festes Intervall **oder** Metrik-Threshold triggert
- Eine Watch aktivieren/deaktivieren und löschen
- Im UI-Tab „Watches" Watches und ihren letzten Trigger sehen

## Invariants (was immer gelten muss)

- Ein Trigger startet einen Pipeline-Run aus der gespeicherten `PipelineConfig`
  (siehe [multi-stage-pipelines](multi-stage-pipelines.md))
- `interval`: feuert, sobald seit dem letzten Feuern `interval_seconds` vergangen sind
- `metric_threshold`: feuert, wenn der Mittelwert der Zielmetrik des **zuletzt
  abgeschlossenen** Eval-Runs gegen die Suite unter den Threshold fällt
- Eine deaktivierte Watch wird nicht gepollt und löst nichts aus
- Nach mehreren aufeinanderfolgenden Fehl-Feuern deaktiviert sich die Watch
  automatisch (Failure-Counter, zurückgesetzt bei Erfolg)
- Watches sind persistiert (kind, enabled, pipeline_config, Trigger-Parameter)

## API surface (der Vertrag für Clients)

- POST /watches → 201 (`kind=interval` braucht `interval_seconds`;
  `kind=metric_threshold` braucht `model_name`/`suite_id`/`metric_name`/`threshold`) · 422
- GET /watches → 200 · GET /watches/{id} → 200 · 404
- POST /watches/{id}/enable → 200 · POST /watches/{id}/disable → 200
- DELETE /watches/{id} → 200

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- Eine async Poll-Loop im Watch-Manager (`poll_interval_sec`, Default 30s)
- `failure_disable_threshold` (Default 5) — Fehl-Feuer bis zur Auto-Deaktivierung

## Extension points (für Plugins / externe Nutzung)

- Trigger-Typen (`interval`, `metric_threshold`) — erweiterbar
- Score-Quelle: der jüngste abgeschlossene Eval-Run gegen die konfigurierte Suite

## Tests (müssen existieren und grün sein)

- `tests/test_phase17_watches.py` — Create/Validierung je Kind, enable/disable,
  Interval feuert bei Fälligkeit (und nicht sofort erneut), Threshold feuert unter/
  nicht über dem Wert, Failure-Counter + Auto-Disable, deaktivierte Watch wird nicht gepollt

## Known gaps

- Trigger ist ein festes Intervall, **kein** Cron-Ausdruck.
- Threshold prüft den jüngsten Eval-Run, keinen gleitenden Fenster-Mittelwert über
  Production-Inference-Scores.

## Cross-references

- related_spec: [multi-stage-pipelines](multi-stage-pipelines.md) — Trigger-Ziel
- related_spec: [eval-framework](eval-framework.md) — Score-Quelle für Threshold-Trigger
- related_spec: [inference-playground](inference-playground.md) — Production-Inference-Scores
- adr: ROADMAP.md — Phase 17 „Continuous Training / Drift-Detection"
