---
feature: continuous-training
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-17
---

# Continuous Training / Drift-Detection

**Geplant (ROADMAP Phase 17) — noch nicht implementiert.**

Sinkt die Production-Metrik, wird automatisch ein Retraining getriggert — oder
zeitgesteuert „jede Woche neu trainieren mit den neuen Docs". Ziel: eine
Watch-Config kann auf einen Eval-Score-Threshold oder einen Cron-Schedule
triggern und einen neuen Pipeline-Run starten.

## Capabilities (was der Nutzer tun kann)

- Eine Watch anlegen, die auf Cron-Schedule **oder** Metrik-Threshold triggert
- Eine Watch aktivieren/deaktivieren
- Im UI-Tab „Watches" Watches und ihre letzten Trigger sehen

## Invariants (was immer gelten muss)

- Ein Trigger startet einen Pipeline-Run (siehe [multi-stage-pipelines](multi-stage-pipelines.md))
- Der Drift-Monitor triggert, wenn der gleitende Fenster-Mittelwert der
  Production-Scores unter den Threshold fällt
- Eine deaktivierte Watch löst nichts aus
- Watches sind persistiert (`watches`: trigger, pipeline_id, enabled)

## API surface (geplant — der angestrebte Vertrag)

- (CRUD für Watches; konkrete Routen noch offen in der ROADMAP)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- Schema: `watches`-Tabelle (trigger = cron / metric-threshold, pipeline_id, enabled)
- Scheduler-Komponente (APScheduler o.ä.) im bestehenden Scheduler-Prozess

## Extension points (für Plugins / externe Nutzung)

- Trigger-Typen (cron, metric-threshold) — erweiterbar
- Drift-Quelle: Production-Inference-Scores (rolling window)

## Tests (müssen existieren und grün sein)

- (geplant) Cron-Trigger startet einen Pipeline-Run
- (geplant) Threshold-Unterschreitung im rolling window triggert; disabled triggert nicht

## Known gaps

- Gesamtes Feature noch nicht gebaut: keine `watches`-Tabelle, kein Scheduler-
  Trigger, kein Drift-Monitor, kein UI-Tab.
- Setzt [multi-stage-pipelines](multi-stage-pipelines.md) als Trigger-Ziel voraus.

## Cross-references

- related_spec: [multi-stage-pipelines](multi-stage-pipelines.md) — Trigger-Ziel
- related_spec: [eval-framework](eval-framework.md) — Score-Quelle für Threshold-Trigger
- related_spec: [inference-playground](inference-playground.md) — Production-Inference-Scores
- adr: ROADMAP.md — Phase 17 „Continuous Training / Drift-Detection"
