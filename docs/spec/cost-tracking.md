---
feature: cost-tracking
status: partial
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-20
---

# Cost / Resource-Tracking — GPU-Stunden, Watt, $-Äquivalent

**Teilweise implementiert (ROADMAP Phase 20).** GPU-Sekunden werden bei
Completion aus Wall-Clock × GPU-Anzahl berechnet und am Experiment persistiert;
die Study-weite Aggregation (`/studies/cost-summary`) steht. Die Felder
`peak_vram_mb` / `energy_wh` existieren im Schema, werden aber noch nicht
befüllt — dafür fehlt ein nvml-Polling während des Laufs.

GPU-Stunden, Watt und optional ein $-Äquivalent pro Experiment, plus ein
Ranking-Leaderboard „bang per watt". Ziel: jedes Experiment trägt
gpu_seconds, peak_vram und energy_wh, die UI zeigt optionale Cost-Spalten und
einen „Cost vs. Best-Metric"-Plot je Study.

## Capabilities (was der Nutzer tun kann)

- Pro Experiment GPU-Sekunden sehen — **vorhanden**
- Peak-VRAM und Energie (Wh) sehen — **geplant** (Felder vorhanden, noch nicht befüllt)
- In der Experimente-Tabelle optionale Cost-Spalten einblenden
- In Studies einen „Cost vs. Best-Metric"-Plot betrachten (`/studies/cost-summary`)

## Invariants (was immer gelten muss)

- `gpu_seconds` = Wall-Clock × GPU-Anzahl, berechnet bei Terminierung des Runs
- Die Cost-Kennzahlen hängen am Experiment und überdauern dessen Terminierung
- Die Study-Aggregation summiert die Kennzahlen über alle Trials einer Study
- Ohne GPU-Leases bleibt `gpu_seconds` schlicht ungesetzt (kein Crash)

## API surface (der Vertrag für Clients)

- (keine neue Route — Cost-Felder erscheinen am bestehenden Experiment-Record)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- (für ein späteres nvml-Polling an `heartbeat_interval_sec` anlehnbar)

## Extension points (für Plugins / externe Nutzung)

- `ExperimentRecord.gpu_seconds` / `peak_vram_mb` / `energy_wh` — Cost-Felder am Record
- `repository.study_cost_summary` — Aggregation je Study; optionales $-Mapping aufsetzbar

## Tests (müssen existieren und grün sein)

- `tests/test_phase20_cost.py` — `set_experiment_resource_usage` persistiert/partiell,
  gpu_seconds-Mathematik, Record-Defaults, Study-Cost-Aggregation (auch leere Study)

## Known gaps

- `peak_vram_mb` / `energy_wh` werden noch nicht erfasst: es fehlt das nvml-Polling
  während des Laufs (nur `gpu_seconds` aus Wall-Clock × GPU-Anzahl ist gebaut).
- Kein $-Äquivalent-Mapping und kein „bang per watt"-Leaderboard.

## Cross-references

- related_spec: [training-experiments](training-experiments.md) — Cost hängt am Experiment
- related_spec: [platform-foundation](platform-foundation.md) — nvml im Scheduler, `events`-Tabelle
- adr: ROADMAP.md — Phase 20 „Cost/Resource-Tracking"
