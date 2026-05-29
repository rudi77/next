---
feature: cost-tracking
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-20
---

# Cost / Resource-Tracking — GPU-Stunden, Watt, $-Äquivalent

**Geplant (ROADMAP Phase 20) — noch nicht implementiert.**

GPU-Stunden, Watt und optional ein $-Äquivalent pro Experiment, plus ein
Ranking-Leaderboard „bang per watt". Ziel: jedes Experiment trägt
gpu_seconds, peak_vram und energy_wh, die UI zeigt optionale Cost-Spalten und
einen „Cost vs. Best-Metric"-Plot je Study.

## Capabilities (was der Nutzer tun kann)

- Pro Experiment GPU-Sekunden, Peak-VRAM und Energie (Wh) sehen
- In der Experimente-Tabelle optionale Cost-Spalten einblenden
- In Studies einen „Cost vs. Best-Metric"-Plot betrachten

## Invariants (was immer gelten muss)

- Während eines Runs pollt der Scheduler nvml und aggregiert die Werte in `events`
- Die Cost-Kennzahlen hängen am Experiment und überdauern dessen Terminierung
- Ohne nvml/Treiber degradiert das Tracking sauber (keine Werte statt Crash)

## API surface (der Vertrag für Clients)

- (keine neue Route — Cost-Felder erscheinen am bestehenden Experiment-Record)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- nvml-Polling-Intervall (an `heartbeat_interval_sec` anlehnbar)

## Extension points (für Plugins / externe Nutzung)

- Aggregations-Senke: `events`-Tabelle (bestehend) als Roh-Datenpunkt-Speicher
- optionales $-Mapping (Energie/GPU-Stunde → Kosten)

## Tests (müssen existieren und grün sein)

- (geplant) nvml-Polling aggregiert gpu_seconds/peak_vram/energy_wh in events
- (geplant) Cost-Kennzahlen am Experiment-Record nach Completion

## Known gaps

- Gesamtes Feature noch nicht gebaut: kein nvml-Polling im Scheduler, keine
  Cost-Felder, keine UI-Spalten/-Plots.

## Cross-references

- related_spec: [training-experiments](training-experiments.md) — Cost hängt am Experiment
- related_spec: [platform-foundation](platform-foundation.md) — nvml im Scheduler, `events`-Tabelle
- adr: ROADMAP.md — Phase 20 „Cost/Resource-Tracking"
