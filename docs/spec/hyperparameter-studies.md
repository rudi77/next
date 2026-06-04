---
feature: hyperparameter-studies
status: shipped
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr:
---

# Hyperparameter-Studies — Optuna-Sweeps über Experimente

Eine Study sucht automatisch gute Hyperparameter: Optuna schlägt eine
Konfiguration vor, trainpipe reicht sie als ganz normales Experiment ein,
wartet auf das Ergebnis, liest die Zielmetrik aus MLflow und meldet sie
zurück an Optuna. Mehrere Trials laufen parallel. Der Suchraum wird über
dotted-path-Overrides auf eine Basis-`ExperimentSpec` definiert, sodass
jedes Feld der Spec abgetastet werden kann.

## Capabilities (was der Nutzer tun kann)

- Eine Study mit Basis-Spec + Suchraum + Zielmetrik + Richtung starten
- Den Suchraum als dotted-path → Range/Choices angeben (jedes Spec-Feld erreichbar)
- Sampler wählen (TPE / Random / CMA-ES) und Trial-Budget + Parallelität setzen
- Studies auflisten und Fortschritt (abgeschlossene Trials, bester Wert) abrufen
- Eine laufende Study abbrechen
- Trials zu einem Experiment auflösen (jedes Trial ist ein echtes Experiment)

## Invariants (was immer gelten muss)

- Jeder Trial wird als reguläre Experiment-Zeile angelegt und durchläuft den
  normalen Scheduler — Studies haben keinen eigenen Trainingspfad
- Höchstens `max_concurrent` Trials laufen gleichzeitig (Semaphore)
- Die Zielmetrik wird erst nach `status=completed` aus dem MLflow-Run gelesen;
  ein nicht abgeschlossenes oder metrik-loses Trial wird Optuna als FAIL gemeldet
- Nach Crash hängende Optuna-Trials im Zustand RUNNING werden beim Start als
  FAIL abgeschlossen, bevor neue Trials gezogen werden (keine Doppel-Trials)
- Optuna-State liegt pro Study in einer eigenen SQLite-Datei → Resume nach Neustart
- Aktive Studies werden beim API-Start automatisch wieder aufgenommen
- `ds:`-Referenzen in der Basis-Spec werden einmal beim Create aufgelöst

## API surface (der Vertrag für Clients)

- POST /studies → 201 `{study_id}` · 422 (Dataset-Referenz/Pfad-Fehler in der Basis-Spec)
- GET /studies → 200 (alle Studies, neueste zuerst)
- GET /studies/cost-summary → 200 (eine Zeile je Study: aggregierte GPU-Sekunden,
  Peak-VRAM, Energie + bester Wert/Zielmetrik — für den „Cost vs. Best-Metric"-Plot;
  vor `/{id}` deklariert, damit das Literal-Segment zuerst matcht)
- GET /studies/{id} → 200 (inkl. `best_value`, `best_trial_id`) · 404
- POST /studies/{id}/cancel → 200 `{status: cancelled|not_active}` · 404

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `TRAINPIPE_DATA_DIR: path` — unter `studies/` liegen die per-Study Optuna-SQLite-Dateien
- (Sampler, Trial-Anzahl, Parallelität kommen pro Study aus `StudyConfig`, nicht aus Env)

## Extension points (für Plugins / externe Nutzung)

- `autoresearch/search_spaces.py` (`sample_spec`) — übersetzt Suchraum-Einträge in
  Optuna-Vorschläge und appliziert sie als dotted-path-Overrides auf die Basis-Spec
- `StudyConfig.sampler` — `tpe` | `random` | `cmaes`

## Tests (müssen existieren und grün sein)

- `tests/test_study_manager.py` — Driver-Lifecycle, Resume aktiver Studies, Cancel
- `tests/test_search_spaces.py` — dotted-path-Sampling auf die Basis-Spec

## Known gaps

- Single-Objective: genau eine Zielmetrik pro Study (kein Multi-Objective).
- Liest ms-swift einen Metriknamen nicht in MLflow weg, scheitert das Trial als FAIL —
  es gibt keine Fallback-Metrik-Auflösung.

## Cross-references

- related_spec: [training-experiments](training-experiments.md) — jeder Trial ist ein Experiment
- related_spec: [platform-foundation](platform-foundation.md) — MLflow-Wiring, Persistenz
- related_spec: [cost-tracking](cost-tracking.md) — Quelle der Cost-Kennzahlen in `/studies/cost-summary`
