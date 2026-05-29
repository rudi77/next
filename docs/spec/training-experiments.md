---
feature: training-experiments
status: shipped
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr:
---

# Training-Experimente — Fine-Tuning-Jobs einreichen und überwachen

Das Herzstück: ein Nutzer reicht eine `ExperimentSpec` ein (Modell,
Datasets, Hyperparameter, GPU-Anzahl), der Job landet in einer FIFO-Queue,
wird auf freie GPUs disponiert, startet ms-swift als Subprozess und läuft
bis `completed`/`failed`/`cancelled`. Logs sind live verfolgbar, jeder Run
ist mit einem MLflow-Run verknüpft. Crashes der API verlieren keine Jobs —
laufende Experimente werden beim Neustart requeued.

## Capabilities (was der Nutzer tun kann)

- Ein Experiment einreichen (einzeln oder als Batch)
- Den Trainingstyp wählen (LoRA/QLoRA/Full/… ) plus alle gängigen Hyperparameter
- Datasets per HF-Id, lokalem Pfad oder `ds:<id>`-Registry-Referenz angeben
- Mehrere GPUs pro Job anfordern und eine Priorität setzen
- Experimente auflisten (gefiltert nach Status / Study) und Details abrufen
- Ein queued oder laufendes Experiment abbrechen
- Das Trainings-Log komplett herunterladen oder live per SSE mitlesen
- Optionale multimodale Settings (VLM) und beliebige ms-swift-Extra-Args durchreichen
- Nach erfolgreichem Training automatisch Eval-Suites triggern (`auto_eval`)

## Invariants (was immer gelten muss)

- Die Queue wird FIFO nach `(priorität DESC, queued_at ASC)` abgearbeitet
- Ein Experiment wird genau einmal disponiert: das Claim ist ein atomares
  CAS-UPDATE, ein gleichzeitiger Cancel zwischen Auswahl und Claim gewinnt
- GPU-intensive Arbeit (MLflow-Run, Subprozess-Spawn) läuft **außerhalb** des
  Dispatch-Locks, damit MLflow-Latenz parallele Submits nicht serialisiert
- Fordert ein Spec mehr GPUs als der Pool besitzt, wird es sofort `failed`
  (kein ewiges Hängen in `queued`)
- Bei Launch-Fehlern (MLflow- oder Spawn-Fehler) werden GPU-Leases wieder
  freigegeben und der Run als `failed` markiert — keine geleakten Leases
- Nach API-Crash werden `running`-Zeilen beim Start auf `queued` zurückgesetzt
  (mit neuem `queued_at`, damit Alt-Jobs neue nicht aushungern) **bevor**
  GPU-Leases synchronisiert werden
- Ein leeres `dataset` wird am Submit mit 422 abgelehnt (nicht erst vom Trainer)
- Lokal aussehende Dataset-Pfade, die nicht existieren, werden am Submit mit 422 abgelehnt
- Cancel eines laufenden Jobs beendet die ganze Prozessgruppe (SIGTERM, dann SIGKILL)

## API surface (der Vertrag für Clients)

- POST /experiments → 201 `{experiment_id}`
- POST /experiments → 422 (`empty_dataset` / `unknown_dataset_ref` / `malformed_dataset_ref` / `missing_local_paths`)
- POST /experiments/batch → 201 `{experiment_ids}` · 422 (leere Liste oder ungültige Specs)
- GET /experiments → 200 (Filter `status`, `study_id`, `limit`, `offset`)
- GET /experiments/{id} → 200 · 404
- POST /experiments/{id}/cancel → 200 `{status: cancelled|cancelling|...}` · 404
- GET /experiments/{id}/logs → 200 (Plaintext, leer wenn noch kein Log)
- GET /experiments/{id}/logs/stream → 200 (SSE: `log`-Events bis terminales `end`-Event)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `TRAINPIPE_VISIBLE_GPUS: list[int]` (default alle) — welche GPU-Indizes der Pool sieht
- `TRAINPIPE_POLL_INTERVAL_SEC: float` (default `1.0`) — Scheduler-Tick
- `TRAINPIPE_DATA_DIR: path` (default `./data`) — Wurzel für Outputs/Logs/DB

## Extension points (für Plugins / externe Nutzung)

- `training/swift_builder.py` — einziger Ort, der `ExperimentSpec`-Felder auf
  die aktuellen ms-swift-CLI-Flags abbildet (Versions-Übersetzungsschicht)
- `ExperimentSpec.extra_args` — beliebige zusätzliche ms-swift-Flags durchreichen

## Tests (müssen existieren und grün sein)

- `tests/test_api.py` — Submit/Get/List/Cancel, Dataset-Validierung, SSE-Pfad
- `tests/test_swift_builder.py`, `tests/test_swift_resolver.py` — Flag-Mapping, Binary-Auflösung
- `tests/test_repository.py`, `tests/test_dataset_paths.py`

## Known gaps

- Echtes Training läuft nur unter Linux (POSIX-Prozessgruppen via `os.setsid`/`os.killpg`);
  auf Windows bootet die API, aber Jobs lassen sich nicht real ausführen.
- Kein Heartbeat-Timeout: ein Subprozess, der hängt ohne zu sterben, bleibt `running`.

## Cross-references

- related_spec: [dataset-registry](dataset-registry.md) — Quelle der `ds:`-Referenzen
- related_spec: [hyperparameter-studies](hyperparameter-studies.md) — erzeugt Experimente als Trials
- related_spec: [eval-framework](eval-framework.md) — `auto_eval`-Hook nach Completion
- related_spec: [platform-foundation](platform-foundation.md) — GPU-Pool, Persistenz, Recovery
