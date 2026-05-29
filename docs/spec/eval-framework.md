---
feature: eval-framework
status: shipped
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-6
---

# Eval-Framework — messbare Scores und Run-Vergleiche

Ein Run bekommt einen messbaren Score gegen eine wiederholbare Eval-Suite.
Eine Suite bündelt ein Dataset + eine Liste von Metriken + Inferenz-Parameter.
Ein Eval-Run führt das Modell eines Experiments über die Samples, scort jede
Vorhersage mit jeder Metrik, persistiert pro Sample Prediction+Scores und
aggregiert mean/std/count. Zwei oder mehr Runs lassen sich side-by-side
vergleichen — inklusive der einzelnen Samples, bei denen einer regressed ist.
Metriken sind Plugins (Directory-Scan), Inferenz-Backends austauschbar.

## Capabilities (was der Nutzer tun kann)

- Eine wiederverwendbare Eval-Suite anlegen (Dataset + Metriken + Inferenz-Params)
- Einen Eval-Run manuell triggern (`suite_id` + `experiment_id`)
- Nach erfolgreichem Training automatisch eval'en (`ExperimentSpec.auto_eval`)
- Per-Run-Aggregat und Per-Sample-Predictions+Scores abrufen
- Einen laufenden Eval-Run abbrechen
- N Runs derselben Suite vergleichen (Aggregat-Δ + Liste divergierender Samples)
- Aus 5 mitgelieferten Metriken wählen: `exact_match`, `field_level_f1`,
  `rouge_l`, `bleu`, `llm_as_judge`; eigene als Plugin ergänzen

## Invariants (was immer gelten muss)

- Beim Suite-Create werden `ds:`-Refs aufgelöst, das Dataset muss existieren,
  und jede Metrik wird instanziiert (Config-Validierung) bevor persistiert wird
- Suite-Namen sind eindeutig (409 bei Duplikat)
- Eval-Runs teilen sich den GPU-Pool mit dem Training (1 GPU pro Run, 0 auf
  GPU-losen Hosts), mit atomarem Claim und Crash-Recovery wie beim Scheduler
- Ein Predict-Fehler pro Sample tötet den Run nicht: der Fehler wird auf der
  Result-Zeile vermerkt und die Metrik zählt 0.0 für dieses Sample
- Eine Metrik, die auf einem Sample wirft, liefert 0.0 statt den Run zu killen
- Doppelte Metrik-Namen in einer Suite sind unzulässig (`name` zum Disambiguieren)
- `compare` verlangt ≥2 Runs **derselben** Suite (sonst 422)
- MLflow-Publish nach Completion ist best-effort — schlägt er fehl, bleibt der
  Eval-Run trotzdem `completed`
- Unbekannte `auto_eval`-Suite-IDs werden geloggt und übersprungen, statt das
  gerade abgeschlossene Experiment zu failen

## API surface (der Vertrag für Clients)

- POST /evals/suites → 201 · 409 (`name_exists`) · 422 (Dataset-Ref / Metrik-Config ungültig)
- GET /evals/suites → 200 · GET /evals/suites/{id} → 200 · 404
- DELETE /evals/suites/{id} → 200 · 409 (`suite_in_use`, ohne `?force=true`)
- POST /evals/runs → 201 · 422 (`unknown_suite` / `unknown_experiment`)
- GET /evals/runs → 200 (Filter `suite_id`, `experiment_id`, `status`)
- GET /evals/runs/{id} → 200 · 404 · GET /evals/runs/{id}/results → 200 · 404
- POST /evals/runs/{id}/cancel → 200 `{status}` · 404
- GET /evals/compare?run_ids=a,b,c → 200 (Aggregat-Δ + Regressions-Samples) · 404 · 422 (`suite_mismatch`)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `TRAINPIPE_MLFLOW_TRACKING_URI` — Ziel für `eval.<suite>.<metric>` Metrics + Tags
- Provider/Modell für `llm_as_judge` werden pro Suite + über Provider-Env-Vars konfiguriert
- Inferenz-Backend: `TransformersInferenceBackend` (prod) bzw. `MockInferenceBackend` (Tests/Fallback)

## Extension points (für Plugins / externe Nutzung)

- `trainpipe/evals/metrics/` — jede Datei mit einer `Metric`-Subklasse (`kind` gesetzt,
  `score()` implementiert) wird beim ersten Lookup auto-registriert
- `evals/inference.py` (`InferenceBackend`) — neue Backends (vLLM/sglang/swift) via
  `backend_factory` des Dispatchers einhängen

## Tests (müssen existieren und grün sein)

- `tests/test_eval_metrics_exact_match.py`, `test_eval_metrics_more.py`, `test_eval_metrics_bleu.py`
- `tests/test_eval_runner.py`, `tests/test_eval_repository.py`
- `tests/test_api_evals.py`, `tests/test_auto_eval_hook.py`, `tests/test_eval_mlflow_logging.py`
- `tests/test_eval_e2e.py` — Suite → 2 Experimente mit `auto_eval` → Dispatcher → Compare mit Δ

## Known gaps

- Der In-process-Inferenz-Runner ist nicht throughput-optimiert; `llm_as_judge`
  kann bei großen Suites langsam werden (eigener Pool als Folge-Refactor vorgesehen).
- Eval-Datasets unterstützen jsonl/json/csv/tsv — **kein** Parquet im Runner.

## Cross-references

- related_spec: [training-experiments](training-experiments.md) — `auto_eval`-Hook nach Completion
- related_spec: [dataset-registry](dataset-registry.md) — Suite-Datasets via `ds:`-Refs
- related_spec: [model-registry](model-registry.md) — Eval-Summary ist die Promotion-Basis
- adr: ROADMAP.md — Phase 6 „Eval-Framework"
