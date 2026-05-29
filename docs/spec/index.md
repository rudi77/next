# trainpipe — Spezifikations-Index

Verhaltens-erste Spezifikationen, abgeleitet aus dem Live-Code (Status
`shipped`/`partial`) bzw. aus der `ROADMAP.md` (Status `planned`). Eine Spec
beschreibt **was** ein Feature für den Nutzer tut, nicht **wie** es
implementiert ist — ein Refactor (Umbenennen, Datei-Verschieben) darf eine
Spec nicht brechen, eine echte Regression (fehlende Route, verletzter
Invariant) schon.

Stand: 2026-05-29.

## Status-Legende

- `shipped` — gebaut, im Code verifiziert
- `partial` — teilweise gebaut; siehe „Known gaps" der jeweiligen Spec
- `planned` — aus der ROADMAP abgeleiteter Soll-Vertrag, noch nicht implementiert

## Implementiert

| Feature | Status | Datei | Kurzbeschreibung |
|---|---|---|---|
| training-experiments | shipped | [training-experiments.md](training-experiments.md) | Fine-Tuning-Jobs einreichen, disponieren, überwachen, abbrechen |
| hyperparameter-studies | shipped | [hyperparameter-studies.md](hyperparameter-studies.md) | Optuna-Sweeps über Experimente mit dotted-path-Suchraum |
| dataset-registry | shipped | [dataset-registry.md](dataset-registry.md) | Upload, Format-Validierung, sha256-Dedup, `ds:`-Referenzen |
| eval-framework | shipped | [eval-framework.md](eval-framework.md) | Suites/Runs/Metriken, Auto-Eval, n-way Compare, MLflow-Publish |
| model-registry | shipped | [model-registry.md](model-registry.md) | Benannte, versionierte Modelle + Alias-Promotion |
| mcp-server | shipped | [mcp-server.md](mcp-server.md) | trainpipe-Operationen als Agenten-Tools über MCP |
| web-ui | shipped | [web-ui.md](web-ui.md) | Single-Page-UI (Tailwind+Alpine, kein Build-Schritt) |
| platform-foundation | shipped | [platform-foundation.md](platform-foundation.md) | Auth, migrations-versionierte SQLite, GPU-Pool, MLflow, Recovery |

## Geplant (ROADMAP Phase 8+)

| Phase | Feature | Status | Datei | Kurzbeschreibung |
|---|---|---|---|---|
| 8 | inference-playground | planned | [inference-playground.md](inference-playground.md) | Prompt → gestreamte Antwort, Base ↔ Fine-tuned-Vergleich |
| 9 | multimodal-training | partial | [multimodal-training.md](multimodal-training.md) | Doc-Extraktion mit Vision-LLMs, Bundle-Upload, Doc-Metriken |
| 10 | labelstudio-import | planned | [labelstudio-import.md](labelstudio-import.md) | Import aus Label-Studio-Projekten als Dataset |
| 11 | active-learning | planned | [active-learning.md](active-learning.md) | Uncertainty-Sampling-Schleife train→score→annotate→retrain |
| 12 | multi-stage-pipelines | planned | [multi-stage-pipelines.md](multi-stage-pipelines.md) | CPT → SFT → DPO als ein DAG-Objekt |
| 13 | preference-training | planned | [preference-training.md](preference-training.md) | DPO/RLHF in der ExperimentSpec (`swift rlhf`) |
| 14 | synthetic-data | planned | [synthetic-data.md](synthetic-data.md) | Trainings-Pairs aus einem Teacher-LLM generieren |
| 15 | pii-redaction | planned | [pii-redaction.md](pii-redaction.md) | PII-Redaction, Lineage + Audit-Trail (GDPR) |
| 16 | dataset-versioning | planned | [dataset-versioning.md](dataset-versioning.md) | Immutable Versionen, Splits, gewichtete Mixes |
| 17 | continuous-training | planned | [continuous-training.md](continuous-training.md) | Drift-/Cron-Trigger für automatisches Retraining |
| 18 | distributed-training | planned | [distributed-training.md](distributed-training.md) | Multi-Host-Training (DeepSpeed/FSDP, torchrun) |
| 19 | quantization | planned | [quantization.md](quantization.md) | AWQ/GPTQ-Quantisierung + Auto-Eval-Δ |
| 20 | cost-tracking | planned | [cost-tracking.md](cost-tracking.md) | GPU-Stunden, Watt, $-Äquivalent pro Experiment |
| 21 | tokenizer-extension | planned | [tokenizer-extension.md](tokenizer-extension.md) | Fachvokabular als zusätzliche Tokens |

## Konventionen

- Verhalten zuerst, Mechanismus nie. Keine Klassen-/Datei-/Methodennamen in
  Capabilities/Invariants (Extension-Points dürfen Modulpfade nennen).
- Eine Datei pro Spec-Punkt (Subsystem/Feature). Dateinamen lowercase-kebab-case.
- Ziel-Länge 50–120 Zeilen pro Spec.
- Sprache folgt der Projekt-Doku (Deutsch).
- `planned`-Specs sind Soll-Verträge aus der ROADMAP: API-Surfaces sind als
  „(geplant)" markiert und dürfen nicht als existierend gelesen werden.

## Out of Scope (siehe ROADMAP.md)

RAG-Infrastruktur, volle Annotations-UI, Multi-Tenancy/RBAC, eigene
MLflow-Reimplementierung — bewusst ausgeschlossen, mit Begründung in der ROADMAP.
