# trainpipe — Spezifikations-Index

Verhaltens-erste Spezifikationen, abgeleitet aus dem Live-Code (Status
`shipped`/`partial`) bzw. aus der `ROADMAP.md` (Status `planned`). Eine Spec
beschreibt **was** ein Feature für den Nutzer tut, nicht **wie** es
implementiert ist — ein Refactor (Umbenennen, Datei-Verschieben) darf eine
Spec nicht brechen, eine echte Regression (fehlende Route, verletzter
Invariant) schon.

Stand: 2026-06-04.

## Status-Legende

- `shipped` — gebaut, im Code verifiziert
- `partial` — teilweise gebaut; siehe „Known gaps" der jeweiligen Spec
- `planned` — aus der ROADMAP abgeleiteter Soll-Vertrag, noch nicht implementiert

## Implementiert (`shipped`)

| Phase | Feature | Status | Datei | Kurzbeschreibung |
|---|---|---|---|---|
| — | training-experiments | shipped | [training-experiments.md](training-experiments.md) | Fine-Tuning-Jobs einreichen, disponieren, überwachen, abbrechen |
| — | hyperparameter-studies | shipped | [hyperparameter-studies.md](hyperparameter-studies.md) | Optuna-Sweeps über Experimente mit dotted-path-Suchraum |
| — | dataset-registry | shipped | [dataset-registry.md](dataset-registry.md) | Upload, Format-Validierung, sha256-Dedup, `ds:`-Referenzen |
| — | eval-framework | shipped | [eval-framework.md](eval-framework.md) | Suites/Runs/Metriken, Auto-Eval, n-way Compare, MLflow-Publish |
| 7 | model-registry | shipped | [model-registry.md](model-registry.md) | Benannte, versionierte Modelle + Alias-Promotion |
| — | mcp-server | shipped | [mcp-server.md](mcp-server.md) | trainpipe-Operationen als Agenten-Tools über MCP |
| — | web-ui | shipped | [web-ui.md](web-ui.md) | Single-Page-UI (Tailwind+Alpine, kein Build-Schritt) |
| — | platform-foundation | shipped | [platform-foundation.md](platform-foundation.md) | Auth, migrations-versionierte SQLite, GPU-Pool, MLflow, Recovery |
| 9 | multimodal-training | shipped | [multimodal-training.md](multimodal-training.md) | Doc-Extraktion mit Vision-LLMs, Bundle-Upload, Doc-Metriken |
| 10 | labelstudio-import | shipped | [labelstudio-import.md](labelstudio-import.md) | Import aus Label-Studio-Projekten als Dataset |
| 11 | active-learning | shipped | [active-learning.md](active-learning.md) | Uncertainty-Sampling-Schleife train→score→annotate→retrain |
| 12 | multi-stage-pipelines | shipped | [multi-stage-pipelines.md](multi-stage-pipelines.md) | CPT → SFT → DPO als ein DAG-Objekt |
| 13 | preference-training | shipped | [preference-training.md](preference-training.md) | DPO/RLHF in der ExperimentSpec (`swift rlhf`) |
| 14 | synthetic-data | shipped | [synthetic-data.md](synthetic-data.md) | Trainings-Pairs aus einem Teacher-LLM generieren |
| 15 | pii-redaction | shipped | [pii-redaction.md](pii-redaction.md) | PII-Redaction, Lineage + Audit-Trail (GDPR) |
| 16 | dataset-versioning | shipped | [dataset-versioning.md](dataset-versioning.md) | Immutable Versionen, Splits, gewichtete Mixes |
| 17 | continuous-training | shipped | [continuous-training.md](continuous-training.md) | Interval-/Threshold-Trigger für automatisches Retraining |
| 19 | quantization | shipped | [quantization.md](quantization.md) | AWQ/GPTQ-Quantisierung als neue Modellversion |

## Teilweise implementiert (`partial`)

Kern gebaut, einzelne Spec-Punkte fehlen noch — Details unter „Known gaps" der jeweiligen Spec.

| Phase | Feature | Status | Datei | Was noch fehlt |
|---|---|---|---|---|
| 8 | inference-playground | partial | [inference-playground.md](inference-playground.md) | Single/Stream/Compare/Cache + MCP stehen; `DELETE /inferences/{id}` (Abbruch) fehlt |
| 18 | distributed-training | partial | [distributed-training.md](distributed-training.md) | Single-Host-ZeRO + Multi-Node-Env stehen; echtes torchrun-argv + Multi-Host-Pool fehlen |
| 20 | cost-tracking | partial | [cost-tracking.md](cost-tracking.md) | `gpu_seconds` + Study-Aggregation stehen; `peak_vram_mb`/`energy_wh` (nvml-Polling) fehlen |
| 21 | tokenizer-extension | partial | [tokenizer-extension.md](tokenizer-extension.md) | `extra_tokens`-Durchreichung steht; Tokenisierungs-Eval-Hook fehlt |

## Konventionen

- Verhalten zuerst, Mechanismus nie. Keine Klassen-/Datei-/Methodennamen in
  Capabilities/Invariants (Extension-Points dürfen Modulpfade nennen).
- Eine Datei pro Spec-Punkt (Subsystem/Feature). Dateinamen lowercase-kebab-case.
- Ziel-Länge 50–120 Zeilen pro Spec.
- Sprache folgt der Projekt-Doku (Deutsch).
- `partial`-Specs markieren noch nicht gebaute Surfaces inline (z.B. „— geplant,
  noch nicht gebaut") und führen die Lücke unter „Known gaps"; alles andere ist
  im Code verifiziert. Rein `planned`-Specs gibt es derzeit keine mehr.

## Out of Scope (siehe ROADMAP.md)

RAG-Infrastruktur, volle Annotations-UI, Multi-Tenancy/RBAC, eigene
MLflow-Reimplementierung — bewusst ausgeschlossen, mit Begründung in der ROADMAP.
