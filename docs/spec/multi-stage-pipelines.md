---
feature: multi-stage-pipelines
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-12
---

# Multi-Stage-Pipelines — CPT → SFT → DPO als ein Objekt

**Geplant (ROADMAP Phase 12) — noch nicht implementiert.**

Ein Domain-LLM-Workflow als ein deklariertes Objekt: continued pretraining →
instruction tuning → preference alignment. Jede Stage übernimmt den Checkpoint
der vorigen. Ziel: ein 3-Stage-DAG einreichen, jede Stage startet nach Erfolg
der vorigen, der finale Checkpoint wird registriert.

## Capabilities (was der Nutzer tun kann)

- Eine Pipeline als geordnete Stages (jede referenziert eine ExperimentSpec) deklarieren
- Abhängigkeiten + „input-from-stage" zwischen Stages angeben
- Pipelines auflisten, Detail abrufen, abbrechen (kaskadiert auf alle Stages)
- Im UI eine DAG-Ansicht mit Status pro Stage sehen und in das jeweilige Experiment drillen

## Invariants (was immer gelten muss)

- Eine Stage startet erst, wenn ihr Vorgänger `completed` ist
- Der Adapter-/Checkpoint-Pfad einer Stage wird als Input der Folge-Stage propagiert
- Jede Stage läuft als reguläres Experiment durch den normalen Scheduler
- Angefangene Pipelines werden nach einem Crash resümiert (Recovery wie bei Studies)
- Cancel einer Pipeline kaskadiert auf alle noch nicht terminalen Stages
- Der finale Checkpoint wird in der Modell-Registry registriert

## API surface (geplant — der angestrebte Vertrag)

- POST /pipelines (3-Stage-DAG) → startet die Pipeline
- GET /pipelines · GET /pipelines/{id}
- POST /pipelines/{id}/cancel → kaskadiert auf alle Stages

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- Schema: Migration v5 (`pipelines`, `pipeline_stages`); `PipelineConfig(stages: list[StageSpec])`

## Extension points (für Plugins / externe Nutzung)

- Pipeline-Driver analog zum `StudyDriver` (überwacht Stages, propagiert Pfade, Recovery)

## Tests (müssen existieren und grün sein)

- (geplant) Stage N startet erst nach Completion von Stage N-1
- (geplant) Adapter-Pfad-Propagation + Cancel-Kaskade + Crash-Resume

## Known gaps

- Gesamtes Feature noch nicht gebaut: keine Migration v5, kein Pipeline-Driver,
  keine Routen, kein UI-Tab.
- Die DPO-Stage setzt [preference-training](preference-training.md) (Phase 13) voraus.

## Cross-references

- related_spec: [preference-training](preference-training.md) — liefert die DPO-Stage
- related_spec: [training-experiments](training-experiments.md) — jede Stage ist ein Experiment
- related_spec: [model-registry](model-registry.md) — registriert den finalen Checkpoint
- adr: ROADMAP.md — Phase 12 „Multi-Stage Pipelines (CPT → SFT → DPO)"
