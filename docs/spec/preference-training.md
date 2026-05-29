---
feature: preference-training
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-13
---

# Preference-Training — DPO/RLHF in der ExperimentSpec

**Geplant (ROADMAP Phase 13) — noch nicht implementiert.**

Voraussetzung für die DPO-Stage der Multi-Stage-Pipelines und auch standalone
nützlich. Ziel: einen DPO-Spec mit chosen/rejected-Dataset einreichen,
ms-swift trainiert via `swift rlhf`, MLflow zeigt die Preference-Metriken
(reward_chosen, reward_rejected, kl_divergence).

## Capabilities (was der Nutzer tun kann)

- Den Trainingstyp wählen: `sft` | `dpo` | `kto` | `ppo` | `grpo`
- Ein Preference-Dataset im Format `{prompt, chosen, rejected}` einreichen
- Preference-Metriken im zugehörigen MLflow-Run sehen
- Im UI statt „SFT-Type" einen „Training Type"-Select bedienen

## Invariants (was immer gelten muss)

- `train_kind` ist ein geschlossenes Set (`sft`/`dpo`/`kto`/`ppo`/`grpo`)
- Das `{prompt, chosen, rejected}`-Format wird bei der Dataset-Validierung erkannt
- `swift_builder` schaltet bei Nicht-SFT auf `swift rlhf --rlhf_type <kind>` um
  (heute baut er ausschließlich `swift sft`)
- Public-Feldnamen der Spec bleiben stabil; das Flag-Mapping bleibt im swift_builder isoliert

## API surface (der Vertrag für Clients)

- (keine neue Route — erweitert `POST /experiments` um `train_kind` + Preference-Dataset)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `ExperimentSpec.train_kind: Literal["sft","dpo","kto","ppo","grpo"]` (neu)

## Extension points (für Plugins / externe Nutzung)

- `training/swift_builder.py` — `swift sft` vs `swift rlhf`-Verzweigung (einziger Touchpoint)
- `training/dataset_formats.py` — Erkennung des `{prompt, chosen, rejected}`-Schemas

## Tests (müssen existieren und grün sein)

- (geplant) swift_builder erzeugt `swift rlhf --rlhf_type dpo` für DPO-Specs
- (geplant) Preference-Dataset-Format-Validierung (chosen/rejected vorhanden)

## Known gaps

- Gesamtes Feature noch nicht gebaut: kein `train_kind`-Feld, keine `swift rlhf`-
  Verzweigung, kein Preference-Format-Check, kein UI-Select.

## Cross-references

- related_spec: [training-experiments](training-experiments.md) — erweitert die Spec + den swift_builder
- related_spec: [multi-stage-pipelines](multi-stage-pipelines.md) — konsumiert DPO als Stage
- adr: ROADMAP.md — Phase 13 „DPO/RLHF Support in ExperimentSpec"
- docs: https://swift.readthedocs.io/en/latest/Instruction/RLHF.html
