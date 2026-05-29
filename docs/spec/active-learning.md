---
feature: active-learning
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-11
---

# Active-Learning-Schleife — die ungewissesten Samples zuerst annotieren

**Geplant (ROADMAP Phase 11) — noch nicht implementiert.**

Annotations-Effizienz statt Brute-Force: Nach jedem Training identifiziert das
System die ungewissesten Samples auf unbeschrifteten Dokumenten, surfaced sie
als Annotations-Queue und retraint mit den neu beschrifteten. Ziel ist ein
halb-automatischer Zyklus `train → score uncertain → annotate → retrain` über
mehrere Iterationen mit messbarer Eval-Verbesserung je Runde.

## Capabilities (was der Nutzer tun kann)

- Einen Active-Learning-Lauf starten (Modell + unbeschriftetes Dataset)
- Die Top-N ungewissesten Samples als Annotations-Queue mit Snippet + Confidence sehen
- Die Queue mit Pre-Annotations in ein Label-Studio-Projekt pushen
- Den Loop laufen lassen: Annotation fertig → nächstes Training → Eval → nächster AL-Lauf
- Im UI eine Eval-Score-Kurve über die Iterationen sehen

## Invariants (was immer gelten muss)

- Unsicherheit wird pro Sample gemessen (Token-Entropie und/oder Ensemble-Disagreement)
- Samples werden nach Unsicherheit gerankt; nur die Top-N landen in der Queue
- Eine Queue ist persistiert (`annotation_queues`) und einem Lauf zugeordnet
- Inferenz über unbeschriftete Samples nutzt dieselbe Backend-Schicht wie die Evals

## API surface (geplant — der angestrebte Vertrag)

- POST /active-learning/runs (model + unlabeled_dataset) → startet einen Lauf

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- Top-N-Queue-Größe, Unsicherheits-Methode (entropy / ensemble) pro Lauf

## Extension points (für Plugins / externe Nutzung)

- Unsicherheits-Scorer (token-entropy, ensemble-disagreement) — erweiterbar
- Label-Studio-Push (siehe [labelstudio-import](labelstudio-import.md)) als Gegenrichtung

## Tests (müssen existieren und grün sein)

- (geplant) Ranking liefert die unsichersten Samples zuerst
- (geplant) Loop schließt eine Iteration train→score→retrain ab

## Known gaps

- Gesamtes Feature noch nicht gebaut: keine `annotation_queues`-Tabelle, keine
  Route, kein Scorer, kein UI-Tab, keine LS-Loop-Integration.

## Cross-references

- related_spec: [labelstudio-import](labelstudio-import.md) — Annotations-Quelle/-Ziel
- related_spec: [eval-framework](eval-framework.md) — Eval je Iteration, Inferenz-Backend
- related_spec: [multi-stage-pipelines](multi-stage-pipelines.md) — Retrain-Orchestrierung
- adr: ROADMAP.md — Phase 11 „Active-Learning-Schleife"
