---
feature: active-learning
status: shipped
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-11
---

# Active-Learning-Schleife — die ungewissesten Samples zuerst annotieren

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

- Unsicherheit wird pro Sample gemessen — `double_pass` (zwei T=0.7-Samples +
  Diff) oder `length_zscore` (Abweichung von der mittleren Antwortlänge)
- Samples werden nach Unsicherheit gerankt; nur die Top-N landen in der Queue
- Eine Queue ist persistiert und einem Lauf zugeordnet
- Inferenz über unbeschriftete Samples nutzt dieselbe Backend-Schicht wie die Evals
- Stürzt der Scorer ab, endet der Lauf sauber als `failed` (kein Hängen)

## API surface (der Vertrag für Clients)

- POST /active-learning/runs → 201 (`model_ref` + `dataset` + `top_n` + `scorer`) ·
  422 (`bad_model_ref` / fehlende Dataset-Datei)
- GET /active-learning/runs → 200 · GET /active-learning/runs/{id} → 200 · 404
- GET /active-learning/runs/{id}/queue → 200 (Top-N mit Snippet + Uncertainty)
- POST /active-learning/runs/{id}/queue/{item_id}/annotated → 200 (Item als annotiert markieren)
- POST /active-learning/runs/{id}/push-labelstudio → schiebt die Queue mit Pre-Annotations nach LS

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- Top-N-Queue-Größe (`top_n`), Scorer (`double_pass` / `length_zscore`),
  optionales `sample_limit` — pro Lauf im Request

## Extension points (für Plugins / externe Nutzung)

- Unsicherheits-Scorer (`double_pass`, `length_zscore`) — erweiterbar
- Label-Studio-Push (siehe [labelstudio-import](labelstudio-import.md)) als Gegenrichtung

## Tests (müssen existieren und grün sein)

- `tests/test_phase11_active_learning.py` — Scorer-Verhalten (double_pass/length_zscore),
  Ranking nach Unsicherheit, End-to-End-Lauf, 422-Pfade, Mark-Annotated, Scorer-Crash → failed

## Known gaps

- Der `train → score → retrain`-Loop wird heute Schritt-für-Schritt vom Nutzer
  ausgelöst (Annotation fertig → neues Training); keine vollautomatische Mehr-Runden-Orchestrierung.

## Cross-references

- related_spec: [labelstudio-import](labelstudio-import.md) — Annotations-Quelle/-Ziel
- related_spec: [eval-framework](eval-framework.md) — Eval je Iteration, Inferenz-Backend
- related_spec: [multi-stage-pipelines](multi-stage-pipelines.md) — Retrain-Orchestrierung
- adr: ROADMAP.md — Phase 11 „Active-Learning-Schleife"
