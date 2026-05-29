---
feature: multimodal-training
status: partial
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-9
---

# Multimodal-Training — Doc-Extraktion mit Vision-LLMs

**Teilweise implementiert (ROADMAP Phase 9).** Die ms-swift-Env-Verdrahtung
für VLMs (SIZE_FACTOR/MAX_PIXELS via `MultimodalSettings`) steht bereits; das
bild-haltige Dataset-Format, der Bundle-Upload und die Doc-Metriken fehlen noch.

Ziel: Dokument-Extraktion mit Qwen2-VL & Co. end-to-end — Upload eines
image-haltigen Datasets, Training, Inferenz, Eval. Ein Qwen2-VL-Fine-Tune auf
einem kleinen Doc-Set soll als promotbares Modell landen.

## Capabilities (was der Nutzer tun kann)

- Multimodale Settings (Bildgröße/Pixel-Budget) an einem Experiment setzen — **vorhanden**
- Ein bild-haltiges Dataset als Bundle hochladen (Zip / Multi-File) — geplant
- Ein VLM (z.B. Qwen2-VL) fine-tunen, mit korrekt gesetztem Modelltyp — geplant
- Doc-Extraktion eval'en (Layout-IoU, Feld-für-Feld-F1) — geplant
- Im Datasets-Tab eine Bild-Vorschau-Thumbnail sehen — geplant

## Invariants (was immer gelten muss)

- Bild-Pfade sind relativ zum Dataset-Root und werden als Bundle mit hochgeladen
- Die Format-Erkennung erkennt `images`/`videos`-Schema in JSONL
- `swift_builder` setzt `--model_type` für VLMs korrekt und reicht die
  multimodalen Env-Variablen (SIZE_FACTOR, MAX_PIXELS) durch — **vorhanden** für Env
- Ein Bundle-Upload durchläuft dieselbe sha256-Dedup/Validierung wie normale Datasets

## API surface (geplant — der angestrebte Vertrag)

- POST /datasets/bundle → Multi-File-/Zip-Upload eines bild-haltigen Datasets

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `MultimodalSettings.size_factor` (default 8), `.max_pixels` (default 602112) — **vorhanden**
- diese werden zu den ms-swift-Env-Variablen SIZE_FACTOR / MAX_PIXELS — **vorhanden**

## Extension points (für Plugins / externe Nutzung)

- `training/dataset_formats.py` — `images`/`videos`-Schema-Erkennung (zu ergänzen)
- neue Eval-Metriken `bounding_box_iou`, `structured_extraction_f1` als Plugins
  unter `trainpipe/evals/metrics/`

## Tests (müssen existieren und grün sein)

- (geplant) E2E mit minimalem Qwen2-VL-Sample-Set
- (geplant) Bundle-Upload + relative Bild-Pfad-Auflösung

## Known gaps

- `POST /datasets/bundle`, die Image-Schema-Erkennung, die VLM-`model_type`-
  Verifizierung und die Doc-Metriken sind noch nicht gebaut.
- Die UI hat keine Bild-Vorschau.

## Cross-references

- related_spec: [dataset-registry](dataset-registry.md) — Bundle erweitert den Upload
- related_spec: [training-experiments](training-experiments.md) — `MultimodalSettings` ist Teil der Spec
- related_spec: [eval-framework](eval-framework.md) — Heimat der Doc-Metriken
- adr: ROADMAP.md — Phase 9 „Multimodal-Verifizierung + Image-JSONL"
