---
feature: multimodal-training
status: shipped
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-9
---

# Multimodal-Training — Doc-Extraktion mit Vision-LLMs

Dokument-Extraktion mit Vision-LLMs end-to-end: die ms-swift-Env-Verdrahtung
(SIZE_FACTOR/MAX_PIXELS via `MultimodalSettings`), der bild-haltige Bundle-Upload
(Zip mit Zip-Slip-/Symlink-Abwehr), das `images`/`videos`-Schema-Sniffing, die
Media-Auslieferung und die Doc-Metriken (Layout-IoU, strukturierte F1) sind gebaut.

Ziel: Dokument-Extraktion mit Qwen2-VL & Co. end-to-end — Upload eines
image-haltigen Datasets, Training, Inferenz, Eval. Ein Qwen2-VL-Fine-Tune auf
einem kleinen Doc-Set soll als promotbares Modell landen.

## Capabilities (was der Nutzer tun kann)

- Multimodale Settings (Bildgröße/Pixel-Budget) an einem Experiment setzen — **vorhanden**
- Ein bild-haltiges Dataset als Bundle (Zip) hochladen — **vorhanden**
- Ein VLM (z.B. Qwen2-VL) fine-tunen, mit gesetztem `model_type` — **vorhanden** (Durchreichung)
- Doc-Extraktion eval'en (Layout-IoU `bbox_iou`, strukturierte `structured_extraction_f1`) — **vorhanden**
- Bild-Thumbnails eines Bundles über die Media-Route abrufen — **vorhanden**

## Invariants (was immer gelten muss)

- Bild-Pfade sind relativ zum Dataset-Root und werden als Bundle mit hochgeladen
- Die Format-Erkennung erkennt `images`/`videos`/`audios`-Schema in JSONL (Stichprobe)
- `swift_builder` reicht `--model_type` und die multimodalen Env-Variablen
  (SIZE_FACTOR, MAX_PIXELS) durch
- Ein Bundle-Upload durchläuft dieselbe sha256-Dedup/Validierung wie normale Datasets
- Der Bundle-Upload wehrt Zip-Slip und Symlink-Einträge ab; die Media-Route
  blockiert Path-Traversal

## API surface (der Vertrag für Clients)

- POST /datasets/bundle → 201 (Zip-Upload eines bild-haltigen Datasets) · 422
  (text-only Bundle / kein JSONL / ungültiges Zip / Symlink- bzw. Zip-Slip-Eintrag)
- GET /datasets/{id}/media?path=… → 200 (liefert eine Bild-/Mediendatei aus dem
  Bundle) · 404 (Text-Dataset) · blockt Traversal

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `MultimodalSettings.size_factor` (default 8), `.max_pixels` (default 602112) — **vorhanden**
- diese werden zu den ms-swift-Env-Variablen SIZE_FACTOR / MAX_PIXELS — **vorhanden**

## Extension points (für Plugins / externe Nutzung)

- `training/dataset_formats.py` — `images`/`videos`/`audios`-Schema-Erkennung
- Eval-Metriken `bbox_iou`, `structured_extraction_f1` als Plugins unter
  `trainpipe/evals/metrics/`

## Tests (müssen existieren und grün sein)

- `tests/test_phase9_multimodal.py` — Schema-Erkennung (text/image/video, ungültige
  Media-Felder), Bundle-Upload + Abwehr (text-only, Symlink, Zip-Slip, kein Zip/JSONL),
  Media-Route (Thumbnail, Traversal-Block, 404 bei Text), `bbox_iou` + `structured_extraction_f1`

## Known gaps

- Ein echtes End-to-End mit einem Qwen2-VL-Sample-Set läuft nur auf einem GPU-Host
  (kein automatisierter CI-Test, da ohne GPU/Modellgewichte).
- Die UI bindet die Media-Route noch nicht als Thumbnail-Vorschau ein.

## Cross-references

- related_spec: [dataset-registry](dataset-registry.md) — Bundle erweitert den Upload
- related_spec: [training-experiments](training-experiments.md) — `MultimodalSettings` ist Teil der Spec
- related_spec: [eval-framework](eval-framework.md) — Heimat der Doc-Metriken
- adr: ROADMAP.md — Phase 9 „Multimodal-Verifizierung + Image-JSONL"
