---
feature: inference-playground
status: partial
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-8
---

# Inference-Probe / Playground — Prompts gegen ein Modell schicken

**Teilweise implementiert (ROADMAP Phase 8).** Single- und gestreamte Inferenz,
der N-Modell-Vergleich, der Modell-Cache (samt Introspektion) und die MCP-Tools
stehen; der explizite Abbruch eines laufenden Requests über `DELETE
/inferences/{id}` ist noch nicht gebaut.

Vor einer Production-Promotion will man selbst ein paar Prompts durchschicken
und sehen, wie das Modell antwortet — auch Base ↔ Fine-tuned im direkten
Vergleich. Ein Playground im UI plus eine `POST /inferences`-API mit gestreamter
Antwort.

## Capabilities (was der Nutzer tun kann)

- Einen Prompt an ein Modell schicken und die Antwort gestreamt sehen
- Ein Modell per Registry-Referenz (`name@alias`/`name/version`) oder Base-Modell wählen
- Zwei Versionen mit derselben Prompt side-by-side vergleichen
- Im UI-Tab „Playground" Modell-Dropdown (Registered Models + Base) bedienen
- Dieselbe Aktion via MCP (`inference`, `inference_compare`) aus einem Agenten

## Invariants (was immer gelten muss)

- Streaming-Antworten werden über SSE geliefert (konsistent mit dem Log-Stream)
- Modell-Referenzen werden über die Modell-Registry zu Adapter-Pfad + Basis aufgelöst
- Geladene Modelle liegen in einem LRU-Cache mit Obergrenze (max N gleichzeitig)
- Der Vergleichs-Modus schickt dieselbe Prompt parallel an beide Modelle

## API surface (der Vertrag für Clients)

- POST /inferences (model_ref, prompt, params) → 200 (vollständige Antwort)
- POST /inferences/stream (model_ref, prompt, params) → SSE-Stream
- POST /inferences/compare (model_refs[], prompt) → parallele Antworten
- GET /inferences/cache → 200 (welche Modelle aktuell geladen sind)
- DELETE /inferences/{id} → bricht einen laufenden Request ab — **geplant, noch nicht gebaut**

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- Modell-Cache-Größe (max gleichzeitig geladene Modelle) — konfigurierbar
- nutzt bestehende `TRAINPIPE_DATA_DIR`/`output_base_dir` für Adapter-Pfade

## Extension points (für Plugins / externe Nutzung)

- v1-Backend: `transformers` + `peft` (Base laden, Adapter aufsetzen, generieren)
- v2 (später): optionales vLLM/sglang-Backend, analog zum Eval-Inferenz-Backend

## Tests (müssen existieren und grün sein)

- `tests/test_inference.py` — Modell-Ref-Auflösung, Cache-Verhalten, Single-/
  Stream-/Compare-Pfade

## Known gaps

- `DELETE /inferences/{id}` (Abbruch eines laufenden Requests) ist noch nicht gebaut.
- Voraussetzung: [model-registry](model-registry.md) (für `name@alias`-Auflösung) — vorhanden.

## Cross-references

- related_spec: [model-registry](model-registry.md) — Quelle der Modell-Referenzen
- related_spec: [eval-framework](eval-framework.md) — teilt das Inferenz-Backend-Muster
- adr: ROADMAP.md — Phase 8 „Inference-Probe / Playground"
