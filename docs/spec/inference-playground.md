---
feature: inference-playground
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-8
---

# Inference-Probe / Playground — Prompts gegen ein Modell schicken

**Geplant (ROADMAP Phase 8) — noch nicht implementiert.**

Vor einer Production-Promotion will man selbst ein paar Prompts durchschicken
und sehen, wie das Modell antwortet — auch Base ↔ Fine-tuned im direkten
Vergleich. Diese Spec beschreibt den angestrebten Vertrag: ein Playground im
UI plus eine `POST /inferences`-API mit gestreamter Antwort.

## Capabilities (was der Nutzer tun kann)

- Einen Prompt an ein Modell schicken und die Antwort gestreamt sehen
- Ein Modell per Registry-Referenz (`name@alias`/`name/version`) oder Base-Modell wählen
- Zwei Versionen mit derselben Prompt side-by-side vergleichen
- Im UI-Tab „Playground" Modell-Dropdown (Registered Models + Base) bedienen
- Dieselbe Aktion via MCP (`inference`, `inference_compare`) aus einem Agenten

## Invariants (was immer gelten muss)

- Antworten werden über SSE gestreamt (konsistent mit dem Log-Stream)
- Ein laufender Inferenz-Request ist abbrechbar über eine separate
  `DELETE /inferences/{id}`-Operation (kein bidirektionaler Kanal)
- Modell-Referenzen werden über die Modell-Registry zu Adapter-Pfad + Basis aufgelöst
- Geladene Modelle liegen in einem LRU-Cache mit Obergrenze (max N gleichzeitig)
- Der Vergleichs-Modus schickt dieselbe Prompt parallel an beide Modelle

## API surface (geplant — der angestrebte Vertrag)

- POST /inferences (model_ref, prompt, params) → gestreamte SSE-Antwort
- POST /inferences/compare (model_refs[], prompt) → parallele Antworten
- DELETE /inferences/{id} → bricht einen laufenden Request ab

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- Modell-Cache-Größe (max gleichzeitig geladene Modelle) — konfigurierbar
- nutzt bestehende `TRAINPIPE_DATA_DIR`/`output_base_dir` für Adapter-Pfade

## Extension points (für Plugins / externe Nutzung)

- v1-Backend: `transformers` + `peft` (Base laden, Adapter aufsetzen, generieren)
- v2 (später): optionales vLLM/sglang-Backend, analog zum Eval-Inferenz-Backend

## Tests (müssen existieren und grün sein)

- (geplant) Modell-Cache-Eviction unter Last
- (geplant) Streaming-Chunk-Format / SSE-Terminierung

## Known gaps

- Gesamtes Feature noch nicht gebaut: keine `/inferences`-Routen, kein
  Playground-Tab, keine MCP-Tools, kein Modell-Cache.
- Voraussetzung: [model-registry](model-registry.md) (für `name@alias`-Auflösung) — vorhanden.

## Cross-references

- related_spec: [model-registry](model-registry.md) — Quelle der Modell-Referenzen
- related_spec: [eval-framework](eval-framework.md) — teilt das Inferenz-Backend-Muster
- adr: ROADMAP.md — Phase 8 „Inference-Probe / Playground"
