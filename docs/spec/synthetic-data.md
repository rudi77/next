---
feature: synthetic-data
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-14
---

# Synthetic Data Generation — Trainings-Pairs aus einem Teacher-LLM

**Geplant (ROADMAP Phase 14) — noch nicht implementiert.**

Aus wenigen Beispielen viele Trainings-Pairs mit einem Teacher-LLM generieren
— z.B. aus 1000 Rechnungen + ihren JSONs 5000 augmentierte Varianten. Ziel:
ein `POST /synth` mit Provider + Instruction läuft als Job und schreibt das
Ergebnis als neues, registriertes Dataset mit Provenienz-Tags.

## Capabilities (was der Nutzer tun kann)

- Einen Synthese-Job starten (Provider, Modell, Quell-Dataset, Instruction, Zielanzahl, Seed)
- Das Ergebnis automatisch als neues Dataset registrieren lassen
- Die Provenienz nachvollziehen (welcher Teacher, welche Instruction)
- Die Synthese via MCP (`synth_dataset`) als Zwischenschritt selbst auslösen

## Invariants (was immer gelten muss)

- Der Job läuft als eigener trainpipe-Subprozess (kein ms-swift), via Anthropic-/OpenAI-SDK
- Outputs werden inkrementell in eine neue JSONL geschrieben
- Bei `completed` wird das Resultat automatisch als Dataset registriert, getaggt
  mit „source: synth from <X> via <Y>"
- Provenienz-Tags (Teacher-Modell, Instruction) hängen am erzeugten Dataset

## API surface (geplant — der angestrebte Vertrag)

- POST /synth (provider, model, source_dataset_id, instruction, target_count, seed) → Job

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- Provider-Credentials (Anthropic/OpenAI) über Env, analog zu `llm_as_judge`

## Extension points (für Plugins / externe Nutzung)

- `synth_dataset` (MCP) — Agenten-getriggerte Synthese
- Provider-Adapter (Anthropic/OpenAI) — erweiterbar um weitere Teacher-Backends

## Tests (müssen existieren und grün sein)

- (geplant) Job schreibt inkrementell und registriert bei Completion ein Dataset
- (geplant) Provenienz-Tags landen am Dataset

## Known gaps

- Gesamtes Feature noch nicht gebaut: keine `/synth`-Route, kein Synthese-Job,
  kein MCP-Tool, keine Provenienz-Tags.

## Cross-references

- related_spec: [dataset-registry](dataset-registry.md) — Registrierung + Tags des Ergebnisses
- related_spec: [eval-framework](eval-framework.md) — teilt die Provider-Env-Konvention (`llm_as_judge`)
- adr: ROADMAP.md — Phase 14 „Synthetic Data Generation"
