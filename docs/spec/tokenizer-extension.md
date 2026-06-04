---
feature: tokenizer-extension
status: partial
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-21
---

# Tokenizer-Erweiterung — Fachvokabular als zusätzliche Tokens

**Teilweise implementiert (ROADMAP Phase 21).** Das `extra_tokens`-Feld und das
Durchreichen an ms-swift (`--special_tokens` je Token, resize der Embedding-Layer)
sind gebaut; der Eval-Hook für den Tokenisierungs-Vergleich vorher/nachher fehlt noch.

Fachvokabular (Buchhaltungsbegriffe, interne Codes, Produkt-Codes) als
zusätzliche Tokens, damit das Modell sie nicht in 3–4 BPE-Stücke zerlegen muss.
Ziel: ein `extra_tokens`-Feld an der Spec, das ms-swift via
resize_token_embeddings durchreicht, plus ein Eval-Hook, der die Tokenisierung
vorher/nachher vergleicht.

## Capabilities (was der Nutzer tun kann)

- Eine Liste zusätzlicher Tokens an einem Experiment angeben — **vorhanden**
- Vor/Nach-Vergleich der Tokenisierung der Eval-Suite sehen — **geplant**

## Invariants (was immer gelten muss)

- `extra_tokens` wird je Token als `--special_tokens` an ms-swift weitergereicht
  (resize_token_embeddings); leere Liste emittiert nichts — **vorhanden**
- Die Liste ist auf 10000 Einträge begrenzt (Validierung am Spec-Eingang)
- Public-Feldnamen bleiben stabil; das Flag-Mapping bleibt im swift_builder isoliert

## API surface (der Vertrag für Clients)

- (keine neue Route — erweitert `POST /experiments` um `extra_tokens`)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `ExperimentSpec.extra_tokens: list[str]` (neu)

## Extension points (für Plugins / externe Nutzung)

- `training/swift_builder.py` — Weiterreichen von `extra_tokens` an ms-swift
- Eval-Hook für den Tokenisierungs-Vergleich (siehe [eval-framework](eval-framework.md)) — geplant

## Tests (müssen existieren und grün sein)

- `tests/test_phase21_tokens.py` — Default leer, ein `--special_tokens` je Token,
  Reihenfolge erhalten, kein Flag ohne Tokens, Zusammenspiel mit DPO/distributed, max_length-Grenze

## Known gaps

- Der Eval-Hook, der die Tokenisierung der Suite vor und nach der Erweiterung
  vergleicht, ist noch nicht gebaut (nur das Durchreichen an ms-swift steht).

## Cross-references

- related_spec: [training-experiments](training-experiments.md) — erweitert Spec + swift_builder
- related_spec: [eval-framework](eval-framework.md) — Heimat des Tokenisierungs-Hooks
- adr: ROADMAP.md — Phase 21 „Tokenizer-Erweiterung"
