---
feature: tokenizer-extension
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-21
---

# Tokenizer-Erweiterung — Fachvokabular als zusätzliche Tokens

**Geplant (ROADMAP Phase 21) — noch nicht implementiert.**

Fachvokabular (Buchhaltungsbegriffe, interne Codes, Produkt-Codes) als
zusätzliche Tokens, damit das Modell sie nicht in 3–4 BPE-Stücke zerlegen muss.
Ziel: ein `extra_tokens`-Feld an der Spec, das ms-swift via
resize_token_embeddings durchreicht, plus ein Eval-Hook, der die Tokenisierung
vorher/nachher vergleicht.

## Capabilities (was der Nutzer tun kann)

- Eine Liste zusätzlicher Tokens an einem Experiment angeben
- Vor/Nach-Vergleich der Tokenisierung der Eval-Suite sehen

## Invariants (was immer gelten muss)

- `extra_tokens` wird an ms-swift weitergereicht (resize_token_embeddings)
- Public-Feldnamen bleiben stabil; das Flag-Mapping bleibt im swift_builder isoliert
- Der Eval-Hook vergleicht die Tokenisierung der Suite vor und nach Erweiterung

## API surface (der Vertrag für Clients)

- (keine neue Route — erweitert `POST /experiments` um `extra_tokens`)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `ExperimentSpec.extra_tokens: list[str]` (neu)

## Extension points (für Plugins / externe Nutzung)

- `training/swift_builder.py` — Weiterreichen von `extra_tokens` an ms-swift
- Eval-Hook für den Tokenisierungs-Vergleich (siehe [eval-framework](eval-framework.md))

## Tests (müssen existieren und grün sein)

- (geplant) swift_builder reicht `extra_tokens` korrekt an ms-swift weiter
- (geplant) Eval-Hook misst die Tokenisierungs-Differenz vorher/nachher

## Known gaps

- Gesamtes Feature noch nicht gebaut: kein `extra_tokens`-Feld, kein
  Durchreichen, kein Tokenisierungs-Eval-Hook.

## Cross-references

- related_spec: [training-experiments](training-experiments.md) — erweitert Spec + swift_builder
- related_spec: [eval-framework](eval-framework.md) — Heimat des Tokenisierungs-Hooks
- adr: ROADMAP.md — Phase 21 „Tokenizer-Erweiterung"
