---
feature: quantization
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-19
---

# Quantization-Pipeline — AWQ/GPTQ für günstige Inference

**Geplant (ROADMAP Phase 19) — noch nicht implementiert.**

Nach SFT/DPO automatisch quantisieren (AWQ/GPTQ), eval'en und promotbar machen
— für günstige Inference. Ziel: `POST /models/{id}/quantize?method=awq&bits=4`
erzeugt einen neuen Model-Eintrag mit quantisierter Variante plus Auto-Eval,
um den Quality-Loss zu messen.

## Capabilities (was der Nutzer tun kann)

- Eine registrierte Modellversion quantisieren (Methode + Bits wählen)
- Die quantisierte Variante als neuen Registry-Eintrag promotbar machen
- Den Quality-Loss gegen das Original messen (Auto-Eval mit derselben Suite, Δ)
- Die „Quantize"-Aktion im Model-Detail des UI auslösen

## Invariants (was immer gelten muss)

- Quantisierung läuft als Job; das Ergebnis ist ein **neuer** Registry-Eintrag
  (eigene Version), nicht eine Mutation des Originals
- Das Auto-Eval nutzt dieselbe Suite wie das Original → vergleichbares Δ-Tracking
- Adapter-/Modell-Pfad und Eval-Summary werden wie bei jeder Version eingefroren

## API surface (geplant — der angestrebte Vertrag)

- POST /models/{id}/quantize?method=awq&bits=4 → Job, erzeugt neue Modellversion

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- AWQ-/GPTQ-Backends (autoawq / gptqmodel) als optionale Dependencies

## Extension points (für Plugins / externe Nutzung)

- Quantisierungs-Backend (AWQ vs GPTQ) — austauschbar/erweiterbar

## Tests (müssen existieren und grün sein)

- (geplant) Quantize erzeugt neue Registry-Version + Auto-Eval-Δ
- (geplant) Backend-Auswahl (awq/gptq) + Bits-Parameter

## Known gaps

- Gesamtes Feature noch nicht gebaut: keine Quantize-Route, keine AWQ/GPTQ-
  Backends, keine Δ-Verknüpfung, keine UI-Aktion.

## Cross-references

- related_spec: [model-registry](model-registry.md) — quantisierte Variante = neue Version
- related_spec: [eval-framework](eval-framework.md) — Auto-Eval für Quality-Loss-Δ
- adr: ROADMAP.md — Phase 19 „Quantization-Pipeline"
- docs: https://github.com/casper-hansen/AutoAWQ
