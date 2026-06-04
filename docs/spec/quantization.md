---
feature: quantization
status: shipped
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-19
---

# Quantization-Pipeline — AWQ/GPTQ für günstige Inference

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

- Quantisierung läuft über ein austauschbares Backend; das Ergebnis ist ein
  **neuer** Registry-Eintrag (eigene Version), nicht eine Mutation des Originals
- Die neue Version erbt die Eval-Summary des Eltern-Modells als Vergleichsbasis
- Adapter-/Modell-Pfad und Eval-Summary werden wie bei jeder Version eingefroren

## API surface (der Vertrag für Clients)

- POST /models/{id}/quantize?method=awq&bits=4 → 201 (erzeugt eine neue Modellversion) ·
  422 (unbekannte Methode / Eltern-Modell ohne Adapter) · 404 (unbekanntes Modell) ·
  500 (Backend-Fehler)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- AWQ-/GPTQ-Backends (autoawq / gptqmodel) als optionale Dependencies; ohne sie
  ist nur das Mock-Backend verfügbar

## Extension points (für Plugins / externe Nutzung)

- `quantization/runner.py` — Quantisierungs-Backend (AWQ vs GPTQ vs Mock) — austauschbar/erweiterbar

## Tests (müssen existieren und grün sein)

- `tests/test_phase19_quantize.py` — neue Version erzeugt, unbekannte Methode → 422,
  fehlendes Modell → 404, Eltern ohne Adapter → 422, Backend-Fehler → 500, Eval-Summary-Vererbung

## Known gaps

- Die quantisierte Version erbt die Eval-Summary des Originals; ein automatischer
  Re-Eval-Lauf mit Δ-Berechnung gegen dieselbe Suite ist noch nicht verdrahtet.
- Reale AWQ/GPTQ-Backends erfordern die optionalen Dependencies; ohne sie läuft nur das Mock-Backend.

## Cross-references

- related_spec: [model-registry](model-registry.md) — quantisierte Variante = neue Version
- related_spec: [eval-framework](eval-framework.md) — Auto-Eval für Quality-Loss-Δ
- adr: ROADMAP.md — Phase 19 „Quantization-Pipeline"
- docs: https://github.com/casper-hansen/AutoAWQ
