---
feature: model-registry
status: shipped
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-7
---

# Modell-Registry & Promotion — benannte, versionierte Run-Ergebnisse

Ein abgeschlossenes Training liefert einen Adapter-Pfad und (seit dem
Eval-Framework) Eval-Scores. Die Registry gibt diesem Ergebnis einen
**Namen** (Modell-Familie, z.B. `invoice-extractor`), eine **Version**
(1-basiert, pro Familie hochgezählt) und beliebige **Aliase**
(`production`, `staging`, …), sodass nachgelagerte Schritte ein Modell
stabil über `name/version` oder `name/alias` referenzieren können statt
über eine kryptische Run-ID. Jede Version friert Adapter-Pfad, Basis-Modell
und einen Snapshot der letzten Eval-Scores ein.

## Capabilities (was der Nutzer tun kann)

- Ein abgeschlossenes Experiment als benanntes Modell registrieren — Version
  wird automatisch hochgezählt oder explizit angegeben
- Optional in einem Schritt direkt einen Alias (`staging`/`production`) setzen
- Alle Modelle auflisten, gefiltert nach Familie und/oder Alias
- Alle Versionen einer Familie abrufen
- Ein Modell über Version-Nummer **oder** Alias auflösen (eine Route, ein `ref`)
- Einen Alias innerhalb einer Familie auf eine andere Version umhängen oder entfernen
- Eine Modellversion löschen, geschützt solange sie noch einen Alias hält
- Dieselben Aktionen via MCP-Tool aus einem Agenten heraus auslösen
- Im UI-Tab „Models" Versionen + Aliase sehen, registrieren, umhängen

## Invariants (was immer gelten muss)

- Nur ein Experiment mit `status=completed` ist registrierbar; alles andere → 422
- `(name, version)` ist eindeutig; eine explizit doppelt vergebene Version → 409,
  eine ausgelassene Version wird zu `max(version)+1` der Familie
- Pro Familie zeigt ein Alias auf genau eine Version; Umhängen ist atomar
  (der alte Verweis wird im selben Schritt ersetzt)
- Adapter-Pfad, Basis-Modell und Eval-Summary werden bei der Registrierung
  aus dem Run übernommen und sind danach unveränderlich
- Die Eval-Summary ist das `{suite: {metric: mean}}`-Aggregat des je Suite
  **zuletzt** abgeschlossenen Eval-Runs des Experiments (leer, wenn keiner lief)
- Ein Alias kann nur auf eine Version derselben Familie zeigen (cross-family → 422)
- Löschen des Quell-Experiments kaskadiert auf seine Modelle; Löschen eines
  Modells kaskadiert auf dessen Aliase
- Eine Version, die noch einen Alias hält, lässt sich nur mit `?force=true` löschen
- Auflösen einer unbekannten Familie/Version/Alias ist ein 404, kein 500

## API surface (der Vertrag für Clients)

- POST /models → 201 (registriert ein `completed` Experiment als Familie+Version)
- POST /models → 422 (`unknown_experiment` / `experiment_not_completed`)
- POST /models → 409 (`version_exists`, bei explizit doppelter Version)
- GET /models → 200 (optional gefiltert nach `name`, `alias`)
- GET /models/{name} → 200 (alle Versionen)
- GET /models/{id}/datasets → 200 (Trainings-Lineage: welche Datasets diese
  Version sah — Details siehe [pii-redaction](pii-redaction.md))
- POST /models/{id}/quantize → 201 (erzeugt eine neue, quantisierte Version —
  Details siehe [quantization](quantization.md))
- GET /models/{name}/{ref} → 200 (`ref` = Versionsnummer oder Alias) · 404
- POST /models/{name}/aliases/{alias} → 200 (Body `{model_id}` oder `{version}`) · 404 · 422 (`missing_target`/`cross_family_alias`)
- DELETE /models/{name}/aliases/{alias} → 200 `{status}`
- DELETE /models/{id} → 200 `{deleted}` · 404 · 409 (`model_has_aliases`, ohne `?force=true`)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- (keine eigenen Schlüssel — nutzt `TRAINPIPE_DATA_DIR` für die SQLite-DB und
  `output_base_dir` zur Auflösung des Adapter-Pfads)

## Extension points (für Plugins / externe Nutzung)

- `register_model` (MCP) — registriert ein Experiment als Familie+Version, optional mit Alias
- `list_models` / `get_model` (MCP) — auflisten bzw. per `name`+`ref` auflösen
- `set_alias` (MCP) — Alias auf eine Version umhängen
- `delete_model` (MCP) — Version löschen (`force` für Alias-Halter)

## Tests (müssen existieren und grün sein)

- `tests/test_api_models.py` — Registrierung, Versions-Auto-Increment, `(name,version)`-Konflikt,
  Alias-Umhängen, Resolve über Version/Alias, Delete-Schutz bei gehaltenem Alias

## Known gaps

- Die Promotion-Regressions-Warnung (schlechterer Score als aktuelles
  `production`) ist **UI-seitig** und rein beratend — die API registriert und
  promotet bedingungslos, es gibt keinen Server-seitigen Block oder Warn-Code.
- Es gibt keine dedizierte „promote"-Route; Promotion = Alias `production` setzen.
- Das Projekt hat keine `spec(...)`-Marker-Konvention; Tests sind reine pytest-Dateien.
- ROADMAP.md Phase 7 zeigt die Items noch als `[ ]`, obwohl das Feature steht.

## Cross-references

- related_spec: [eval-framework](eval-framework.md) — Quelle der eingefrorenen Eval-Summary
- related_spec: [mcp-server](mcp-server.md) — Agenten-Tools für die Registry
- related_spec: [quantization](quantization.md) — `POST /models/{id}/quantize` erzeugt eine Version
- related_spec: [pii-redaction](pii-redaction.md) — `GET /models/{id}/datasets` (Lineage)
- adr: ROADMAP.md — Phase 7 „Modell-Registry & Promotion"
