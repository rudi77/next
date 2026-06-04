---
feature: labelstudio-import
status: shipped
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-10
---

# Annotation-Bridge — Direkter Import aus Label Studio

Label Studio existiert und ist gut; trainpipe baut keine eigene Annotations-UI,
sondern nur einen Import-Adapter. Ziel: ein Label-Studio-Projekt direkt holen,
seine Exports auf das passende JSONL-Format mappen und als Dataset registrieren
— ohne manuelle Format-Frickelei.

## Capabilities (was der Nutzer tun kann)

- Ein Label-Studio-Projekt per Id + Token importieren und als Dataset registrieren
- Aus den unterstützten Annotations-Typen automatisch das passende JSONL erzeugen:
  Text-NER, Doc-Layout, Conversation
- Inkrementell importieren (nur neue Annotationen seit dem letzten Lauf)
- Den Import aus dem Dataset-Upload-Modal heraus auslösen („Import from Label Studio")

## Invariants (was immer gelten muss)

- Der Mapper übersetzt Label-Studio-Annotations-Schemas in die hauseigenen
  JSONL-Formate; nicht abbildbare Schemas werden klar abgelehnt
- Das Ergebnis läuft durch die normale Dataset-Registrierung (sha256-Dedup,
  Format-Validierung gelten)
- Inkrementeller Import zieht nur Annotationen, die seit der letzten Marke neu sind

## API surface (der Vertrag für Clients)

- POST /datasets/from-labelstudio → 201 (registriert das Projekt als Dataset; Body
  mit `base_url`/`project_id`/`token`, optional `since_iso` für inkrementellen Import)
  · 422 (`no_records` — leeres Projekt/Filter ohne Treffer) · 502 (`labelstudio_error`)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- Label-Studio-Basis-URL + Token (pro Aufruf übergeben, nicht dauerhaft gespeichert)

## Extension points (für Plugins / externe Nutzung)

- Mapper pro Annotations-Typ (NER / Doc-Layout / Conversation) — erweiterbar um weitere Typen

## Tests (müssen existieren und grün sein)

- `tests/test_phase10_labelstudio.py` — Mapping je Annotations-Typ auf das
  erwartete JSONL, SSRF-Abwehr beim LS-Client, Import-Route registriert ein Dataset

## Known gaps

- Inkrementeller Import filtert server-seitig über `completed_at__gte` (`since_iso`);
  die „letzte Marke" muss der Aufrufer halten — trainpipe persistiert keinen Cursor.
- Bewusst kein Nachbau der Label-Studio-Annotations-UI (nur Import).

## Cross-references

- related_spec: [dataset-registry](dataset-registry.md) — registriert das Import-Ergebnis
- related_spec: [active-learning](active-learning.md) — gegenläufiger Pfad (Queue → LS)
- adr: ROADMAP.md — Phase 10 „Annotation-Bridge (Label Studio Import)"
