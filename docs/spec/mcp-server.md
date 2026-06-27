---
feature: mcp-server
status: shipped
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr:
---

# MCP-Server — trainpipe-Operationen als Agenten-Tools

Ein FastMCP-Server stellt dieselbe Oberfläche, die ein Mensch per `curl`
bedienen würde, als MCP-Tools bereit, damit ein Agent (z.B. Claude) Trainings,
Sweeps, Datasets und die Modell-Registry selbst steuern kann. Der Server ist
ein dünner Wrapper um die REST-API: er hält einen httpx-Client mit dem
`X-API-Key`-Header und reicht Tool-Aufrufe an die HTTP-Endpunkte durch.

## Capabilities (was der Nutzer/Agent tun kann)

- Experimente: einreichen, abrufen, auflisten, abbrechen, Log-Tail lesen
- Studies: starten, auflisten, abrufen, abbrechen
- GPUs: Pool-Status abfragen
- Datasets: hochladen (base64), auflisten, abrufen, Vorschau, löschen
- Modelle: registrieren, auflisten, auflösen, Alias umhängen, löschen
- Inferenz: einen Prompt schicken (`inference`) und zwei Modelle vergleichen (`inference_compare`)
- Evals: Suites anlegen/auflisten/abrufen/löschen, Runs starten/auflisten/abrufen, Per-Sample-Ergebnisse lesen, Runs abbrechen und vergleichen (`create_eval_suite`, `list_eval_suites`, `get_eval_suite`, `delete_eval_suite`, `run_eval`, `list_eval_runs`, `get_eval_run`, `get_eval_results`, `cancel_eval_run`, `compare_evals`) — schließt die train→eval→improve-Schleife für Agenten
- Synthetische Daten aus einem Teacher-LLM generieren (`synth_dataset`)
- Compliance: betroffene Datasets/Modelle für „Recht auf Löschung" scannen (`forget_scan`)
- Den Server standalone betreiben (`python -m trainpipe.mcp` / `trainpipe-mcp`)

## Invariants (was immer gelten muss)

- `import trainpipe.mcp` funktioniert **ohne** gesetzten `TRAINPIPE_API_KEY`:
  die FastMCP-Instanz wird beim Import gebaut, der httpx-Client erst beim
  ersten Tool-Aufruf (`_get_client`) — Tests können die Tool-Liste so introspizieren
- Fehlt beim ersten echten Aufruf der API-Key, beendet sich der Prozess mit
  klarer Meldung
- HTTP-Fehler werden mit erhaltenem Response-Body als `RuntimeError` weitergereicht,
  damit der Agent die umsetzbare Detailmeldung sieht
- Jeder Tool-Aufruf trägt den API-Key als `X-API-Key`-Header (nie als URL-Param)
- Die Tool-Oberfläche spiegelt die REST-Routen — der MCP-Server hält keinen
  eigenen Zustand und keine eigene DB

## API surface (der Vertrag für Clients)

- (kein eigener HTTP-Server — Tools sind MCP-Tools über stdio)
- Tools: `submit_experiment`, `get_experiment`, `list_experiments`, `cancel_experiment`,
  `tail_logs`, `submit_study`, `list_studies`, `get_study`, `cancel_study`, `gpu_status`,
  `upload_dataset`, `list_datasets`, `get_dataset`, `preview_dataset`, `delete_dataset`,
  `register_model`, `list_models`, `get_model`, `set_alias`, `delete_model`,
  `inference`, `inference_compare`, `synth_dataset`, `forget_scan`,
  `create_eval_suite`, `list_eval_suites`, `get_eval_suite`, `delete_eval_suite`,
  `run_eval`, `list_eval_runs`, `get_eval_run`, `get_eval_results`,
  `cancel_eval_run`, `compare_evals`

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `TRAINPIPE_API_KEY` — muss gesetzt sein, identisch zum API-Key der trainpipe-Instanz
- `TRAINPIPE_BASE_URL` (default `http://127.0.0.1:8080`) — Adresse der REST-API

## Extension points (für Plugins / externe Nutzung)

- Neue Tools = neue `@mcp.tool()`-Funktion, die einen REST-Endpunkt über
  `_get_client()` aufruft und mit `_unwrap()` das Ergebnis entpackt

## Tests (müssen existieren und grün sein)

- `tests/test_mcp_module.py` — Import ohne API-Key, Registrierung **aller** oben
  gelisteten Tools, jedes Tool hat eine Beschreibung, Client-Lazy-Init
  (übersprungen, wenn das `mcp`-Extra nicht installiert ist)

## Known gaps

- `upload_dataset` überträgt base64-kodiert; für große Dateien wird ein direkter
  `curl -F file=@…`-Upload empfohlen (base64-Overhead).
- Der MCP-Server validiert Specs nicht selbst — Fehler kommen als HTTP-Fehler der API zurück.

## Cross-references

- related_spec: [training-experiments](training-experiments.md), [hyperparameter-studies](hyperparameter-studies.md)
- related_spec: [dataset-registry](dataset-registry.md), [model-registry](model-registry.md)
- related_spec: [inference-playground](inference-playground.md) — `inference` / `inference_compare`
- related_spec: [synthetic-data](synthetic-data.md) — `synth_dataset`
- related_spec: [pii-redaction](pii-redaction.md) — `forget_scan`
- related_spec: [platform-foundation](platform-foundation.md) — Auth-Header, REST-Vertrag
