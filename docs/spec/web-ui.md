---
feature: web-ui
status: shipped
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr:
---

# Web-UI — Single-Page-Oberfläche ohne Build-Schritt

Eine einzelne, selbst-enthaltene `trainpipe/ui/index.html` (Tailwind + Alpine
via CDN, kein Build-Schritt) bietet eine Tab-Oberfläche für alle Kern-Workflows.
Der API-Key wird einmal eingegeben und im Browser gehalten; alle Aufrufe gehen
gegen dieselbe REST-API wie MCP/`curl`. Beim Start zieht die Seite eine
sekret-freie Konfiguration (u.a. die MLflow-URL) von einem öffentlichen Endpunkt.

## Capabilities (was der Nutzer tun kann)

- Über Tabs navigieren: Experiments, Studies, Evals, Models, Datasets, GPUs
- Experimente einreichen, Detail/Logs ansehen, abbrechen
- Studies starten und Fortschritt verfolgen
- Eval-Suites/-Runs anlegen, Per-Sample-Resultate und Run-Vergleiche ansehen
- Modelle registrieren, Aliase setzen/umhängen, Versionen sehen
- Datasets hochladen, Vorschau ansehen, löschen
- GPU-Pool-Belegung sehen und von einer Lease zum Experiment springen
- Direkt zum zugehörigen MLflow-Run verlinken

## Invariants (was immer gelten muss)

- Der API-Key lebt in `localStorage` und reist ausschließlich als
  `X-API-Key`-Header — nie als URL-Parameter (hält ihn aus Server-Logs)
- Die Seite wird über `GET /` unauthentifiziert ausgeliefert; alle Daten-Calls
  brauchen den Key
- Die Bootstrap-Config kommt von `GET /ui/config` und enthält keine Geheimnisse
  (insb. ist die MLflow-URL um eingebettete Credentials bereinigt)
- Kein Build-Schritt, kein Bundler: die Datei läuft direkt im Browser
- Promotion-Warnung (`production` mit schlechterem/fehlendem Eval-Score) ist hier
  client-seitig implementiert — sie ist beratend, nicht serverseitig erzwungen

## API surface (der Vertrag für Clients)

- GET / → 200 (liefert die SPA als `text/html`, unauthentifiziert)
- (alle Daten-Operationen nutzen die Endpunkte der übrigen Specs)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `GET /ui/config` liefert `{mlflow_tracking_uri}` — abgeleitet aus
  `TRAINPIPE_MLFLOW_TRACKING_URI`, ohne eingebettete `user:pass`-Credentials

## Extension points (für Plugins / externe Nutzung)

- Ein neuer Tab = ein weiterer Eintrag in der Tab-Liste + eine `x-show`-Section
  in `index.html` (kein Build, direkt editierbar)

## Tests (müssen existieren und grün sein)

- `tests/test_ui.py` — `/` liefert die SPA, `/ui/config` ist credential-frei

## Known gaps

- Keine Browser-/E2E-Tests (nur Server-seitige Auslieferungs-Tests); UI-Logik ist
  inline in `index.html` und nicht unit-getestet.
- `index.fluent.html` ist eine zweite, untrackte UI-Variante im Repo (kein
  produktiver Pfad — `main.py` serviert ausschließlich `index.html`).

## Cross-references

- related_spec: [platform-foundation](platform-foundation.md) — `/ui/config`, Credential-Stripping, Auth
- related_spec: alle Feature-Specs liefern die Daten-Endpunkte der jeweiligen Tabs
