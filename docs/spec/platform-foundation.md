---
feature: platform-foundation
status: shipped
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr:
---

# Plattform-Fundament — Auth, Persistenz, GPU-Pool, MLflow

Das Substrat, auf dem alle Features aufsetzen: API-Key-Authentifizierung,
eine migrations-versionierte SQLite-Datenbank, ein SQLite-gestützter GPU-Pool
mit Lease-Accounting, MLflow-Anbindung und der FastAPI-Lifespan, der Scheduler,
Study-Manager und Eval-Dispatcher startet/stoppt. Diese Spezifikation hält die
querschnittlichen Garantien fest, die nicht zu einem einzelnen Feature gehören.

## Capabilities (was der Nutzer/Betreiber tun kann)

- Die API auf jeder Maschine starten — auch ohne GPU/NVIDIA-Treiber
- Alle Datenpfade über eine einzige `TRAINPIPE_DATA_DIR` konfigurieren
- Den API-Key und die MLflow-URL über Env/`.env` setzen
- Health-Check (`/health`) und Frontend-Bootstrap (`/ui/config`) ohne Auth nutzen
- Nach einem Neustart laufen unterbrochene Jobs/Studies/Evals automatisch weiter

## Invariants (was immer gelten muss)

- Jeder Daten-Router verlangt einen gültigen `X-API-Key` (konstantzeit-Vergleich);
  fehlt/falsch → 401
- Genau drei Routen sind öffentlich: `GET /`, `GET /health`, `GET /ui/config`
- `/ui/config` gibt die MLflow-URL nur ohne eingebettete `user:pass`-Credentials aus
- Die DB ist append-only migrationsversioniert: `MIGRATIONS[i]` ist Version `i+1`,
  bereits angewandte Versionen werden nie editiert, nur neue angehängt
- SQLite läuft im WAL-Modus mit `synchronous=NORMAL` und `foreign_keys=ON`;
  Transaktionen werden kurz gehalten, pro Aufruf eine frische Connection
- Der GPU-Pool degradiert ohne Treiber/pynvml zu einem leeren Pool (API bootet,
  Jobs bleiben `queued`); Allocation läuft atomar unter einem asyncio-Lock
- Training und Eval teilen sich **denselben** GPU-Pool; `sync_leases` gibt
  verwaiste Leases frei, nimmt aber `running` Experimente **und** `running`
  Eval-Runs vom Orphan-Sweep aus
- Crash-Recovery-Reihenfolge: `running` requeuen **vor** `sync_leases`, sonst
  orphanen die Leases nach dem nächsten Status-Update
- MLflow wird überall lazy importiert (off the cold path) und ist best-effort:
  ein nicht erreichbarer Tracking-Server killt keinen Job

## API surface (der Vertrag für Clients)

- GET /health → 200 `{status: ok}` (öffentlich)
- GET /ui/config → 200 `{mlflow_tracking_uri}` (öffentlich, credential-frei)
- GET / → 200 (SPA, öffentlich)
- (alle übrigen Router hängen `require_api_key` als Router-Dependency)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `TRAINPIPE_API_KEY` (default `dev-key-change-me`) — API-Key
- `TRAINPIPE_DATA_DIR` (default `./data`) — Wurzel für `trainpipe.sqlite3`, `logs/`, `outputs/`, `datasets/`, `studies/`
- `TRAINPIPE_MLFLOW_TRACKING_URI` (default `http://localhost:5000`)
- `TRAINPIPE_HOST` / `TRAINPIPE_PORT` (default `0.0.0.0:8080`)
- `TRAINPIPE_VISIBLE_GPUS`, `TRAINPIPE_POLL_INTERVAL_SEC`, `TRAINPIPE_HEARTBEAT_INTERVAL_SEC`, `TRAINPIPE_MAX_DATASET_UPLOAD_BYTES`

## Extension points (für Plugins / externe Nutzung)

- `core/db.py` `MIGRATIONS` — neue Schema-Version = neuer Listeneintrag (nie alte ändern)
- `api/deps.py` — `get_db`/`get_scheduler`/`get_gpu_pool`/`get_study_manager`/`get_eval_dispatcher`
  als FastAPI-Dependencies; in Tests via `app.dependency_overrides` ersetzbar

## Tests (müssen existieren und grün sein)

- `tests/test_db.py` — Migrationsanwendung, Versions-Invariante
- `tests/test_api.py` — Auth (401), öffentliche Routen, Credential-Stripping in `/ui/config`
- `tests/test_repository.py`, `tests/test_schemas.py`

## Known gaps

- Single-Tenant: ein API-Key pro Instanz, keine Rollen/RBAC (bewusst out of scope).
- Kein verteilter GPU-Pool: GPUs eines einzelnen Hosts.
- Default-API-Key `dev-key-change-me` muss im Deployment überschrieben werden.

## Cross-references

- related_spec: alle Feature-Specs setzen auf Auth, DB und GPU-Pool dieser Grundlage auf
- docs: CLAUDE.md (Architektur-Notizen), ROADMAP.md (Out-of-Scope-Begründungen)
