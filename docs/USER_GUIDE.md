# trainpipe / next — Benutzerhandbuch

> **Stand:** 2026-06-27 · **Sprache der Doku:** Deutsch · **Sprache der API:** Englisch (Bezeichner unverändert)

Dieses Dokument erklärt **was** trainpipe ist, **wofür** du es einsetzt, **wie**
du jedes einzelne Feature benutzt — und wie ein **Agent** (Claude Code,
Claude Desktop, eigener MCP-Client, klassischer REST-Client) ein Training
steuert.

Die kurze Übersicht (Installation, Deployment-Patterns, Architektur-Diagramm)
findest du in der [README.md](../README.md). Die verhaltensgetriebenen
Verträge pro Subsystem stehen in [docs/spec/](spec/). Dieses Handbuch ist
die ausführliche Anleitung dazwischen.

---

## Inhalt

1. [Was ist trainpipe?](#1-was-ist-trainpipe)
2. [In 60 Sekunden: Die wichtigsten Konzepte](#2-in-60-sekunden-die-wichtigsten-konzepte)
3. [Server starten & Zugriff prüfen](#3-server-starten--zugriff-prüfen)
4. [Datasets — Daten verwalten](#4-datasets--daten-verwalten)
5. [Experiments — Ein Training einreichen](#5-experiments--ein-training-einreichen)
6. [Studies — Hyperparameter-Suche mit Optuna](#6-studies--hyperparameter-suche-mit-optuna)
7. [Evals — Modelle messen](#7-evals--modelle-messen)
8. [Models — Registry, Versionen, Aliase](#8-models--registry-versionen-aliase)
9. [Quantization — Modelle verkleinern](#9-quantization--modelle-verkleinern)
10. [Inference & Playground](#10-inference--playground)
11. [Pipelines — Mehrstufige Workflows](#11-pipelines--mehrstufige-workflows)
12. [Active Learning](#12-active-learning)
13. [Watches — Continuous Training](#13-watches--continuous-training)
14. [Synthetic Data](#14-synthetic-data)
15. [GPUs — Pool-Status](#15-gpus--pool-status)
16. [Compliance / GDPR](#16-compliance--gdpr)
17. [Web-UI — Tour](#17-web-ui--tour)
18. [Agent-Integration](#18-agent-integration)
19. [Konfiguration & Settings](#19-konfiguration--settings)
20. [Troubleshooting](#20-troubleshooting)
21. [Data Acquisition — Datensatz aus einem Auftrag](#21-data-acquisition--datensatz-aus-einem-auftrag)
22. [CLI — Der `trainpipe`-Terminal-Client](#22-cli--der-trainpipe-terminal-client)

---

## 1. Was ist trainpipe?

trainpipe (Codename **next**) ist eine **Trainings-Orchestrierung für einen
einzelnen Linux-Host mit 1–N NVIDIA-GPUs**. Du wirfst Jobs hinein (ms-swift):
**Supervised Fine-Tuning** (LoRA, QLoRA, Full-FT, LongLoRA, AdaLoRA, IA3),
**(Continued) Pretraining** und **Preference/RL-Training** (DPO, KTO, PPO,
GRPO). trainpipe verwaltet die Warteschlange, allokiert GPUs, streamt Logs
nach MLflow und speichert alles in SQLite. Eine REST-API ist die einzige
Eingangstür — sowohl für Menschen (über die Web-UI) als auch für Agenten
(über MCP oder den `trainpipe`-CLI-Client, siehe Kapitel 22).

### Wofür ist es gut?

- **Du hast eine GPU-Box** und willst nicht jedes Training zu Fuß starten
  und Logs aus dem Terminal kratzen.
- **Du willst Experimente vergleichen.** trainpipe taggt jeden Run mit
  Experiment-ID, Study-ID und Trial-Nummer, sodass die MLflow-UI
  zusammenhängende Runs sauber gruppiert.
- **Du willst, dass ein LLM-Agent eigenständig iteriert.** Submit, Cancel,
  Status, Logs, Hyperparameter-Search — alles ist über eine
  versionierte API erreichbar.
- **Du willst Datasets, Evals und Modelle als erstklassige Ressourcen
  pflegen**, mit Versionierung, Aliasen, Lineage und Audit-Trail.
- **Du hast noch gar keine Trainingsdaten.** Die agentische
  Data-Acquisition (Kapitel 21) baut dir aus einem Auftrag in
  natürlicher Sprache einen registrierten, PII-bereinigten Datensatz.
- **Du brauchst Compliance-Werkzeuge** — PII-Redaktion, GDPR-Scan,
  Lineage-Queries („welche Modelle haben auf diesen Daten trainiert?").

### Wofür ist es **nicht** gut?

- Kein Multi-Host-Cluster (geplant, siehe `docs/spec/distributed-training.md`).
- Kein Multi-Tenant-RBAC. Es gibt **einen** API-Key. Wer den hat, kann
  alles. Für Mehrbenutzer-Setups: mehrere trainpipe-Instanzen mit
  unterschiedlichen Keys.
- Keine eigene Annotations-UI. Stattdessen Label-Studio-Anbindung.
- Keine RAG-Infrastruktur. Trainpipe macht Training, nicht Retrieval.

### Architektur in einem Bild

```
Browser-UI  ──┐
              │  X-API-Key
Agent (MCP) ──┼──►  FastAPI (8080) ──►  Scheduler (asyncio)
              │         │                    │
curl / Code ──┘         ▼                    ▼
                    SQLite (WAL)        ms-swift Subprocesses
                    + Datasets/Logs          │
                                             ▼
                                       MLflow (5000)
                                       Metrics + Artefakte
```

Drei nebenläufige Async-Loops besitzen die Lebenszyklen:

1. **Scheduler-Loop** wählt aus der Queue, allokiert GPUs, spawnt den
   Subprocess.
2. **Per-Experiment-Monitor** wartet auf das Subprocess-Ende, finalisiert
   den MLflow-Run, gibt GPUs frei.
3. **Per-Study-Driver** (Optuna) iteriert: ask → enqueue → wait → tell.

---

## 2. In 60 Sekunden: Die wichtigsten Konzepte

### API-Key

Jeder Aufruf außer `GET /health`, `GET /ui/config` und `GET /` braucht
den Header `X-API-Key: <wert>`. Setze ihn auf dem Server in `.env` via
`TRAINPIPE_API_KEY=…`, in der UI per Browser-localStorage, im Client per
ENV-Variable.

### Dataset-Referenzen (`ds:`)

Wo immer du Trainingsdaten benennst — in `ExperimentSpec.dataset`,
`val_dataset`, `EvalSuite.dataset`, `SynthRequest.source_dataset` — sind
folgende Formen erlaubt:

| Form | Beispiel | Wann benutzen |
|---|---|---|
| HuggingFace-ID | `"AI-ModelScope/alpaca-gpt4-data-en"` | Public Datasets |
| ms-swift-Registry | `"alpaca-en"` | Vorgefertigte Sets |
| Lokaler Pfad | `"/srv/data/train.jsonl"` | Auf der Box liegend |
| Verzeichnis | `"/srv/data/chunks/"` | Mehrere Files |
| **`ds:<hex>`** | `"ds:abc123def456"` | Über `/datasets` hochgeladen |
| **Mit Sub-Sample** | `"ds:abc123#500"`, `"/path/train.jsonl#100"` | Nur erste N Records |

Die Auflösung von `ds:<id>` erfolgt **beim Submit**, nicht im Trainer —
ein falscher `ds:`-String führt zu 422 mit klarem `error`-Feld
(`malformed_dataset_ref` oder `unknown_dataset_ref`).

### Model-Referenzen

Bei Inference, Active Learning, Eval-Runs verwendest du Modell-Refs:

| Form | Beispiel | Bedeutung |
|---|---|---|
| `base:<id>` | `"base:Qwen/Qwen2.5-1.5B-Instruct"` | Basis-Modell ohne Adapter |
| `exp:<id>` | `"exp:exp-abc123…"` | Basis + Adapter eines Experiments |
| `<name>@<alias>` | `"invoice-extractor@production"` | Registriertes Modell, Alias |
| `<name>@<int>` | `"invoice-extractor@3"` | Registriertes Modell, Version |

### Status-Maschinen

- **Experiment**: `queued → running → completed | failed | cancelled`.
- **Study**: `running → completed | cancelled`.
- **EvalRun**: `queued → running → completed | failed | cancelled`.
- **ActiveLearningRun**: `queued → running → completed | failed | cancelled`.
- **Pipeline**: `running → completed | failed | cancelled`.

Cancel ist immer idempotent — wiederholtes Cancel auf einen schon
abgeschlossenen Job ist ein No-Op, kein Fehler.

### OpenAPI

Die komplette, immer aktuelle Spec liefert FastAPI selbst:

- `GET /openapi.json` — Schema (Maschinenformat)
- `GET /docs` — Swagger UI (interaktiv)
- `GET /redoc` — ReDoc (lesefreundliche Variante)

Wenn dieses Handbuch und die OpenAPI je auseinanderlaufen, hat die
OpenAPI recht.

---

## 3. Server starten & Zugriff prüfen

### Auf der GPU-Box (Linux/WSL)

```bash
./.scripts/next.sh install     # venv, pip install -e ".[training,dev]", .env
docker compose up -d           # MLflow auf :5000
# .env editieren: TRAINPIPE_API_KEY auf was Eigenes setzen
./.scripts/next.sh start       # uvicorn auf :8080, mit Health-Check
./.scripts/next.sh status      # PID + Health
./.scripts/next.sh logs -f     # Server-Log live folgen
./.scripts/next.sh stop        # graceful stop
```

Vom Windows-Hostsystem aus genau dieselben Befehle über WSL:

```powershell
wsl -d Ubuntu-24.04 -- /home/rudi/src/next/.scripts/next.sh status
```

### Health-Probe

```bash
curl http://localhost:8080/health
# → {"status":"ok"}

curl -H "X-API-Key: $TRAINPIPE_API_KEY" http://localhost:8080/gpus
# → {"total":4,"free":[1,2,3],"leases":[…]}
```

### Smoke-Tests

`.scripts/smoke_e2e.py` exerziert die ~20 wichtigsten Feature-Pfade
gegen einen laufenden Server. Bei Inbetriebnahme oder vor einem Deploy:

```bash
TRAINPIPE_BASE_URL=http://192.168.2.213:8080 \
TRAINPIPE_API_KEY=local-dev-secret \
python .scripts/smoke_e2e.py
```

Details: [.scripts/SMOKE.md](../.scripts/SMOKE.md).

---

## 4. Datasets — Daten verwalten

Datasets sind der **gemeinsame Nenner** für jeden anderen Endpunkt.
Alles, was du trainierst, evaluierst, mixt, redigierst oder mit AL
scort, läuft über ein Dataset-Objekt mit `id`, `version`, `sha256`,
`format`, `line_count` und einer Lineage-Spur zu den Eltern.

### 4.1 Upload (JSONL / Parquet)

`POST /datasets` (multipart) lädt eine einzelne Datei hoch.

```bash
curl -X POST http://localhost:8080/datasets \
  -H "X-API-Key: $TRAINPIPE_API_KEY" \
  -F "file=@./train.jsonl" \
  -F "name=invoices-v1" \
  -F "description=Original-Trainingsdaten"
```

Antwort (gekürzt):

```json
{
  "id": "abc123def456",
  "name": "invoices-v1",
  "format": "jsonl",
  "line_count": 4200,
  "size_bytes": 8412331,
  "sha256": "e2a…",
  "version": 1,
  "path": "/srv/data/datasets/abc123…/train.jsonl"
}
```

**Wichtig:**

- **Dedup über sha256.** Lädst du dieselbe Datei zweimal hoch, bekommst
  du **200** statt **201** und denselben Datensatz zurück — keine
  doppelte Speicherung.
- **Format wird validiert.** JSONL: erste 100 Records werden geparst;
  kaputte JSON-Zeilen sind ein 422.
- **Max-Größe**: standardmäßig 5 GB, einstellbar über
  `TRAINPIPE_MAX_DATASET_UPLOAD_BYTES`.

### 4.2 Listing, Detail, Preview, Löschen

```bash
# Alle Datasets
curl -H "X-API-Key: $K" http://localhost:8080/datasets

# Ein konkreter
curl -H "X-API-Key: $K" http://localhost:8080/datasets/abc123def456

# Erste 5 Zeilen ansehen (nur JSONL/Text)
curl -H "X-API-Key: $K" "http://localhost:8080/datasets/abc123/preview?n=5"

# Löschen
curl -X DELETE -H "X-API-Key: $K" \
  http://localhost:8080/datasets/abc123def456
# 409, wenn ein queued/running Experiment darauf referenziert
# → ?force=true erzwingt
```

### 4.3 Split — Train/Val deterministisch teilen

```bash
curl -X POST http://localhost:8080/datasets/abc123/split \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{"ratio":"90:10","seed":42,"train_name":"invoices-train","val_name":"invoices-val"}'
```

Antwort enthält zwei neue Datasets mit `version=2` und einer
Lineage-Spur zum Original. Reproduzierbar dank `seed`.

### 4.4 Mix — Gewichtete Kombination

```bash
curl -X POST http://localhost:8080/datasets/mixes \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "name": "mixed-3-to-1-tech-vs-general",
    "sources": [
      {"dataset_id":"abc123","weight":3.0},
      {"dataset_id":"def456","weight":1.0}
    ],
    "target_count": 10000,
    "seed": 0
  }'
```

Sampelt mit Replacement, alle Eltern landen in `dataset_lineage`.
Wesentlich für GDPR-Queries: trainpipe kann später beantworten, welche
Modelle indirekt aus `abc123` gelernt haben.

### 4.5 Redact — PII rausziehen

```bash
curl -X POST http://localhost:8080/datasets/abc123/redact \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{"entities":["email","phone","iban","credit_card","de_tax_id"]}'
```

Erzeugt ein **neues** Dataset (Original bleibt unangetastet → Audit).
Provenance enthält Trefferzahlen pro Entity-Typ.

### 4.6 Bundle — Bilder + JSONL als ZIP

Multimodale Trainings (Vision-LLM-Doc-Extraktion etc.) brauchen Bilder
neben dem JSONL-Manifest. `POST /datasets/bundle` nimmt ein ZIP mit:

- **genau einer** `.jsonl` als Manifest
- einem `images/`-Verzeichnis mit PNG/JPG/GIF/WebP/BMP/TIFF

```bash
zip -r invoice-bundle.zip manifest.jsonl images/
curl -X POST http://localhost:8080/datasets/bundle \
  -H "X-API-Key: $K" \
  -F "file=@./invoice-bundle.zip" \
  -F "name=invoices-with-images"
```

Validierungen: kein Zip-Slip, keine Symlinks, jedes referenzierte Bild
muss existieren.

Bilder ausliefern (z. B. für UI-Thumbnails):

```
GET /datasets/{id}/media?path=images/doc-001.png
```

### 4.7 Label-Studio-Import

```bash
curl -X POST http://localhost:8080/datasets/from-labelstudio \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "base_url": "https://label.example.com",
    "token":    "<ls-token>",
    "project_id": 17,
    "name": "ner-batch-2026-05",
    "import_kind": "text_ner",
    "since_iso": "2026-05-01T00:00:00Z",
    "max_tasks": 5000
  }'
```

trainpipe holt abgeschlossene Annotationen, leitet die Task-Shape
(`conversation` / `text_ner` / `doc_layout`) aus den ersten ~10 Tasks
ab und registriert das Ergebnis als JSONL-Dataset. Credentials werden
**nicht** in der Lineage gespeichert, nur ein nicht-geheimes Summary.

### 4.8 Lineage / GDPR-Query

> **Welche Modelle haben jemals (direkt oder indirekt) auf diesem
> Dataset trainiert?**

```bash
# Direkt
curl -H "X-API-Key: $K" \
  http://localhost:8080/datasets/abc123/models

# Inklusive aller Ableitungen (Splits, Mixes, Redacts)
curl -H "X-API-Key: $K" \
  "http://localhost:8080/datasets/abc123/models?recursive=true"
# → {"model_ids":["m-1","m-7","m-42"]}
```

Das ist die Grundlage für deinen GDPR-Forget-Workflow (siehe Kapitel 16).

---

## 5. Experiments — Ein Training einreichen

Ein **Experiment** ist genau **ein** Trainings-Subprocess von ms-swift.
Du beschreibst, was trainiert werden soll, trainpipe stellt es in die
Queue, allokiert GPUs, startet den Prozess, streamt Logs, finalisiert
den MLflow-Run.

### 5.1 Einreichen

```bash
curl -X POST http://localhost:8080/experiments \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "name": "qwen2-vl-lora",
    "model": "qwen/Qwen2-VL-2B-Instruct",
    "sft_type": "lora",
    "dataset": ["ds:abc123"],
    "val_dataset": ["ds:abc123#100"],
    "gpu_count": 1,
    "hyperparameters": {
      "num_train_epochs": 3,
      "learning_rate": 1e-4,
      "lora_rank": 8,
      "lora_alpha": 16,
      "max_length": 2048,
      "gradient_accumulation_steps": 4,
      "warmup_ratio": 0.03,
      "lr_scheduler_type": "cosine"
    },
    "tags": {"mlflow_experiment": "vlm-explore"}
  }'
# → {"experiment_id":"exp-1234…"}
```

Wichtige Felder:

| Feld | Typ | Bedeutung |
|---|---|---|
| `model` | str | HF-ID, ms-swift-Shortcut oder lokaler Pfad |
| `train_kind` | enum | `sft` (Default) / `pt` / `dpo` / `kto` / `ppo` / `grpo` — siehe 5.1.1 |
| `sft_type` | enum | `lora` / `qlora` / `full` / `longlora` / `adalora` / `ia3` |
| `dataset`, `val_dataset` | list[str] | Refs (siehe Kapitel 2); `dataset` braucht ≥ 1 Eintrag |
| `gpu_count` | int 1–8 | **Hardlimit 8** pro Job; größeres → 422 |
| `lora_target_modules` | list[str] | Standard sind ms-swift-Defaults |
| `hyperparameters` | dict | Standard-Trainings-Knöpfe (LR, Epochen, LoRA-Rank …) |
| `rlhf` | dict \| null | Preference/RL-Knöpfe — **nur** bei `train_kind ∈ {dpo,kto,ppo,grpo}` |
| `tags.mlflow_experiment` | str | Gruppen-Name in der MLflow-UI |
| `priority` | int | Höher = früher dran; Default 0 |
| `auto_eval` | list[str] | Suite-IDs, die nach Trainingsende automatisch laufen |

> **Tipp für Agenten:** Halte `priority` agent-submittierter Jobs auf
> 0 oder negativ, sodass ein Mensch jederzeit dazwischenfunken kann.

### 5.1.1 Trainingsart wählen: `train_kind`

`train_kind` entscheidet, welches ms-swift-Unterkommando trainpipe startet.
Default ist `sft` (Instruction-Tuning). `swift_builder` übersetzt das intern —
du fasst nie ms-swift-Flags an.

| `train_kind` | ms-swift | Wofür |
|---|---|---|
| `sft` | `swift sft` | Supervised Fine-Tuning (Default) |
| `pt` | `swift pt` | (Continued) Pretraining auf Rohtext |
| `dpo` | `swift rlhf --rlhf_type dpo` | Direct Preference Optimization |
| `kto` | `swift rlhf --rlhf_type kto` | KTO-Preference-Tuning |
| `ppo` | `swift rlhf --rlhf_type ppo` | PPO (braucht Reward-Model) |
| `grpo` | `swift rlhf --rlhf_type grpo` | Group Relative Policy Optimization |

**Continued Pretraining** (`train_kind: "pt"`) — Rohtext statt Chat-Format,
kein `rlhf`-Block:

```bash
curl -X POST http://localhost:8080/experiments \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B",
    "train_kind": "pt",
    "sft_type": "full",
    "dataset": ["ds:corpus01"],
    "gpu_count": 1
  }'
```

**RL / Preference** (`dpo`/`kto`/`ppo`/`grpo`) — der optionale `rlhf`-Block
trägt die RL-spezifischen Knöpfe. Beispiel GRPO mit eingebauten
Reward-Funktionen:

```bash
curl -X POST http://localhost:8080/experiments \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "train_kind": "grpo",
    "dataset": ["ds:rl-prompts"],
    "rlhf": {
      "reward_funcs": ["accuracy", "format"],
      "num_generations": 8,
      "beta": 0.04,
      "temperature": 0.9,
      "max_completion_length": 512
    }
  }'
```

`rlhf`-Felder (alle optional, außer wo unten gefordert):

| Feld | Gilt für | Bedeutung |
|---|---|---|
| `beta` ≥ 0 | dpo, kto, grpo | KL-Regularisierung |
| `reward_model` | ppo, grpo | Reward-Model-Pfad/-ID; **Pflicht bei ppo** |
| `reward_funcs` | **nur** grpo | Eingebaute Reward-Funktionen, z. B. `["accuracy","format"]` |
| `num_generations` ≥ 2 | **nur** grpo | Completions pro Prompt (Gruppengröße) |
| `max_completion_length` ≥ 1 | grpo, ppo | Rollout-Länge |
| `temperature` 0–2 | grpo, ppo | Sampling-Temperatur der Rollouts |

Die Validierung ist streng (sonst 422):
- `rlhf` darf **nur** bei einem RL-/Preference-`train_kind` gesetzt sein.
- `reward_funcs` / `num_generations` sind **GRPO-only**.
- `ppo` braucht `rlhf.reward_model`; `grpo` braucht `reward_model` **oder**
  `reward_funcs`.

### 5.2 Batch-Submit (atomar)

```bash
curl -X POST http://localhost:8080/experiments/batch \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{"specs":[<spec1>, <spec2>, <spec3>]}'
```

Entweder **alle** Specs valide → alle queued, oder **eine** ist kaputt
→ keine. Saubere Semantik für Agenten, die nicht mit Teilausfällen
hantieren wollen.

### 5.3 Status & Logs

```bash
# Listen
curl -H "X-API-Key: $K" \
  "http://localhost:8080/experiments?status=running&limit=20"

# Ein Detail
curl -H "X-API-Key: $K" \
  http://localhost:8080/experiments/exp-1234

# Vollständiges Log
curl -H "X-API-Key: $K" \
  http://localhost:8080/experiments/exp-1234/logs > train.log

# Live-Tail (Server-Sent-Events)
curl -N -H "X-API-Key: $K" \
  http://localhost:8080/experiments/exp-1234/logs/stream
```

Das Detail enthält u. a. `status`, `gpu_ids`, `mlflow_run_id`,
`mlflow_experiment_id`, `created_at`, `started_at`, `finished_at` und
das eingereichte `spec`.

### 5.4 Cancel

```bash
curl -X POST -H "X-API-Key: $K" \
  http://localhost:8080/experiments/exp-1234/cancel
# → {"status":"cancelling"}      (oder "already_done", "not_found")
```

Idempotent. Sendet SIGTERM an die Prozessgruppe; nach 10 s SIGKILL.
GPU-Lease wird im Monitor freigegeben.

---

## 6. Studies — Hyperparameter-Suche mit Optuna

Eine **Study** ist ein Optuna-Driver, der wiederholt Specs aus einem
Suchraum sampelt, sie als Experimente in dieselbe Queue stellt und das
Zielmetrik-Ergebnis aus MLflow ausliest.

### 6.1 Sweep starten

```bash
curl -X POST http://localhost:8080/studies \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "name": "lr-rank-sweep-v2",
    "base_spec": {
      "model":  "Qwen/Qwen2.5-1.5B-Instruct",
      "dataset": ["ds:abc123"],
      "sft_type": "lora",
      "hyperparameters": {"num_train_epochs": 2}
    },
    "search_space": {
      "hyperparameters.learning_rate": {"kind":"loguniform","low":1e-5,"high":1e-3},
      "hyperparameters.lora_rank":     {"kind":"categorical","choices":[4,8,16,32]},
      "hyperparameters.warmup_ratio":  {"kind":"uniform","low":0.0,"high":0.1}
    },
    "target_metric": "eval/loss",
    "direction": "minimize",
    "n_trials": 16,
    "max_concurrent": 4,
    "sampler": "tpe"
  }'
# → {"study_id":"study-…"}
```

`search_space` benutzt **dotted-path-Override** — der Key wird auf das
`base_spec` angewendet, bevor das Trial eingereicht wird. Damit kannst
du beliebig tiefe Spec-Felder variieren.

### 6.2 Fortschritt verfolgen

```bash
# Liste
curl -H "X-API-Key: $K" http://localhost:8080/studies

# Detail (mit best_value, best_trial_id)
curl -H "X-API-Key: $K" http://localhost:8080/studies/study-abc

# Trials einer Study (die sind ganz normale Experimente)
curl -H "X-API-Key: $K" \
  "http://localhost:8080/experiments?study_id=study-abc"

# Cancel — laufende Trials werden gekillt, der Driver hört auf zu samplen
curl -X POST -H "X-API-Key: $K" \
  http://localhost:8080/studies/study-abc/cancel
```

### 6.3 Cost-Summary (für Plots)

```bash
curl -H "X-API-Key: $K" http://localhost:8080/studies/cost-summary
# → [{"study_id":"…","n_trials":16,"total_gpu_seconds":3600.5,
#     "peak_vram_mb":8192.0,"total_energy_wh":45.2,
#     "best_value":0.42,"target_metric":"eval/loss","direction":"minimize"},…]
```

Damit füttert die UI ihren Cost-vs-Quality-Plot.

---

## 7. Evals — Modelle messen

Evals sind **Suite + Run**:

- Eine **Suite** ist ein wiederverwendbarer Test: Dataset + Metriken +
  Inference-Parameter.
- Ein **Run** ist eine konkrete Ausführung einer Suite gegen ein Modell.
- Ein **Compare** ist eine N-Wege-Δ-Analyse mehrerer Runs derselben Suite.

### 7.1 Suite anlegen

```bash
curl -X POST http://localhost:8080/evals/suites \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "name": "invoice-extract-v1",
    "description": "Field extraction over invoices",
    "dataset": "ds:eval-set-abc",
    "metrics": [
      {"metric_name":"em","kind":"exact_match","config":{}},
      {"metric_name":"bleu","kind":"bleu","config":{"n_gram_weights":[0.25,0.25,0.25,0.25]}}
    ],
    "inference_params": {"temperature":0.0,"max_new_tokens":256}
  }'
```

Suite-Name ist global eindeutig — Konflikt → 409. Metrik-Konfiguration
wird beim Anlegen validiert.

### 7.2 Run starten

```bash
curl -X POST http://localhost:8080/evals/runs \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "suite_id":     "suite-abc",
    "experiment_id":"exp-1234",
    "triggered_by": "manual"
  }'
# → {"id":"er-…","status":"queued",…}
```

Der Run holt sich das Modell aus dem `experiment_id` (Basis + Adapter),
spielt die Suite-Dataset-Zeilen durch, scort jede Vorhersage.

### 7.3 Aggregate & Per-Sample-Ergebnisse

```bash
curl -H "X-API-Key: $K" http://localhost:8080/evals/runs/er-… 
# → "aggregate":{"em":{"mean":0.78,"std":0.41,"sample_count":500},…}

# Per-Sample (paginiert)
curl -H "X-API-Key: $K" \
  "http://localhost:8080/evals/runs/er-…/results?limit=100&offset=0"
```

### 7.4 Compare — Regressionen finden

```bash
curl -H "X-API-Key: $K" \
  "http://localhost:8080/evals/compare?run_ids=er-1,er-2,er-3"
```

Liefert pro Metrik die Means der drei Runs **und** eine Liste von
Samples, in denen mindestens ein Run schlechter ist als die anderen —
der direkte Weg zu „diese 14 Inputs sind nach dem Re-Training kaputt".

---

## 8. Models — Registry, Versionen, Aliase

Die Model-Registry trennt **Trainings-Artefakt** (Experiment-Output, ein
Adapter unter `data/outputs/<exp-id>/`) von der **gepflegten Identität**
(„invoice-extractor@production"). Modelle sind versioniert, Aliase
zeigen wie Pointer auf eine Version.

### 8.1 Registrieren

```bash
curl -X POST http://localhost:8080/models \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "exp-1234",
    "name": "invoice-extractor",
    "alias": "staging",
    "description": "First LoRA on cleaned dataset v3"
  }'
# → {"id":"mod-…","name":"invoice-extractor","version":1,
#     "adapter_path":"/srv/data/outputs/exp-1234","aliases":["staging"]}
```

- `experiment_id` muss `completed` sein, sonst 422.
- `version` wird automatisch hochgezählt, kann auch explizit gesetzt
  werden.
- `alias` (optional) wird im selben Call gesetzt.

### 8.2 Auflösen

```bash
# Alle Versionen einer Familie
curl -H "X-API-Key: $K" \
  http://localhost:8080/models/invoice-extractor

# Per Version
curl -H "X-API-Key: $K" \
  http://localhost:8080/models/invoice-extractor/2

# Per Alias
curl -H "X-API-Key: $K" \
  http://localhost:8080/models/invoice-extractor/production
```

Der Server unterscheidet numerisch (→ Version) vs. nicht-numerisch
(→ Alias) am Suffix.

### 8.3 Aliase verwalten

```bash
# Alias umsetzen (atomar)
curl -X POST http://localhost:8080/models/invoice-extractor/aliases/production \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{"version": 3}'

# Alias löschen
curl -X DELETE -H "X-API-Key: $K" \
  http://localhost:8080/models/invoice-extractor/aliases/staging
```

Aliase über Familien-Grenzen hinweg sind nicht erlaubt → 422.

### 8.4 Dataset-Lineage des Modells

```bash
curl -H "X-API-Key: $K" \
  http://localhost:8080/models/mod-abc/datasets
# → [{"id":"…","name":"invoices-v1","version":1,"line_count":4200,…},…]
```

Pendant zu Kapitel 4.8: dort fragst du das Dataset, hier das Modell.

### 8.5 Löschen

```bash
curl -X DELETE -H "X-API-Key: $K" \
  http://localhost:8080/models/mod-abc
# 409, wenn noch Aliase auf diese Version zeigen → ?force=true erzwingt
```

---

## 9. Quantization — Modelle verkleinern

```bash
curl -X POST http://localhost:8080/models/mod-abc/quantize \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{"method":"awq","bits":4,"description":"AWQ-4bit, eval-Δ < 0.5%"}'
```

Resultat ist eine **neue Version** in derselben Familie. Eltern-Modell
muss einen `adapter_path` haben. `method` ∈ `awq`/`gptq`, `bits` ∈ 2–16
(praxisrelevant: 4, 8). Läuft im Hintergrund-Thread; der Response
kommt zurück, sobald der Job gestartet ist.

> **Hinweis:** Auf einer Box ohne CUDA-Build oder ohne `awq`/`gptq`
> liefert der Endpunkt einen strukturierten Fehler-Envelope statt
> 500 — gut für agentisches Retry/Fallback.

---

## 10. Inference & Playground

### 10.1 Synchrones Predict

```bash
curl -X POST http://localhost:8080/inferences \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "model_ref": "invoice-extractor@production",
    "prompt": "Extract fields from the following invoice…",
    "params": {"max_new_tokens":512,"temperature":0.0,"top_p":0.95}
  }'
# → {"model_ref":"…","base_model":"Qwen/…","adapter_path":"/…","prediction":"…"}
```

`model_ref` folgt der Syntax aus Kapitel 2 (`base:`, `exp:`, `name@alias`,
`name@<int>`).

### 10.2 Streaming

```bash
curl -N -X POST http://localhost:8080/inferences/stream \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{"model_ref":"exp:exp-1234","prompt":"…"}'
```

Liefert SSE-Events: `token` (Chunks à ~64 Zeichen), `done`, `error`.

### 10.3 N-Wege-Vergleich (Playground)

```bash
curl -X POST http://localhost:8080/inferences/compare \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "model_refs": [
      "base:Qwen/Qwen2.5-1.5B-Instruct",
      "invoice-extractor@2",
      "invoice-extractor@production"
    ],
    "prompt": "…",
    "params": {"max_new_tokens":256,"temperature":0.0}
  }'
```

Identischer Prompt gegen bis zu 8 Refs — perfekt, um zu prüfen, was
das letzte Re-Training tatsächlich gebracht hat.

### 10.4 Cache-Diagnose

```bash
curl -H "X-API-Key: $K" http://localhost:8080/inferences/cache
# → {"max_loaded":3,"loaded":[{"base_model":"…","adapter_path":"/…"},…]}
```

LRU mit konfigurierbarer Maximalzahl (Settings). Schlägt ein Predict
auf einem geladenen Backend fehl, wird das Backend invalidiert.

---

## 11. Pipelines — Mehrstufige Workflows

Eine **Pipeline** ist eine geordnete Liste von Experiment-Specs.
Klassisches Muster: CPT → SFT → DPO. Outputs einer Stufe können in
die nächste fließen (`ds:<exp-id-der-vorigen-stufe>` als
Dataset-Referenz).

```bash
curl -X POST http://localhost:8080/pipelines \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "name": "cpt-then-sft",
    "concurrency": 1,
    "stages": [
      {"name":"cpt","model":"…","dataset":["ds:raw-corpus"],"sft_type":"full",…},
      {"name":"sft","model":"…","dataset":["ds:instruct-set"],"sft_type":"lora",…}
    ]
  }'
```

- Stufen laufen sequenziell, bis zu `concurrency` parallel.
- Fehlschlag einer Stufe → Pipeline `failed`, kein automatischer Retry.
- Cancel kapt alle aktiven Stufen.

Status & Cancel analog zu Experimenten.

---

## 12. Active Learning

Active Learning identifiziert die **N unsichersten Samples** in einem
ungelabelten Pool, schiebt sie in eine Annotation-Queue (UI oder Label
Studio), nimmt die Labels und füttert die nächste Trainings-Runde.

### 12.1 Run starten

```bash
curl -X POST http://localhost:8080/active-learning/runs \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "dataset":      "ds:unlabeled-pool",
    "model_ref":    "invoice-extractor@production",
    "top_n":        100,
    "sample_limit": 5000,
    "scorer":       "entropy"
  }'
```

`scorer` ∈ `entropy` / `margin` / `least_confidence` / `custom`. Der
Aufruf läuft synchron, das Scoring im Worker-Thread (event-loop bleibt
frei).

### 12.2 Queue lesen / abhaken

```bash
curl -H "X-API-Key: $K" \
  "http://localhost:8080/active-learning/runs/al-…/queue?only_unannotated=true"

curl -X POST -H "X-API-Key: $K" \
  "http://localhost:8080/active-learning/runs/al-…/queue/<item_id>/annotated"
```

Items haben `uncertainty` (0–1, höher = unsicherer), die Prediction und
das Original-Input.

### 12.3 An Label Studio pushen

```bash
curl -X POST http://localhost:8080/active-learning/runs/al-…/push-labelstudio \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{"base_url":"https://label.example.com","token":"<ls-token>","project_id":17}'
# → {"pushed": 87}
```

Jeder unannotierte Queue-Eintrag wird zu einer LS-Task mit der
Model-Prediction als Pre-Annotation. Per-Item-Fehler sind nicht fatal.

---

## 13. Watches — Continuous Training

Ein **Watch** ist eine Regel: „Wenn X, dann starte folgende Pipeline."

Zwei Trigger-Arten:

| `kind` | Auslöser | Felder |
|---|---|---|
| `interval` | alle N Sekunden | `interval_seconds` |
| `metric_threshold` | Eval-Run einer Suite verletzt Schwelle | `suite_id`, `metric_name`, `threshold` |

```bash
curl -X POST http://localhost:8080/watches \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "name": "retrain-when-accuracy-drops",
    "kind": "metric_threshold",
    "suite_id": "suite-abc",
    "metric_name": "exact_match",
    "threshold": 0.75,
    "pipeline_config": {…vollständige Pipeline-Config…}
  }'
```

Standardmäßig **disabled** angelegt. Erst nach `POST /watches/{id}/enable`
wird der Watch scharfgestellt. Symmetrisch `disable`, `delete`.

---

## 14. Synthetic Data

Erweitert ein Quell-Dataset mit Hilfe eines Teacher-LLMs (Anthropic,
OpenAI oder Mock).

```bash
curl -X POST http://localhost:8080/synth \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "model":    "claude-3-5-sonnet",
    "source_dataset": "ds:seed-examples",
    "instruction":    "Generate paraphrases of the user query that preserve intent.",
    "target_count": 2000,
    "seed": 42,
    "max_tokens": 1024,
    "name": "paraphrased-seeds"
  }'
```

- **Provider `mock`** erzeugt deterministisch synthetische Zeilen ohne
  echte API-Calls — gut für Smoke-Tests und Pipelines, in denen ein
  echter Provider zu teuer wäre.
- **Provider `anthropic` / `openai`** brauchen die jeweiligen API-Keys
  als ENV (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
- Antwort ist ein registriertes Dataset; SHA256-Dedup greift auch hier.
- Provenance enthält `_source`-Markierungen pro Zeile.

---

## 15. GPUs — Pool-Status

```bash
curl -H "X-API-Key: $K" http://localhost:8080/gpus
# → {
#     "total": 4,
#     "free":  [2,3],
#     "leases":[
#       {"index":0,"experiment_id":"exp-abc"},
#       {"index":1,"experiment_id":"exp-def"},
#       {"index":2,"experiment_id":null},
#       {"index":3,"experiment_id":null}
#     ]
#   }
```

`detect_gpus()` degradiert sauber: ohne `pynvml`/Driver → leere Liste,
trainpipe startet trotzdem. Tests laufen mit leerem Pool.

---

## 16. Compliance / GDPR

trainpipe trackt Dataset-Lineage explizit, damit du beantworten kannst,
**wer von welcher Datenzeile gelernt hat**. Das Werkzeug dafür heißt
**forget**.

### 16.1 CLI: `trainpipe-forget`

```bash
# Substring-Scan, case-insensitive
trainpipe-forget jane.doe@example.com

# Regex (anschnallen)
trainpipe-forget --regex 'AT[0-9]{18}' --case-sensitive

# JSON-Report rausschreiben
trainpipe-forget --output report.json "credit-card"
```

Exit-Codes:

- `0` — keine Treffer
- `1` — Treffer (Shell-/CI-Friendly für „fail wenn drin")
- `2` — Argumentfehler

Reportstruktur:

```json
{
  "term": "jane.doe@example.com",
  "is_regex": false,
  "case_sensitive": false,
  "scanned_datasets": 42,
  "hits": [
    {"dataset_id":"…","dataset_name":"customer-emails-v3",
     "hit_count":3,"sample_line_numbers":[5,17,42]}
  ],
  "impacted_models": [
    {"name":"email-classifier","version":1,"model_id":"m-…",
     "via_dataset_ids":["…","…"]}
  ],
  "skipped_datasets": []
}
```

### 16.2 MCP-Tool: `forget_scan(term, regex=False, case_sensitive=False)`

Identischer Mechanismus, aber direkt im Agenten-Workflow nutzbar.
Liest die SQLite direkt — kein REST-Roundtrip, schnell auf großen
Beständen.

### 16.3 Workflow „Forget User X"

1. `forget_scan("jane.doe@example.com")` — finde alle Hits & impacted Models.
2. Für jedes betroffene Dataset: `POST /datasets/{id}/redact` mit
   passendem `entities`-Set → neues, gesäubertes Dataset.
3. Trigger ein Re-Training der impacted Models gegen das gesäuberte
   Dataset (am ehesten als Pipeline oder Watch).
4. Wenn das neue Modell durch deine Evals kommt, Alias `production`
   umsetzen, altes Modell löschen.

---

## 17. Web-UI — Tour

trainpipe liefert zwei UI-Varianten, beide single-file, ohne Build-Schritt:

| Datei | Tech-Stack | Wofür |
|---|---|---|
| `trainpipe/ui/index.html` | Tailwind + Alpine | Standard, alle Features |
| `trainpipe/ui/index.fluent.html` | Fluent UI Web Components + Alpine | Optisch poliert, Subset |

### Tabs (in `index.html`)

| Tab | Was du tust |
|---|---|
| **Experiments** | Submit-Formular, Tabelle mit Status, Live-Log-Tail, MLflow-Link, Cancel |
| **Studies** | Sweep-Submit, Fortschritts-Tabelle, Cost-Plot |
| **Pipelines** | DAG-Visualisierung, Stage-Dependencies |
| **Evals** | Suite-Editor, Run-Trigger, N-Wege-Compare |
| **Models** | Family-Browser, Versionen, Alias-Promotion |
| **Playground** | Prompt + Single/Compare-Inference |
| **Active Learning** | Run-Liste, Queue-Annotation, AL-Iteration-Dashboard |
| **Watches** | Liste, Enable/Disable, Delete |
| **Datasets** | Drag-and-Drop-Upload, Preview, `ds:<id>` zum Kopieren, Trained-On-View |
| **GPUs** | Karten pro Device mit Lease-State |

### Erstkontakt

1. UI öffnen: `http://<server>:8080/`
2. Modal fragt nach API-Key → in localStorage gespeichert
3. Polling refresht alle 4 s; Log-Detail alle 2.5 s
4. MLflow-Link unter jedem Experiment springt direkt zum Run (URLs
   werden Host-bewusst neu geschrieben, damit das auch aus dem Tailnet
   oder vom Handy funktioniert)

---

## 18. Agent-Integration

Es gibt drei Möglichkeiten, einen Agenten an trainpipe anzubinden — von
„brutal direkt" bis „strukturiert und sicher".

### 18.1 Variante A — REST + Bash

Jeder Agent, der `curl` ausführen kann, ist sofort einsatzbereit. In
Claude Code lohnt sich eine kurze Permissions-Allowlist, damit nicht
jeder Call eine Bestätigung braucht:

```jsonc
// .claude/settings.json
{
  "permissions": {
    "allow": [
      "Bash(curl http://localhost:8080/*)",
      "Bash(curl http://localhost:5000/*)"
    ]
  }
}
```

Dann ist eine Aufforderung wie *„lade `./train.jsonl` hoch, starte einen
LoRA-Job, beobachte die Logs, melde dich wenn er durch ist"* direkt
ausführbar.

**Vorteil:** maximale Flexibilität, kein zusätzlicher Setup.
**Nachteil:** der API-Key landet im Kontext, der Agent muss
JSON-Body-Konstruktion selbst hinbekommen.

### 18.2 Variante B — MCP Server (empfohlen)

Das MCP-Layer versteckt den Key, exponiert jede Operation als typisiertes
Tool und gibt dem Agenten saubere Schemata.

#### Installation

```bash
pip install -e ".[mcp]"
```

#### Registrierung in Claude Code

```bash
claude mcp add trainpipe -- env \
  TRAINPIPE_API_KEY=$TRAINPIPE_API_KEY \
  TRAINPIPE_BASE_URL=http://localhost:8080 \
  python -m trainpipe.mcp
```

Für einen Remote-Server (Tailscale):

```bash
claude mcp add trainpipe -- env \
  TRAINPIPE_API_KEY=$TRAINPIPE_API_KEY \
  TRAINPIPE_BASE_URL=https://trainpipe.<tailnet>.ts.net:8443 \
  python -m trainpipe.mcp
```

#### Konfiguration in Claude Desktop

`%APPDATA%\Claude\claude_desktop_config.json` (Windows) bzw.
`~/Library/Application Support/Claude/claude_desktop_config.json` (Mac):

```json
{
  "mcpServers": {
    "trainpipe": {
      "command": "python",
      "args": ["-m", "trainpipe.mcp"],
      "env": {
        "TRAINPIPE_API_KEY":   "…",
        "TRAINPIPE_BASE_URL":  "http://localhost:8080"
      }
    }
  }
}
```

#### Transport & Lifecycle

- **stdio** — JSON-RPC 2.0 über stdin/stdout, stderr ist Log-Kanal.
- **Lazy init** — Import ohne `TRAINPIPE_API_KEY` ist OK; der erste
  Tool-Call schlägt fehl, wenn der Key fehlt.
- **Timeouts** — `httpx` connect 5 s, read 30 s; Fehler werden als
  `RuntimeError("HTTP <code>: <detail>")` an den Agenten zurückgegeben.

#### Tool-Übersicht (40 Stück)

| Bereich | Tools |
|---|---|
| Experiments | `submit_experiment`, `get_experiment`, `list_experiments`, `cancel_experiment`, `tail_logs` |
| Studies | `submit_study`, `list_studies`, `get_study`, `cancel_study` |
| Datasets | `upload_dataset`, `list_datasets`, `get_dataset`, `preview_dataset`, `delete_dataset`, `synth_dataset` |
| Models | `register_model`, `list_models`, `get_model`, `set_alias`, `delete_model` |
| Inference | `inference`, `inference_compare` |
| Evals | `create_eval_suite`, `list_eval_suites`, `get_eval_suite`, `delete_eval_suite`, `run_eval`, `list_eval_runs`, `get_eval_run`, `get_eval_results`, `cancel_eval_run`, `compare_evals` |
| Acquisition | `start_acquisition`, `get_acquisition`, `get_acquisition_sources`, `list_acquisitions`, `answer_acquisition`, `cancel_acquisition` |
| GPU | `gpu_status` |
| Compliance | `forget_scan` |

Datasets per MCP zu uploaden bedeutet `content_b64` — Base64-Encoded.
Für Dateien > ~10 MB lieber direkt per `curl -F file=@…` an die REST-API
gehen.

> Die Eval-Tools schließen die Agenten-Schleife **train → eval → improve**:
> der Agent trainiert, misst gegen eine Suite, vergleicht Runs und
> entscheidet selbst über den nächsten Schritt. Die Acquisition-Tools
> setzen noch eine Stufe davor an — **Daten beschaffen → train** (Kapitel 21).

### 18.3 Variante C — Autoresearch-Loop

Die spannende Variante: der Agent submittet nicht *einen* Job, sondern
*iteriert*.

**Pattern 1 — Optuna-getrieben** (wenn Suchraum klar ist):

```python
# Pseudo, läuft sowohl per curl als auch per MCP
study_id = mcp.submit_study(config={
    "name": "lr-sweep",
    "base_spec": {"model": "Qwen/Qwen2.5-1.5B-Instruct",
                  "dataset": ["ds:abc"], "sft_type": "lora"},
    "search_space": {
        "hyperparameters.learning_rate": {"kind":"loguniform","low":1e-5,"high":1e-3}
    },
    "target_metric": "eval/loss", "direction": "minimize",
    "n_trials": 16, "max_concurrent": 4, "sampler": "tpe"
})["study_id"]

while True:
    s = mcp.get_study(study_id)
    if s["status"] != "running": break
    time.sleep(30)

best_trial_id = s["best_trial_id"]
```

**Pattern 2 — LLM-in-the-loop** (wenn „was als nächstes" eigentliche
Reasoning braucht):

```python
goal = "eval/loss < 1.0 on Qwen2.5-1.5B + my dataset"
history = []
for i in range(8):
    spec = propose_next_spec(history, goal)     # ← LLM-Call
    eid = mcp.submit_experiment(spec)["experiment_id"]

    while True:
        rec = mcp.get_experiment(eid)
        if rec["status"] in ("completed","failed","cancelled"): break
        time.sleep(15)

    # Metric aus MLflow lesen (Tags helfen beim Filtern)
    metric = read_mlflow_metric(rec["mlflow_run_id"], "eval/loss")
    history.append({"spec":spec, "metric":metric, "status":rec["status"]})
    if metric is not None and metric < 1.0: break
```

#### Was der Agent variieren kann (billig, nur Spec-Felder)

- `learning_rate`, `lora_rank`, `lora_alpha`, `lora_dropout`,
  `warmup_ratio`, `weight_decay`, `lr_scheduler_type`
- `max_length`, `gradient_accumulation_steps` (Memory-gebunden)
- `sft_type` (`lora` ↔ `qlora` wenn VRAM knapp; `full` wenn passt)
- `lora_target_modules`
- Dataset-Mix, `#N`-Sub-Sampling, Splits

#### Was der Agent **nicht** aus einem Spec ändern kann

- Modell-Architektur, Tokenizer
- Loss-Funktion, Trainer-Hooks
- Hardware-Topologie

### 18.4 End-to-End-Walkthrough (MCP)

Konkret, Schritt für Schritt:

```text
1.  upload_dataset(name="invoices-v1",
                   filename="train.jsonl",
                   content_b64="<base64>")
    → {id:"ds_abc123", line_count:4200, sha256:"…"}

2.  submit_experiment(spec={
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset": ["ds:ds_abc123"],
        "sft_type": "lora",
        "hyperparameters": {"num_train_epochs":2,"learning_rate":2e-4},
        "tags": {"mlflow_experiment":"invoice-experiments"}
    })
    → {experiment_id:"exp_xyz789"}

3.  loop: get_experiment("exp_xyz789")
    → status: queued / running / completed
    + tail_logs("exp_xyz789", n_lines=80)  zur Fortschritts-Anzeige

4.  Wenn status=="completed":
    register_model(name="invoice-extractor",
                   experiment_id="exp_xyz789",
                   alias="staging")
    → {id:"mod_v1", name:"invoice-extractor", version:1, aliases:["staging"]}

5.  Eval anstoßen (per REST, MCP exponiert das nicht direkt):
    POST /evals/runs {suite_id:"suite_abc", experiment_id:"exp_xyz789"}
    → poll bis status=="completed", lies aggregate

6.  Wenn die Eval-Aggregate gut sind:
    set_alias(name="invoice-extractor", alias="production", version=1)

7.  inference(model_ref="invoice-extractor@production",
              prompt="Extract fields from …",
              max_new_tokens=512)
    → {prediction:"…"}
```

**Häufige Fehler in diesem Flow:**

- Step 2: `422 unknown_dataset_ref` — `ds:`-ID stimmt nicht. Lösung:
  `list_datasets()` nochmal abrufen.
- Step 2: `422 less_than_equal` — `gpu_count > 8`. Hardlimit, ändert
  nichts dran, dass die Box vielleicht weniger GPUs hat.
- Step 4: `422` — Experiment ist nicht `completed`. Vorher prüfen.
- Step 5: Run-Status `failed` → `error`-Feld lesen, oft ist der
  Adapter-Pfad nicht erreichbar.

### 18.5 Guardrails für unbeaufsichtigte Agenten

- **Budget**: `max_experiments` + Wallclock-Cap.
- **Abandon-Ship-Regel**: Stop nach K Trials ohne Improvement.
- **Niedrige `priority`** auf agent-submittierten Jobs.
- **Emergency Cancel** als immer-erreichbarer Tool-Call.
- **Eigene trainpipe-Instanz oder Scoped-Key**, damit ein Agent nicht
  versehentlich Produktions-Runs killt.

---

## 19. Konfiguration & Settings

Alle Settings werden via Pydantic aus ENV-Variablen mit Präfix
`TRAINPIPE_` (und/oder `.env`) gelesen.

| Variable | Default | Zweck |
|---|---|---|
| `TRAINPIPE_API_KEY` | `dev-key-change-me` | **Pflicht in Produktion.** Einziger Auth-Faktor. |
| `TRAINPIPE_HOST` | `0.0.0.0` | Bind-Adresse |
| `TRAINPIPE_PORT` | `8080` | API-Port |
| `TRAINPIPE_DATA_DIR` | `./data` | Wurzel für SQLite, Logs, Datasets, Outputs |
| `TRAINPIPE_MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow-Server (Credentials erlaubt, werden in `/ui/config` rausgefiltert) |
| `TRAINPIPE_VISIBLE_GPUS` | unset | JSON-Liste `[0,1,2]` zum Einschränken; default: alle sichtbaren |
| `TRAINPIPE_POLL_INTERVAL_SEC` | `1.0` | Scheduler-Tick |
| `TRAINPIPE_HEARTBEAT_INTERVAL_SEC` | `5.0` | Reserviert |
| `TRAINPIPE_MAX_DATASET_UPLOAD_BYTES` | `5 GB` | Upload-Limit (UI-Form); cURL/MCP haben kein eigenes Limit |

Abgeleitete Pfade (rechnen sich automatisch aus `data_dir`):

- `sqlite_path` → `{data_dir}/trainpipe.sqlite3`
- `logs_dir` → `{data_dir}/logs/`
- `output_base_dir` → `{data_dir}/outputs/`
- `datasets_dir` → `{data_dir}/datasets/`

### `.env`-Beispiel für eine private Box

```bash
TRAINPIPE_API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
TRAINPIPE_HOST=127.0.0.1
TRAINPIPE_PORT=8080
TRAINPIPE_DATA_DIR=/srv/trainpipe
TRAINPIPE_MLFLOW_TRACKING_URI=http://localhost:5000
TRAINPIPE_VISIBLE_GPUS=[0,1,2,3]
```

### Installation-Extras

| Extra | Inhalt | Wann |
|---|---|---|
| `training` | `ms-swift>=3.0.3` | Auf der GPU-Box, sonst kein Training |
| `mcp` | `mcp>=1.0` | Wenn du `trainpipe-mcp` brauchst |
| `dev` | `pytest`, `ruff`, `mypy` | Lokale Entwicklung, Tests, Lint |

Typische Kombinationen:

- **GPU-Box**: `pip install -e ".[training,dev]"`
- **Agent-Client (Laptop)**: `pip install -e ".[mcp]"`

---

## 20. Troubleshooting

### „cannot reach <url>"

Server läuft nicht oder LAN-IP unerreichbar. Diagnose vom Windows-Host
aus:

```powershell
wsl -d Ubuntu-24.04 -- /home/rudi/src/next/.scripts/next.sh status
```

### „401 Unauthorized"

API-Key passt nicht. Prüfe Server-`.env`:

```powershell
wsl -d Ubuntu-24.04 -- grep TRAINPIPE_API_KEY /home/rudi/src/next/.env
```

Stimmt das mit dem überein, was du im `X-API-Key` schickst?

### „422 missing_local_paths"

Du hast einen lokalen Pfad in `dataset` angegeben, der auf der Box nicht
existiert. Server liefert die komplette Liste — alle in einem
Roundtrip korrigierbar.

### „422 less_than_equal · gpu_count"

`gpu_count` > 8 ist hart begrenzt (Pydantic-Validator). Größere Werte
werden nicht akzeptiert — selbst, wenn die Box weniger GPUs hat. Für
gewollten Submit-Time-Fehler im Smoke-Test (Scheduler-FAIL bei
Übergröße) muss man heutzutage einen anderen Trigger nehmen, z. B.
einen ungültigen Modellnamen.

### Experiment hängt in `queued`

Wahrscheinlich kein GPU frei. `GET /gpus` schauen, ggf. andere
Experimente canceln. Achte auch auf `priority` — höhere Werte gehen
vor.

### s06 / s13 im Smoke-Test instabil

`s13 active-learning` lädt ein echtes Modell (Qwen2.5-0.5B-Instruct).
Auf einer 4-GB-Karte mit parallelen Jobs kann das OOMen. Re-Run wenn
die Box still ist, oder per `--only` ohne `s13` laufen lassen.

### Stale `smoke-*`-Ressourcen

Pre-Flight des Smoke-Scripts cleant ältere Reste automatisch (>1 h).
Manuell:

```bash
curl -s -H "X-API-Key: $K" "$URL/datasets" | \
  jq -r '.[] | select(.name|startswith("smoke-")) | .id' | \
  xargs -I{} curl -s -X DELETE -H "X-API-Key: $K" "$URL/datasets/{}?force=true"
```

---

## 21. Data Acquisition — Datensatz aus einem Auftrag

Die übrigen Kapitel setzen voraus, dass du **schon** Trainingsdaten hast.
Die agentische **Data Acquisition** dreht das um: du gibst einen Auftrag in
natürlicher Sprache (*„Trainingsdaten für einen Buchhaltungs-LLM für den
DACH-Raum"*), und trainpipe baut daraus einen **registrierten,
PII-bereinigten Datensatz** — am Ende bekommst du eine ganz normale
`ds:<id>`-Referenz, die du sofort in einem Experiment (Kapitel 5) benutzt.

### 21.1 Die Phasen eines Acquisition-Runs

Ein Run läuft durch eine feste Phasenmaschine:

```
intake → research → acquire → synthesize → curate → register
```

| Phase | Was passiert |
|---|---|
| **intake** | Der Auftrag (`brief`) wird in einen strukturierten `AcquisitionSpec` (Domäne, Fähigkeiten, Format) übersetzt. Bleiben Rückfragen offen, parkt der Run in `awaiting_input`. |
| **research** | Nur wenn `search_provider != "none"`: Web-Suchanfragen → Kandidaten-URLs → Lizenz-/robots.txt-Prüfung → Quellen-Ledger. |
| **acquire** | Erlaubte Quellen abrufen, per LLM zu belegten Records destillieren. |
| **synthesize** | Restliche Records aus dem Spec generieren, bis `target_count` erreicht ist. |
| **curate** | **Pflicht-PII-Redaktion** (rekursiv) + exakte Dedup nach (prompt, completion). |
| **register** | JSONL in die Dataset-Registry schreiben (sha256-Dedup), `dataset_id` zurückgeben. |

Status-Werte: `queued` → `running` → (`awaiting_input` ↔) →
`completed` / `failed` / `cancelled`.

### 21.2 Run starten

```bash
curl -X POST http://localhost:8080/acquisitions \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{
    "name": "dach-accounting-sft",
    "brief": "Trainingsdaten für einen Buchhaltungs-LLM für den DACH-Raum",
    "provider": "anthropic",
    "model": "claude-opus-4-8",
    "target_count": 500,
    "search_provider": "tavily",
    "max_sources": 10,
    "strict_license": true,
    "max_llm_calls": 2000
  }'
# → AcquisitionRun (status "queued")
```

Felder von `AcquisitionRequest`:

| Feld | Typ | Default | Bedeutung |
|---|---|---|---|
| `name` | str | — | Name des entstehenden Datensatzes (Pflicht) |
| `brief` | str | — | Auftrag in natürlicher Sprache (Pflicht) |
| `provider` | enum | `mock` | Teacher-LLM: `anthropic` / `openai` / `mock` (offline) |
| `model` | str | `mock` | Modell-ID des Teacher-LLM |
| `target_count` | int 1–100000 | 50 | Ziel-Datensatzgröße |
| `search_provider` | enum | `none` | `none` (nur synthetisieren, kein Netz) / `mock` / `tavily` |
| `max_sources` | int 0–50 | 5 | Obergrenze entdeckter Quellen-URLs |
| `strict_license` | bool | `false` | **Guardrail:** Quellen mit unklarer Lizenz überspringen |
| `max_llm_calls` | int | 0 | **Kostenbudget:** Cap auf Teacher-LLM-Calls (0 = unbegrenzt) |
| `spec` | obj \| null | null | Vorausgefüllter `AcquisitionSpec` — überspringt `intake` |

### 21.3 Rückfragen beantworten (Human-in-the-Loop)

Wenn `intake` offene Fragen hat und keine Antworten vorliegen, parkt der Run
in `awaiting_input`. Die Fragen stehen in `spec.open_questions`. Du
beantwortest sie, der Driver läuft weiter:

```bash
# Aktuellen Stand inkl. open_questions ansehen
curl -H "X-API-Key: $K" http://localhost:8080/acquisitions/<run_id>

# Antworten (Keys = Fragetext)
curl -X PATCH http://localhost:8080/acquisitions/<run_id>/answers \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -d '{"answers": {"Welche Sprachen?": "de-DE, de-AT, de-CH"}}'
# 409, wenn der Run nicht in awaiting_input steht
```

Ein **Agent** kann das Parken ganz vermeiden, indem er einen vorausgefüllten
`spec` (mit beantworteten `open_questions`) direkt an `start_acquisition`
übergibt.

### 21.4 Fortschritt, Quellen, Cancel

```bash
# Detail: phase, raw_count, final_count, redaction-Trefferzahlen, dataset_id
curl -H "X-API-Key: $K" http://localhost:8080/acquisitions/<run_id>

# Geprüfte Web-Quellen (Audit-Trail; inkl. abgelehnter)
curl -H "X-API-Key: $K" http://localhost:8080/acquisitions/<run_id>/sources

# Liste (Filter optional)
curl -H "X-API-Key: $K" "http://localhost:8080/acquisitions?status=running"

# Abbrechen
curl -X POST -H "X-API-Key: $K" \
  http://localhost:8080/acquisitions/<run_id>/cancel
```

Ist der Run `completed`, trägt `dataset_id` die fertige `ds:<id>`-Referenz —
der Datensatz steht mit Provenienz (Provider, Modell, Domäne, Run-ID in der
Beschreibung) in der Registry und ist sofort trainierbar.

### 21.5 Guardrails

- **Pflicht-Redaktion:** Die `curate`-Phase redigiert PII **immer** rekursiv,
  bevor irgendetwas registriert wird. Die Trefferzahlen stehen im Feld
  `redaction` des Runs.
- **Kostenbudget:** `max_llm_calls` deckelt die Teacher-LLM-Aufrufe. Ist das
  Budget aufgebraucht, werden keine neuen Calls mehr gestartet (der Run
  bricht nicht ab, er liefert nur weniger Records).
- **Lizenz-Gate:** `strict_license=true` lässt nur Quellen mit bestätigt
  offener Lizenz zu (Wikipedia/Wikimedia, `.gov`, `.europa.eu`,
  Creative Commons, Project Gutenberg …). Jede Quelle wird vor dem Abruf
  zusätzlich gegen robots.txt und einen SSRF-Schutz geprüft.

> Sechs MCP-Tools spiegeln das (`start_acquisition`, `get_acquisition`,
> `get_acquisition_sources`, `list_acquisitions`, `answer_acquisition`,
> `cancel_acquisition`) — siehe Kapitel 18. Eine Acquisition lässt sich
> auch als erste Stufe einer Pipeline (Kapitel 11) verketten.

---

## 22. CLI — Der `trainpipe`-Terminal-Client

`trainpipe` ist **zweierlei in einem Binary**: ohne Subcommand (oder mit
`serve`) startet es den FastAPI-Server; mit einem Subcommand wird es zum
**operativen Client** über dieselbe REST-API, die auch der MCP-Server nutzt.
Damit fährst du die volle Schleife train → eval → improve im Terminal, ohne
`curl`-Bodies von Hand zu bauen.

### 22.1 Konfiguration

Der Client liest zwei Umgebungsvariablen:

| Variable | Default | Bedeutung |
|---|---|---|
| `TRAINPIPE_API_KEY` | — | **Pflicht** für jeden Subcommand; geht als `X-API-Key` raus |
| `TRAINPIPE_BASE_URL` | `http://127.0.0.1:8080` | Ziel-Server |

Jeder Befehl gibt **JSON auf stdout** aus (Ausnahme: `logs` gibt Klartext) —
ideal zum Durchpipen in `jq`. Fehlt der Key, kommt `MissingAPIKey`.

### 22.2 Befehlsübersicht

```bash
# Server
trainpipe                         # = trainpipe serve  → uvicorn auf :8080

# Experiments
trainpipe submit --model Qwen/Qwen2.5-0.5B --dataset ds:ab12 \
                 --train-kind sft --sft-type lora --gpu-count 1
trainpipe submit --spec @spec.json          # ganzer ExperimentSpec aus Datei/@- (stdin)
trainpipe experiments --status running --limit 20
trainpipe get <exp-id>
trainpipe logs <exp-id> -n 50               # -n 0 = alles
trainpipe cancel <exp-id>

# Datasets
trainpipe datasets
trainpipe upload my-set ./train.jsonl --description "..."

# Models & Inference
trainpipe models --name my-model
trainpipe register-model --name my-model --experiment <exp-id> --alias staging
trainpipe set-alias my-model prod 3
trainpipe inference my-model@staging "Summarize: ..." --max-new-tokens 256

# Studies & GPUs
trainpipe studies
trainpipe gpus

# Evals
trainpipe eval-suites
trainpipe create-suite @suite.json
trainpipe run-eval --suite <suite-id> --experiment <exp-id>
trainpipe eval-runs --experiment <exp-id> --status completed
trainpipe eval-results <run-id>
trainpipe compare-evals <run-a> <run-b>

# Generischer Escape-Hatch: jeder Endpoint, jede Methode
trainpipe api GET  /datasets/<id>/models
trainpipe api POST /acquisitions --json @acq.json
```

`--spec`, `create-suite`s Spec-Argument und `api --json` akzeptieren alle
denselben Eingabestil: Inline-JSON, `@datei.json` oder `@-` für stdin.

> Acquisitions, Pipelines, Watches und Active Learning haben (noch) keine
> dedizierten Subcommands — dafür ist `trainpipe api <METHOD> <PATH>` da, das
> die komplette REST-API abdeckt.

---

## Weiterführend

- **Spezifikationen pro Feature**: [`docs/spec/`](spec/) — verhaltens-erste,
  versionierte Verträge. Für API-Vertragsfragen die maßgebliche Quelle.
- **README**: [`README.md`](../README.md) — Architektur, Deployment-Patterns
  (Tailscale, Caddy), Projektstruktur.
- **Smoke-Tests**: [`.scripts/SMOKE.md`](../.scripts/SMOKE.md) — was die
  20 Sections prüfen, wie der Bericht aussieht.
- **Architektur-Detail**: [`CLAUDE.md`](../CLAUDE.md) — die Stellen, an
  denen mehrere Dateien zusammenspielen (Scheduler-Order, Crash-Recovery,
  swift_builder-Mapping).

Wenn etwas im Handbuch und in der OpenAPI auseinanderläuft: OpenAPI
gewinnt. Bug-Reports willkommen.
