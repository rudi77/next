# trainpipe Roadmap

Ziel: trainpipe ist die zentrale Plattform für Training, Verbesserung
und Evaluierung von LLMs — heute für Dokumentenextraktion, später für
Domänen-LLMs (Buchhalter-DACH, Company-LLMs).

Diese Datei dient gleichzeitig als Backlog **und** als Fortschritts-Tracker.
Items werden abgehakt sobald sie merged sind; eine Phase ist *fertig*,
wenn alle Subtasks abgehakt **und** das Acceptance-Kriterium nachweisbar
erfüllt ist.

## Legende

- `[ ]` todo
- `[~]` in arbeit
- `[x]` fertig
- **(P0/P1/P2)** Priorität — P0 schließt einen Loop, P1 verlangsamt
  einen konkreten Use-Case wenn er fehlt, P2 ist nice-to-have
- ~~strike~~ = verworfen mit Begründung

## Currently in focus

> **Phase 6 — Eval-Framework**. Ohne Eval ist „verbessern" blind.

---

## Was steht (Stand 2026-05-29)

- [x] FastAPI + asyncio Scheduler + SQLite-Queue, Crash-Recovery
- [x] Atomic Claim/Launch-Split, GPU-Lease-Accounting
- [x] REST API + X-API-Key Auth + SSE-Logs
- [x] Optuna-Sweep-Driver mit dotted-path Search-Space
- [x] Single-page Web-UI (Tailwind + Alpine, no build step)
- [x] MCP-Server mit 15 Tools
- [x] Dataset-Upload + Registry + `ds:<id>`-Refs + Format-Validation
- [x] SHA256-Dedup, Active-Reference-Protection on delete
- [x] MLflow Wiring (Auto-Tags, Run-Finalize, Credential-Stripping in /ui/config)
- [x] Path-Validation am Submit (422 statt Trainer-Crash)
- [x] Empty-Dataset-Check route-level

Was bis hier fehlt um „zentrale LLM-Plattform" zu sein:
Evaluierung jenseits von train_loss · Modell-Promotion · Inference-Probe ·
Multimodal-Verifizierung · Multi-Stage-Pipelines · PII-Audit · Active
Learning. Siehe Phasen unten.

---

## Phase 6 — Eval-Framework (P0, ~2-3 Wochen)

**Goal:** Ein Run hat einen messbaren Score gegen eine wiederholbare
Eval-Suite. Zwei Runs lassen sich side-by-side vergleichen — incl. der
einzelnen Beispiele bei denen einer regressed ist.

**Acceptance:** End-to-end `submit experiment → auto-eval against suite X
→ Comparison-UI zeigt run A vs run B mit per-sample Δ` funktioniert.

### Schema + DB
- [ ] Migration v3: `eval_suites`, `eval_runs`, `eval_results`
- [ ] Pydantic-Modelle: `EvalSuite`, `EvalRun`, `EvalResult`, `MetricConfig`
- [ ] Repository CRUD

### Metric-Backends
- [ ] `exact_match` (für Extraktion / strukturierte Outputs)
- [ ] `field_level_f1` (JSON vs Gold-JSON, schema-aware, partial credit)
- [ ] `rouge_l` (für Chat-Antworten)
- [ ] `llm_as_judge` mit Claude/GPT-4 — Provider env-konfigurierbar,
  Rubrik als YAML
- [ ] `bleu` (optional)
- [ ] Plugin-Interface: weitere Metrics als Python-Entry-Point registrierbar

### Eval-Runner
- [ ] Spawnt `swift infer` (oder vLLM-Call) gegen den Adapter eines
  fertigen Runs
- [ ] Pro Sample: prediction + gold + per-metric Score → `eval_results`
- [ ] Aggregat: mean, std, count, per-class wenn anwendbar
- [ ] Reuse Scheduler-Lease-Logic für GPU-Belegung

### API
- [ ] `POST /evals/suites` — create + validate (Dataset existiert, Metric-Config parsed)
- [ ] `GET /evals/suites`, `GET /evals/suites/{id}`, `DELETE`
- [ ] `POST /evals/runs` — manuell triggern (`model_target + suite_id`)
- [ ] `GET /evals/runs/{id}` incl. per-sample
- [ ] `GET /evals/compare?run_ids=a,b,c` — n-way Vergleich + Δ-Berechnung

### After-Training-Hook
- [ ] `ExperimentSpec.auto_eval: list[suite_id]` Feld
- [ ] Scheduler triggert Eval nach `status=completed`
- [ ] Eval-Result mit `experiment_id` verknüpft

### UI
- [ ] Tab „Evals": Suites listen + erstellen
- [ ] Per-Experiment Panel mit Eval-Resultaten
- [ ] Comparison-View: 2-N Runs nebeneinander, mit
  Δ-Highlights für regressierte Beispiele
- [ ] MLflow-Run-Tags um Eval-Resultate ergänzen (Sortierung/Filter)

### Tests
- [ ] Metric unit tests (edge cases: leere Outputs, falscher Schema-Typ, Unicode)
- [ ] Eval-Runner mit gemocktem swift infer
- [ ] End-to-end smoke: Experiment → Auto-Eval → Result via API

---

## Phase 7 — Modell-Registry & Promotion (P0, ~1 Woche)

**Goal:** Ein Run-Ergebnis hat einen Namen, eine Version, und kann als
`production` / `staging` markiert werden. Eval-Resultate aus Phase 6
sind die Promotion-Basis.

**Acceptance:** `claude → "promote run abc123 als invoice-extractor v4
production" → MCP-Tool macht es, UI zeigt invoice-extractor@production
→ Inference (Phase 8) kann das Modell laden`.

### Schema + DB
- [ ] Migration v4: `models` (name, version, run_id, adapter_path,
  eval_summary, created_at), `model_aliases` (name, alias, model_id)
- [ ] Pydantic: `RegisteredModel`, `ModelAlias`

### API
- [ ] `POST /models` — registriert ein Run als named model + version
- [ ] `GET /models` (filter: name, alias)
- [ ] `GET /models/{name}` — alle Versionen
- [ ] `GET /models/{name}/{alias}` — resolved
- [ ] `POST /models/{name}/aliases/{alias}` — assign/move alias
- [ ] `DELETE /models/{id}` — mit Active-Use-Check

### UI
- [ ] Tab „Models": Versionen + Aliases, klick → Run-Detail
- [ ] Im Experiment-Detail: „Register as model" Button (wenn `status=completed`)
- [ ] Promotion-Workflow: warning wenn Eval-Resultate fehlen / Score schlechter als aktuelles `production`

### MCP
- [ ] `register_model`, `set_alias`, `resolve_model` Tools

### Tests
- [ ] Alias-Constraint (nur ein Modell pro alias pro name)
- [ ] Promotion-Regression-Warning

---

## Phase 8 — Inference-Probe / Playground (P0, ~1 Woche)

**Goal:** Vor Production-Promotion willst du selbst ein paar Prompts
durchschicken und sehen wie das Modell reagiert. Auch Side-by-Side
Base ↔ Fine-tuned.

**Acceptance:** UI-Playground mit Prompt → Antwort, optional Vergleich
zwei Versionen, plus `POST /inferences` als API.

### Backend
- [ ] v1: `transformers.AutoModel.from_pretrained(base) +
  PeftModel.from_pretrained(adapter)` → generate
- [ ] Modell-Cache (LRU, max N geladen)
- [ ] Streaming-Response über SSE
- [ ] v2 (später): vLLM/sglang Backend optional

### API
- [ ] `POST /inferences` (model_ref, prompt, params) → streamed response
- [ ] `POST /inferences/compare` (model_refs[], prompt) → parallel responses

### UI
- [ ] Tab „Playground": Modell-Auswahl (Dropdown von Registered Models +
  Base-Model), Prompt, Streaming-Antwort
- [ ] Compare-Modus: 2 Spalten, dieselbe Prompt, beide Modelle

### MCP
- [ ] `inference(model_ref, prompt)` Tool
- [ ] `inference_compare(model_refs, prompt)` Tool

### Tests
- [ ] Modell-Cache eviction
- [ ] Streaming-Chunk-Format

---

## Phase 9 — Multimodal-Verifizierung + Image-JSONL (P1, ~1 Woche)

**Goal:** Doc-Extraktion mit Qwen2-VL etc. funktioniert end-to-end:
Upload eines image-haltigen Datasets, Training, Inference, Eval.

**Acceptance:** Qwen2-VL-2B-Fine-Tune auf einem 100-Sample-Doc-Set
landet als promotbares Modell.

### Dataset-Format
- [ ] `dataset_formats.detect_and_validate` erkennt `images` und
  `videos` Schema in JSONL
- [ ] Image-Pfade müssen relativ zum Dataset-Root sein und beim Upload
  als Bundle hochgeladen werden (Zip oder Multi-File)
- [ ] `POST /datasets/bundle` für Multi-File-Upload

### Swift-Builder
- [ ] Verifizieren dass `--model_type` für VLMs korrekt gesetzt wird
- [ ] `multimodal: MultimodalSettings` → richtige env (SIZE_FACTOR, MAX_PIXELS)
- [ ] Test E2E mit minimal Qwen2-VL Sample-Set

### Eval-Metriken für Doc-Extraktion
- [ ] `bounding_box_iou` (für Layout-Tasks)
- [ ] `structured_extraction_f1` (Feld-für-Feld vs Gold)

### UI
- [ ] Datasets-Tab erkennt multimodal und zeigt Image-Preview-Thumb

---

## Phase 10 — Annotation-Bridge (Label Studio Import) (P1, ~3-4 Tage)

**Goal:** Direkter Import aus Label Studio Projects ohne Format-Frickelei.

**Acceptance:** `POST /datasets/from-labelstudio?project_id=42&token=...`
holt das Projekt, mapped Exports auf das passende JSONL-Format,
registriert als Dataset.

- [ ] LS-Client (Auth, get-export)
- [ ] Mapper: LS-Annotation-Schemas → unsere JSONL-Formate
- [ ] Support: Text-NER, Doc-Layout, Conversation
- [ ] Inkrementeller Import (nur neue Annotations seit X)
- [ ] UI: Dataset-Upload-Modal hat „Import from Label Studio"

---

## Phase 11 — Active-Learning-Schleife (P1, ~2 Wochen)

**Goal:** Annotation-Effizienz statt brute-force. Nach jedem Training
identifiziert das System die ungewissesten Samples auf unbeschrifteten
Docs, surfaced sie als Annotation-Queue, retraint mit den neu
beschrifteten.

**Acceptance:** Cycle `train → score uncertain → annotate → retrain`
läuft halb-automatisch über 3 Iterationen, jede mit messbarer
Eval-Verbesserung.

### Backend
- [ ] `POST /active-learning/runs` (model + unlabeled_dataset)
- [ ] Inference über alle unlabeled samples, Confidence + Uncertainty
  (token-entropy, ensemble disagreement)
- [ ] Ranking + Top-N als „Queue"
- [ ] Schema: `annotation_queues` Tabelle

### Integration
- [ ] Label Studio-Push: Queue → LS-Project mit Pre-Annotations
- [ ] Loop: Annotation done → trigger next train → eval → next al run

### UI
- [ ] Tab „Active Learning": Queue mit Sample-Snippets + Confidence
- [ ] Iteration-Dashboard: Eval-Score-Curve über Iterationen

---

## Phase 12 — Multi-Stage Pipelines (CPT → SFT → DPO) (P1, ~2 Wochen)

**Goal:** Domain-LLM-Workflow als ein Objekt deklarieren: continued
pretraining → instruction tuning → preference alignment. Jede Stage
übernimmt den Checkpoint der vorigen.

**Acceptance:** `POST /pipelines` mit 3-stage DAG, jede Stage spawnt
nach Erfolg der vorigen, finaler Checkpoint wird registriert.

### Schema
- [ ] Migration v5: `pipelines`, `pipeline_stages`
- [ ] Pydantic: `PipelineConfig` (stages: list[StageSpec])
- [ ] StageSpec referenziert ExperimentSpec + dependencies + input-from-stage

### Orchestrator
- [ ] Pipeline-Driver (ähnlich StudyDriver): überwacht alle Stages,
  wartet auf Vorgänger, propagiert Adapter-Pfade nach unten
- [ ] Crash-Recovery: angefangene Pipelines resuminen

### API
- [ ] `POST /pipelines`, `GET /pipelines`, `GET /pipelines/{id}`
- [ ] `POST /pipelines/{id}/cancel` (kaskadiert auf alle Stages)

### UI
- [ ] Tab „Pipelines": DAG-View mit Status pro Stage
- [ ] Stage-Detail-Drill-down → das jeweilige Experiment

---

## Phase 13 — DPO/RLHF Support in ExperimentSpec (P1, ~1 Woche)

**Goal:** Voraussetzung für Phase 12's DPO-Stage und auch standalone
brauchbar.

**Acceptance:** Submit eines DPO-Specs mit chosen/rejected-Dataset
→ ms-swift trainiert via `swift rlhf`, MLflow zeigt DPO-Metrics
(reward_chosen, reward_rejected, kl_divergence).

- [ ] `ExperimentSpec.train_kind: Literal["sft", "dpo", "kto", "ppo", "grpo"]`
- [ ] Eigenes Dataset-Format `{prompt, chosen, rejected}` im
  `dataset_formats.validate`
- [ ] `swift_builder` switched auf `swift rlhf` mit `--rlhf_type dpo`
- [ ] UI: SFT-Type Select wird zu „Training Type" Select

---

## Phase 14 — Synthetic Data Generation (P1, ~1 Woche)

**Goal:** Mit einem Teacher-LLM aus wenigen Beispielen viele
Trainings-Pairs generieren — z.B. aus 1000 Rechnungen + ihren JSONs
5000 augmentierte Varianten.

**Acceptance:** `POST /synth` mit Anthropic/OpenAI-Provider und
Instruction → läuft als Job, schreibt das Resultat als neues Dataset.

### Backend
- [ ] `POST /synth` (provider, model, source_dataset_id, instruction,
  target_count, seed)
- [ ] Job läuft als trainpipe-Subprozess (kein swift), nutzt Anthropic
  oder OpenAI SDK
- [ ] Outputs werden incrementally in eine neue JSONL geschrieben
- [ ] Bei completed: automatisch als Dataset registrieren mit Tag
  „source: synth from X via Y"

### MCP
- [ ] `synth_dataset` Tool (so dass der Agent Synthese als
  zwischenschritt selbst auslösen kann)

### Audit
- [ ] Provenance-Tags am Dataset (welcher Teacher, welche Instruction)

---

## Phase 15 — PII Redaction & Audit-Trail (P1, ~1-2 Wochen)

**Goal:** GDPR-tauglich für DACH/Company-Daten. Vor jedem Training
wird der Datensatz durch eine PII-Detection geschickt; auditierbarer
Trail welches Modell mit welchen Datensätzen (Hashes) trainiert wurde;
"Recht auf Löschung" verfolgbar.

**Acceptance:** Dataset-Upload kann `--auto-redact` setzen; produziert
einen redacted-Twin; Modell-Detail zeigt aus welchen Datasets es
trainiert wurde; ein Tool listet alle Modelle die ein bestimmtes
Original-Dataset gesehen haben.

### Backend
- [ ] presidio (oder Spacy-NER) als optional dependency
- [ ] `POST /datasets/{id}/redact` (entities: list, replacement_strategy)
- [ ] Redacted result = neues Dataset mit Provenance-Link
- [ ] Migration v6: `model_lineage` (model_id, dataset_id, used_at)

### UI
- [ ] „Redact" Action im Dataset-Detail
- [ ] „Trained on" Liste im Model-Detail mit Dataset-Versions
- [ ] Suche: „welche Modelle haben Dataset X benutzt?"

### Compliance-Workflow
- [ ] „Forget user Y" Skript: identifiziert Datasets die Y enthalten,
  markiert Modelle die diese Datasets gesehen haben für Retraining

---

## Phase 16 — Dataset-Versionierung, Splits, Mixing (P2, ~1 Woche)

**Goal:** Datasets sind versioniert (immutable), Splits sind
deklarativ, Training-Mixes (z.B. 30% Domäne + 70% Chat) als
first-class.

**Acceptance:** `dataset@v2` Syntax in `ds:`-Ref, `POST /datasets/{id}/split`,
`POST /mixes` für gewichtete Kombinationen.

- [ ] Dataset-Version-Field (immutable nach create)
- [ ] `POST /datasets/{id}/split?ratio=90:10` → erzeugt train+val
- [ ] `POST /mixes` mit dataset_id+weight Liste → composed dataset
- [ ] `ds:<id>@v2#500` Syntax
- [ ] UI: Version-Badge, Split-Button, Mix-Editor

---

## Phase 17 — Continuous Training / Drift-Detection (P2, ~2 Wochen)

**Goal:** Production-Metric sinkt → automatisch retrain triggern.
Oder zeitgesteuert „jede Woche neu trainieren mit den neuen Docs".

**Acceptance:** Eine Watch-Config kann auf Eval-Score-Threshold oder
Cron-Schedule triggern und einen neuen Pipeline-Run starten.

- [ ] `watches` Tabelle: trigger (cron / metric-threshold), pipeline_id, enabled
- [ ] APScheduler oder ähnliches im Scheduler
- [ ] Drift-Monitor: Production-Inference logged Scores → trigger wenn
  rolling-window-mean < threshold
- [ ] UI: Tab „Watches"

---

## Phase 18 — Distributed Training (Multi-Host, DeepSpeed/FSDP) (P2, ~2-3 Wochen)

**Goal:** Wenn ein Single-Host nicht mehr reicht (13B+ full FT, oder
einfach mehr Throughput), distributed training konfigurierbar.

**Acceptance:** ExperimentSpec kann `deepspeed_zero_stage=3` setzen,
trainpipe orchestriert multi-host wenn nötig.

- [ ] `ExperimentSpec.distributed: DistributedConfig` (zero_stage,
  num_nodes, host_list / SSH config)
- [ ] Scheduler unterstützt Multi-Host-GPU-Pool (mehrere `gpu_leases`-Tabellen?)
- [ ] swift_builder erzeugt torchrun mit --nproc_per_node + --nnodes
- [ ] Out-of-Scope für jetzt: Kubernetes-Backend

---

## Phase 19 — Quantization-Pipeline (P2, ~1 Woche)

**Goal:** Nach SFT/DPO automatisch AWQ/GPTQ quantisieren, eval'en,
promotable machen — für günstige Inference.

**Acceptance:** `POST /models/{id}/quantize?method=awq&bits=4` → neuer
Model-Eintrag mit quantisierter Variante + Auto-Eval um Quality-Loss zu
messen.

- [ ] `POST /models/{id}/quantize` als Job
- [ ] AWQ + GPTQ Backends (autoawq / gptqmodel)
- [ ] Auto-Eval mit derselben Suite wie das Original → Δ-Tracking
- [ ] UI: „Quantize" Action im Model-Detail

---

## Phase 20 — Cost/Resource-Tracking (P2, ~3-4 Tage)

**Goal:** GPU-Stunden, Watts, optional $-Equivalent pro Experiment.
Ranking-Leaderboard „bang per watt".

- [ ] Pro Experiment: gpu_seconds, peak_vram, energy_wh
- [ ] nvml-Polling im Scheduler während des Runs, in `events` aggregieren
- [ ] UI: Experimente-Tabelle hat optionale Cost-Spalten, Studies haben
  „Cost vs. Best-Metric" Plot

---

## Phase 21 — Tokenizer-Erweiterung (P2, ~3 Tage)

**Goal:** Fachvokabular (Buchhaltungsbegriffe, interne Codes,
Produkt-Codes) als zusätzliche Tokens, damit das Modell sie nicht in
3-4 BPE-Stücke zerlegen muss.

- [ ] `ExperimentSpec.extra_tokens: list[str]` Feld
- [ ] swift_builder leitet weiter (ms-swift unterstützt das via
  resize_token_embeddings)
- [ ] Eval-Hook: Vor/Nach-Vergleich der Tokenisierung der Eval-Suite

---

## Out of Scope (mit Begründung)

- ~~RAG-Infrastructure~~ — separates Inference-Time-Problem, gehört in
  einen eigenen Service. trainpipe stellt das fine-tuned Modell; der
  RAG-Stack die Retrieval-Pipeline. Sonst kleben wir zwei orthogonale
  Probleme zusammen.
- ~~Volle Annotation-UI~~ — Label Studio existiert, ist gut, ist OSS.
  Nur Import-Adapter bauen (Phase 10), nicht die UI nachimplementieren.
- ~~Multi-Tenancy / RBAC~~ — solange ihr 1-3 ML-Engineers seid,
  API-Key-pro-Instanz reicht. Multi-Tenant kommt rein wenn das Tool
  Firmen-weit verteilt wird, vorher overkill.
- ~~Eigene MLflow-Reimplementation~~ — MLflow ist gut genug für
  experiment_tracking + artifact storage. Wir bauen darauf auf statt
  drumrum.

## Offene Architektur-Fragen (zu klären in Phase 6 Kick-off)

- Eval-Runner: in-process im trainpipe-Scheduler oder separater
  Worker-Pool? Beide haben Pros — in-process ist simpler, separater
  Pool skaliert besser bei vielen LLM-as-Judge-Calls
- Metric-Plugin-Mechanismus: Entry-Point in pyproject oder
  Verzeichnis-Scan unter `trainpipe/evals/metrics/`?
- LLM-as-Judge: über die `claude-api` Skill (intern) oder direkt
  Anthropic SDK?
- Streaming-Inference: SSE oder WebSocket? SSE konsistent mit
  bestehenden /logs/stream, aber WebSocket erlaubt bidirektional
  (z.B. interrupt mid-generation)

## Referenzen

- ms-swift docs für DPO/RLHF: <https://swift.readthedocs.io/en/latest/Instruction/RLHF.html>
- presidio (PII): <https://github.com/microsoft/presidio>
- Label Studio API: <https://labelstud.io/api>
- AutoAWQ: <https://github.com/casper-hansen/AutoAWQ>
- Optuna (bereits integriert): <https://optuna.org>
