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

> **Phasen 7–21 implementiert (2026-05-29)** — Modell-Registry +
> Promotion, Inference Playground mit LRU-Cache, Multimodal+Bundle-Upload,
> Label-Studio-Bridge mit SSRF-Schutz, Active-Learning, Multi-Stage
> Pipelines, DPO/RLHF, Synthetic-Data, PII-Redaction + Lineage,
> Dataset-Versionierung + Splits + Mixes, Watches (Drift-Detection),
> Distributed-Config, Quantization, Cost-Tracking, Tokenizer-Erweiterung.
> 438 Tests grün. Manche UI-Tabs sind als Folge-Items aufgeführt — der
> komplette REST+MCP-Workflow steht.

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
- [x] **Phase 6 ✅**: Eval-Framework (Suites, Runs, Results) — 4 Metric-Backends
  (exact_match, field_level_f1, rouge_l, llm_as_judge), Plugin-Scan,
  in-process Driver+Dispatcher, REST + UI + auto_eval-Hook nach Training,
  Compare-API mit Δ + Regression-Detection

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
- [x] Migration v3: `eval_suites`, `eval_runs`, `eval_results`
- [x] Pydantic-Modelle: `EvalSuite`, `EvalRun`, `EvalResult`, `MetricConfig`
- [x] Repository CRUD

### Metric-Backends
- [x] `exact_match` (für Extraktion / strukturierte Outputs)
- [x] `field_level_f1` (JSON vs Gold-JSON, flatten + partial credit)
- [x] `rouge_l` (LCS-F1 über Whitespace-Tokens, beta-weighted)
- [x] `llm_as_judge` mit Anthropic/OpenAI — Provider env-konfigurierbar,
  Rubrik als YAML, retries + score-normalization
- [x] `bleu` (sentence-level, n-gram precision + BP, Chen-&-Cherry-Smoothing,
  eigene Implementation ohne sacrebleu/nltk-Dep)
- [x] Plugin-Interface: Directory-Scan unter `trainpipe/evals/metrics/`

### Eval-Runner
- [x] In-process Driver mit pluggable Inference-Backend
  (TransformersInferenceBackend prod default, MockInferenceBackend für Tests).
  Swift/vLLM-Backends als zukünftige Plugins.
- [x] Pro Sample: prediction + gold + per-metric Score → `eval_results`
- [x] Aggregat: mean, std, count via Metric.aggregate()
- [x] Reuse GpuPool für Lease-Belegung (geteilter Pool mit Training);
  EvalDispatcher mit Crash-Recovery + atomic claim

### API
- [x] `POST /evals/suites` — create + validate (ds:-ref resolve, Metric-Config validate)
- [x] `GET /evals/suites`, `GET /evals/suites/{id}`, `DELETE` (?force= zum
  Lösen aktiver Runs via CASCADE)
- [x] `POST /evals/runs` — manuell triggern (`suite_id + experiment_id`)
- [x] `GET /evals/runs`, `GET /evals/runs/{id}`, `GET /evals/runs/{id}/results`,
  `POST /evals/runs/{id}/cancel`
- [x] `GET /evals/compare?run_ids=a,b,c` — n-way Vergleich, Δ-Aggregat +
  Regressions-Liste per Sample

### After-Training-Hook
- [x] `ExperimentSpec.auto_eval: list[suite_id]` Feld
- [x] Scheduler triggert Eval nach `status=completed` (Scheduler._enqueue_auto_evals)
- [x] Eval-Run mit `experiment_id` verknüpft, `triggered_by="auto"`,
  unbekannte Suite-IDs als WARN geloggt statt zu failen

### UI
- [x] Tab „Evals": Suites listen + erstellen (JSON-Editor wie bei Studies)
- [x] Per-Run-Detail mit Aggregate + Per-Sample-Predictions + Scores
- [x] Comparison-View: Checkbox-Auswahl auf der Runs-Tabelle → Compare-Modal
  mit Aggregate-Δ-Tabelle + Liste der Samples die zwischen Runs divergieren
- [x] MLflow-Metrics + Tags für Eval-Resultate: nach jedem completed eval
  wird `eval.<suite>.<metric>` (mean/std/count) als Metric auf den
  Experiment-Run gepusht, plus Tag `trainpipe.eval.<suite>` mit der
  eval_run_id → Sortier-/Filterbar in der MLflow-UI

### Tests
- [x] Metric unit tests (15 exact_match + 31 für field_level_f1/rouge_l/
  llm_as_judge inkl. edge cases: leere Outputs, invalid JSON,
  case-(in)sensitivity, retries, scale-normalization)
- [x] Eval-Runner mit MockInferenceBackend, inkl. predict-failure per sample,
  missing-dataset, unknown-metric, sample_limit, dispatcher drain + recovery
- [x] End-to-end smoke (tests/test_eval_e2e.py): create suite → 2 Experimente
  mit auto_eval → Scheduler-Hook → Dispatcher → Compare-API mit per-sample Δ

---

## Phase 7 — Modell-Registry & Promotion (P0, ~1 Woche)

**Goal:** Ein Run-Ergebnis hat einen Namen, eine Version, und kann als
`production` / `staging` markiert werden. Eval-Resultate aus Phase 6
sind die Promotion-Basis.

**Acceptance:** `claude → "promote run abc123 als invoice-extractor v4
production" → MCP-Tool macht es, UI zeigt invoice-extractor@production
→ Inference (Phase 8) kann das Modell laden`.

### Schema + DB
- [x] Migration v4: `models` (name, version, run_id, adapter_path,
  eval_summary, created_at), `model_aliases` (name, alias, model_id)
- [x] Pydantic: `RegisteredModel`, `ModelAlias`

### API
- [x] `POST /models` — registriert ein Run als named model + version
- [x] `GET /models` (filter: name, alias)
- [x] `GET /models/{name}` — alle Versionen
- [x] `GET /models/{name}/{alias}` — resolved (alias OR numeric version)
- [x] `POST /models/{name}/aliases/{alias}` — assign/move alias
- [x] `DELETE /models/{id}` — mit Active-Use-Check (409 wenn Alias hält, ?force=true override)

### UI
- [x] Tab „Models": Versionen + Aliases, Best-Eval Spalte
- [x] Im Experiment-Detail: „Register as model" Button (wenn `status=completed`)
- [x] Promotion-Workflow: warning wenn keine completed evals für die experiment_id beim alias=production

### MCP
- [x] `register_model`, `set_alias`, `get_model` (resolve), `list_models`, `delete_model` Tools

### Tests
- [x] Alias-Constraint (UPSERT primary key auf (name, alias), test_alias_assign_move_and_filter)
- [x] Promotion ohne evals = UI-Warnung (Logik in submitRegisterModel, requires zweiten Klick)

---

## Phase 8 — Inference-Probe / Playground (P0, ~1 Woche)

**Goal:** Vor Production-Promotion willst du selbst ein paar Prompts
durchschicken und sehen wie das Modell reagiert. Auch Side-by-Side
Base ↔ Fine-tuned.

**Acceptance:** UI-Playground mit Prompt → Antwort, optional Vergleich
zwei Versionen, plus `POST /inferences` als API.

### Backend
- [x] v1: `transformers.AutoModel.from_pretrained(base) +
  PeftModel.from_pretrained(adapter)` → generate (wiederverwendet
  TransformersInferenceBackend aus Phase 6 evals/inference.py)
- [x] Modell-Cache LRU (`InferenceService`, max_loaded=2 default,
  evict→close hängt am asyncio-Lock)
- [x] Streaming-Response über SSE (`POST /inferences/stream` chunk-by-chunk;
  token-level Streaming via TextIteratorStreamer als Folge-Refactor möglich
  ohne Wire-Protocol-Änderung)
- [ ] v2 (später): vLLM/sglang Backend optional — out of scope für Phase 8

### API
- [x] `POST /inferences` (model_ref, prompt, params) → synchronous
- [x] `POST /inferences/stream` → SSE-stream (token/done events)
- [x] `POST /inferences/compare` (model_refs[], prompt) → sequential responses
- [x] `GET /inferences/cache` — Diagnose der LRU-Belegung

### UI
- [x] Tab „Playground": Datalist von Registered Models + Base-Model freier Input
- [x] Compare-Modus: 2 Spalten, dieselbe Prompt, beide Modelle (B leer lassen für Single-Mode)

### MCP
- [x] `inference(model_ref, prompt)` Tool
- [x] `inference_compare(model_refs, prompt)` Tool

### Tests
- [x] Modell-Cache eviction (`test_lru_eviction_closes_oldest`)
- [x] Streaming-Chunk-Format (`test_stream_chunks_and_done`)
- [x] Cache-Hit Re-use (`test_cache_hit_does_not_rebuild`)
- [x] close_all drains cache

---

## Phase 9 — Multimodal-Verifizierung + Image-JSONL (P1, ~1 Woche)

**Goal:** Doc-Extraktion mit Qwen2-VL etc. funktioniert end-to-end:
Upload eines image-haltigen Datasets, Training, Inference, Eval.

**Acceptance:** Qwen2-VL-2B-Fine-Tune auf einem 100-Sample-Doc-Set
landet als promotbares Modell.

### Dataset-Format
- [x] `dataset_formats.detect_and_validate_info` erkennt `images`,
  `videos`, `audios` Schema in JSONL (Sampling über erste 100 Zeilen),
  persistiert als `Dataset.media_kinds`
- [x] Image-Pfade relativ zur bundle root; `image_root` Column persistiert
  Extraktionsverzeichnis
- [x] `POST /datasets/bundle` für Zip-Upload (mit zip-slip Defense,
  manifest-only-jsonl validation, single-jsonl requirement)
- [x] `GET /datasets/{id}/media?path=...` serviert Bundle-Files mit
  Traversal-Schutz (für UI thumbnails)

### Swift-Builder
- [x] `--model_type` durchgereicht via `ExperimentSpec.model_type` (war
  schon da, unverändert lassen — ms-swift v4 erwartet das Feld so)
- [x] `multimodal: MultimodalSettings` setzt SIZE_FACTOR + MAX_PIXELS env
  (war schon da von Phase 1)
- [ ] E2E-Test mit minimal Qwen2-VL Sample-Set — out of scope ohne
  echte GPU; Bundle-Upload + Format-Detection sind verifiziert,
  Trainings-Smoke folgt auf echter Hardware

### Eval-Metriken für Doc-Extraktion
- [x] `bounding_box_iou` (greedy matching mit Label-Strict/Lenient,
  konfigurierbarer IoU threshold; both-empty=1.0 Konvention)
- [x] `structured_extraction_f1` (schema-aware, Numeric-Tolerance,
  Case-Insensitive default, Out-of-Schema = FP)

### UI
- [x] Datasets-Tab zeigt Media-Kind-Badges (🖼️ images / 🎬 videos)
- [x] `/datasets/{id}/media` endpoint vorhanden für künftige Thumb-Embeds;
  Thumb-Display selbst pendet auf Preview-Modal-Refactor

---

## Phase 10 — Annotation-Bridge (Label Studio Import) (P1, ~3-4 Tage)

**Goal:** Direkter Import aus Label Studio Projects ohne Format-Frickelei.

**Acceptance:** `POST /datasets/from-labelstudio?project_id=42&token=...`
holt das Projekt, mapped Exports auf das passende JSONL-Format,
registriert als Dataset.

- [x] LS-Client (Auth via Token-Header, pagination, completed-Filter,
  injectable transport für Tests)
- [x] Mapper: LS-Annotation-Schemas → unsere JSONL-Formate (`integrations/
  labelstudio.py`)
- [x] Support: Text-NER (labels result), Doc-Layout (rectanglelabels mit
  Pixel-Konvertierung), Conversation (textarea/choices)
- [x] Inkrementeller Import via `since_iso` Parameter
  (`completed_at__gte` LS query); SHA256-Dedupe verhindert duplikate
  Dataset-Einträge bei identischem Output
- [x] UI: Datasets-Tab hat „⇲ Label Studio" Button mit Modal

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
- [x] `POST /active-learning/runs` (model_ref + dataset + top_n)
- [x] Inference über alle unlabeled samples mit zwei Scorern: `double_pass`
  (zwei T=0.7 Passes, Dice-Distanz der Outputs) und `length_zscore`
  (Längen-Z-Score als Proxy). Pluggable via `UncertaintyScorer` für
  echte token-entropy Erweiterung.
- [x] Ranking + Top-N als „Queue" via `annotation_queue_items` Tabelle
- [x] Schema: Migration v6 (`active_learning_runs` + `annotation_queue_items`)

### Integration
- [x] Label Studio-Push: `POST /active-learning/runs/{id}/push-labelstudio`
  → eine Task pro Queue-Item mit Modell-Prediction als textarea Pre-Annotation
- [ ] Auto-Loop „Annotation done → next train → eval → next al run" —
  scheduler-side closing nicht implementiert; manuelles Re-Submit nach
  Annotation reicht für Acceptance; Auto-Trigger ist Phase 17 (watches)

### UI
- [x] Tab „Active Learning" — Runs-Tabelle + Queue-Modal mit
  per-Item "Done"-Button (mark annotated)
- [ ] Iteration-Dashboard — depends on watches (Phase 17)

---

## Phase 12 — Multi-Stage Pipelines (CPT → SFT → DPO) (P1, ~2 Wochen)

**Goal:** Domain-LLM-Workflow als ein Objekt deklarieren: continued
pretraining → instruction tuning → preference alignment. Jede Stage
übernimmt den Checkpoint der vorigen.

**Acceptance:** `POST /pipelines` mit 3-stage DAG, jede Stage spawnt
nach Erfolg der vorigen, finaler Checkpoint wird registriert.

### Schema
- [x] Migration v7: `pipelines`, `pipeline_stages` (v7 weil v5/v6 schon
  von Phasen 9/11 vergeben)
- [x] Pydantic: `PipelineConfig` (stages: list[StageSpec])
- [x] StageSpec referenziert ExperimentSpec + dependencies + input-from-stage

### Orchestrator
- [x] PipelineDriver — Polling-Loop wie StudyDriver, enqueued pending
  stages wenn alle deps `completed`, observiert Experiment-Status pro Stage,
  schreibt Stage- + Pipeline-Status zurück
- [x] PipelineManager (analog StudyManager): start_existing für resume,
  create_and_start, cancel, stop_all
- [x] Crash-Recovery: `list_active_pipelines` → für jede pipeline mit
  status queued/running einen Driver starten
- [x] DAG-Validierung beim Create: duplicate names, dangling deps,
  cycles, dangling input_from_stage

### API
- [x] `POST /pipelines`, `GET /pipelines`, `GET /pipelines/{id}`
- [x] `POST /pipelines/{id}/cancel`

### UI
- [x] Tab „Pipelines" — Status-Tabelle mit per-Stage Status-Pills
  (pending/queued/running/completed/failed/skipped) und Tooltip mit
  Fehlertext. Volle DAG-SVG-Visualisierung weiterhin offen.

---

## Phase 13 — DPO/RLHF Support in ExperimentSpec (P1, ~1 Woche)

**Goal:** Voraussetzung für Phase 12's DPO-Stage und auch standalone
brauchbar.

**Acceptance:** Submit eines DPO-Specs mit chosen/rejected-Dataset
→ ms-swift trainiert via `swift rlhf`, MLflow zeigt DPO-Metrics
(reward_chosen, reward_rejected, kl_divergence).

- [x] `ExperimentSpec.train_kind: Literal["sft", "dpo", "kto", "ppo", "grpo"]`
  mit Default `sft` (rückwärtskompatibel)
- [x] `dataset_formats.detect_and_validate_info` setzt `is_preference`
  wenn jede gesampelte Zeile `prompt`/`chosen`/`rejected` non-empty strings hat
- [x] `swift_builder` switched für non-sft auf `swift rlhf --rlhf_type <kind>`,
  alle Hyperparameter-Flags bleiben durchgereicht
- [x] UI: Submit-Modal hat „Training Type" Select neben sft_type
  (sft/dpo/kto/ppo/grpo); Default sft

---

## Phase 14 — Synthetic Data Generation (P1, ~1 Woche)

**Goal:** Mit einem Teacher-LLM aus wenigen Beispielen viele
Trainings-Pairs generieren — z.B. aus 1000 Rechnungen + ihren JSONs
5000 augmentierte Varianten.

**Acceptance:** `POST /synth` mit Anthropic/OpenAI-Provider und
Instruction → läuft als Job, schreibt das Resultat als neues Dataset.

### Backend
- [x] `POST /synth` (provider, model, source_dataset, instruction,
  target_count, seed, max_tokens, name)
- [x] In-Process (kein Sub-Process), httpx-Aufrufe gegen Anthropic
  /messages und OpenAI /v1/chat/completions; per-record Failures werden
  geloggt + überspringen statt den ganzen Batch zu killen
- [x] Output JSONL incrementally; ``MockProvider`` für Tests ohne Netz
- [x] Bei completed: SHA256-Dedup, automatisch als Dataset registriert
  inkl. `_source`-Feld pro Record für Lineage

### MCP
- [x] `synth_dataset` Tool

### Audit
- [x] Provenance in Dataset-Description: provider:model + source path +
  truncated Instruction; jeder Output-Record trägt `_source` mit dem
  Original-Source-Record

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
- [x] Regex-Redactor in `trainpipe/redaction/redactor.py` (Phase 1
  Baseline ohne externe deps; presidio kann später ohne Call-Site-
  Änderung eingehängt werden). Erkennt email, phone, IBAN
  (mod-97 checksum), credit card, DE Tax-ID.
- [x] `POST /datasets/{id}/redact` (entities list) → neues redacted
  Dataset mit Provenance "redacted from ds:<src> (...)"
- [x] Migration v8: `model_lineage` (model_id, dataset_id, used_at)
- [x] `register_model` schreibt automatisch lineage rows für alle
  `spec.dataset`/`val_dataset` Pfade die im Registry sind

### UI
- [ ] „Redact" Action im Dataset-Detail — Endpoint vorhanden, UI-Knopf
  folgt; CLI/MCP reicht für Acceptance
- [x] `GET /datasets/{id}/models` — „welche Modelle haben Dataset X
  benutzt?" Endpoint
- [ ] „Trained on" Liste im Model-Detail UI — Endpoint vorhanden
  (`datasets_used_by_model` als repository), UI-Anzeige folgt

### Compliance-Workflow
- [ ] „Forget user Y" Skript — die zwei Bausteine (`models_using_dataset`
  + `redact_jsonl`) sind da; das Compliance-Skript selbst (Liste-aller-
  PII-Hits → mark-models-for-retrain) ist noch ein offenes Item für
  die Compliance-Owner

---

## Phase 16 — Dataset-Versionierung, Splits, Mixing (P2, ~1 Woche)

**Goal:** Datasets sind versioniert (immutable), Splits sind
deklarativ, Training-Mixes (z.B. 30% Domäne + 70% Chat) als
first-class.

**Acceptance:** `dataset@v2` Syntax in `ds:`-Ref, `POST /datasets/{id}/split`,
`POST /mixes` für gewichtete Kombinationen.

- [x] Dataset-Version-Field (Migration v9, default 1; derived_from links
  zur parent dataset)
- [x] `POST /datasets/{id}/split` → train+val Datasets (ratio "90:10"
  als JSON body, seed-deterministisch via `random.shuffle`)
- [x] `POST /datasets/mixes` mit dataset_id+weight Liste → composed
  dataset (weighted random.choices, target_count default = sum aller
  lines)
- [x] `ds:<id>@v2#500` Syntax (Regex erweitert, `resolve_single` checkt
  Version-Match → MalformedDatasetRef bei Mismatch)
- [ ] UI: Version-Badge / Split-Button / Mix-Editor — REST + MCP-Flow
  reicht für Acceptance; UI folgt

---

## Phase 17 — Continuous Training / Drift-Detection (P2, ~2 Wochen)

**Goal:** Production-Metric sinkt → automatisch retrain triggern.
Oder zeitgesteuert „jede Woche neu trainieren mit den neuen Docs".

**Acceptance:** Eine Watch-Config kann auf Eval-Score-Threshold oder
Cron-Schedule triggern und einen neuen Pipeline-Run starten.

- [x] `watches` Tabelle (Migration v10): trigger (`interval` /
  `metric_threshold`), pipeline_config inline (kein FK auf pipelines,
  damit Pipeline-Editing nicht den Watch bricht), enabled-Flag
- [x] WatchManager als async poll-loop im Scheduler-Lifespan (kein
  APScheduler-Dep — die zwei Cases reichen ohne externes Framework)
- [x] Drift-Monitor: für `metric_threshold` watch wird letzte
  completed eval_run gegen die Suite gelesen und mean unterhalb der
  threshold getriggert; Re-fire-Protect verhindert das gleiche Eval
  doppelt zu triggern
- [x] UI: Tab „Watches" — Tabelle mit kind / trigger / status / last-fired,
  Enable/Disable/Delete Actions, zeigt last_error und Failure-Counter

---

## Phase 18 — Distributed Training (Multi-Host, DeepSpeed/FSDP) (P2, ~2-3 Wochen)

**Goal:** Wenn ein Single-Host nicht mehr reicht (13B+ full FT, oder
einfach mehr Throughput), distributed training konfigurierbar.

**Acceptance:** ExperimentSpec kann `deepspeed_zero_stage=3` setzen,
trainpipe orchestriert multi-host wenn nötig.

- [x] `ExperimentSpec.distributed: DistributedConfig` (zero_stage 0-3,
  num_nodes, host_list, master_addr/port)
- [ ] Scheduler multi-host GPU-Pool — bewusst nicht: das Roadmap-Item
  „Kubernetes-Backend" ist out-of-scope und die Single-Host-Pool-Logik
  reicht für die meisten setups; multi-host braucht operator-level SSH-
  Spawn (`TRAINPIPE_HOST_LIST` env wird durchgereicht)
- [x] swift_builder emittiert `--deepspeed_zero<N>` für stages 1-3 und
  setzt NNODES/MASTER_ADDR/MASTER_PORT/TRAINPIPE_HOST_LIST env vars
  wenn num_nodes > 1
- [x] Out-of-Scope: Kubernetes-Backend (siehe ROADMAP-Vorgabe)

---

## Phase 19 — Quantization-Pipeline (P2, ~1 Woche)

**Goal:** Nach SFT/DPO automatisch AWQ/GPTQ quantisieren, eval'en,
promotable machen — für günstige Inference.

**Acceptance:** `POST /models/{id}/quantize?method=awq&bits=4` → neuer
Model-Eintrag mit quantisierter Variante + Auto-Eval um Quality-Loss zu
messen.

- [x] `POST /models/{id}/quantize` (method=awq|gptq, bits=2-16) →
  registriert die quantisierte Variante als neue Version unter derselben
  Family
- [x] AWQ + GPTQ Backends abstrahiert über `QuantizeBackend` interface;
  Default `SubprocessSwiftQuantizer` spawnt `swift export --quant_method`;
  Tests benutzen `MockQuantizeBackend` (autoawq / gptqmodel werden vom
  Default-Backend angefordert ohne harte Python-Dep)
- [x] Eval-Summary des Parent-Modells wird als initiale Baseline auf
  die quantisierte Version übernommen → UI/Compare zeigt Δ direkt;
  auto_eval-Hook (Phase 6) erzeugt frische Resultate beim nächsten
  Eval-Trigger
- [x] UI: „Quantize" Action im Model-Detail (Modal mit method+bits,
  pollt Modelle nach erfolgreichem POST)

---

## Phase 20 — Cost/Resource-Tracking (P2, ~3-4 Tage)

**Goal:** GPU-Stunden, Watts, optional $-Equivalent pro Experiment.
Ranking-Leaderboard „bang per watt".

- [x] Pro Experiment: `gpu_seconds`, `peak_vram_mb`, `energy_wh` Spalten
  (Migration v11)
- [x] Scheduler-Finalize berechnet gpu_seconds = wall_clock * len(gpu_ids)
  beim Monitor-Exit; peak_vram + energy hängen am operator-side
  nvml-poller, der über `set_experiment_resource_usage` ins DB schreiben
  kann (Hook-Funktion vorhanden)
- [x] UI: Cost-Spalten in Experiment-Tabelle (GPU-hrs + VRAM peak,
  sichtbar ab xl-Breakpoint); Studies-Plot weiterhin offen

---

## Phase 21 — Tokenizer-Erweiterung (P2, ~3 Tage)

**Goal:** Fachvokabular (Buchhaltungsbegriffe, interne Codes,
Produkt-Codes) als zusätzliche Tokens, damit das Modell sie nicht in
3-4 BPE-Stücke zerlegen muss.

- [x] `ExperimentSpec.extra_tokens: list[str]` Feld (max_length=10000)
- [x] swift_builder emittiert `--special_tokens <tok>` einmal pro Eintrag;
  ms-swift macht das resize_token_embeddings dann selbst
- [ ] Eval-Hook für Vor/Nach-Vergleich der Tokenisierung — kein Item
  für Acceptance, gehört in Phase 6 Eval-Metric als optionales
  Diagnostik-Metric; offen.

---

## Known follow-ups aus Code-Review (2026-05-29)

Während der Phasen 7-21 wurden mehrere mittel-priorisierte Issues
identifiziert. Status (2026-05-29 Abend, Follow-up-Pass):

### Atomic & lineage
- [x] **Pipeline-Driver**: `enqueue_stage_with_experiment` Helper —
  INSERT experiments + INSERT events + UPDATE pipeline_stages in einer
  BEGIN IMMEDIATE Transaktion, Rollback bei jedem Fehler.
- [x] **Mix-Provenance**: Migration v13 `dataset_lineage` Tabelle
  (N:M, mit `role`). `_persist_derived` schreibt für mix alle Parents
  rein; `models_using_dataset_recursive` walked Descendants → korrektes
  GDPR-Ergebnis.
- [x] **PipelineManager.create_and_start** holt jetzt `self._lock`
  bevor `_drivers`-Map verändert wird; `_start_driver_locked` ist
  idempotent (re-entry no-op).

### Resilience
- [x] **Watch-Manager**: Migration v12 fügt `consecutive_failures`
  und `last_error` Spalten hinzu. `record_watch_failure` zählt, ab
  `failure_disable_threshold` (default 5) wird der Watch
  auto-disabled. Erfolgreicher Fire resettet den Counter.
- [x] **Synth-Runner**: `max_consecutive_failures` (default 5) +
  `FatalHTTPError` (401/400) trip `SynthAborted` sofort. Route
  übersetzt zu 502.
- [x] **Synth retry-on-429**: `_post_with_retry` mit exponential
  backoff für 429/5xx; FatalHTTPError für 4xx-non-429.

### UI tabs (von Phasen 11, 12, 13, 17 aufgeschoben)
- [x] Active Learning Tab — Runs-Tabelle + Queue-Modal mit
  per-Item "Done" Button
- [x] Pipelines Tab — Status-Tabelle mit per-Stage Pills,
  Cancel-Action (volle DAG-SVG offen)
- [x] Watches Tab — Tabelle + Enable/Disable/Delete, zeigt
  last_error und Failure-Counter
- [x] Cost/Resource columns in Experiments table (GPU-hrs + VRAM
  peak, sichtbar ab xl-Breakpoint)
- [x] Quantize button im Models-Tabelle (Modal mit method+bits)

### Offene Items für später
- [ ] Studies "Cost vs Best-Metric" Plot
- [ ] Pipelines: SVG-DAG-Visualisierung
- [ ] Active Learning: Iteration-Dashboard
- [ ] Models: "Trained on" Liste im Model-Detail mit Dataset-Versions
- [ ] Compliance-Skript "Forget user Y"

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

## Architektur-Entscheidungen (Phase 6 Kick-off, 2026-05-29)

- **Eval-Runner: in-process im Scheduler.** Wiederverwendet Lease-Logic,
  GPU-Pool, Crash-Recovery. Wenn LLM-as-Judge später throughput-limitiert
  wird, separater Pool als Folge-Refactor.
- **Metric-Plugins: Directory-Scan unter `trainpipe/evals/metrics/`.**
  Jede Datei mit einer `Metric`-Subklasse wird beim Start registriert.
  Entry-Points kommen wenn externe Pakete Metrics liefern sollen.
- **LLM-as-Judge: direkt Anthropic + OpenAI SDK** (Provider env-konfigurierbar,
  Modell pro Suite). Rubrik als YAML im Suite-Spec.
- **Streaming-Inference (Phase 8): SSE.** Konsistent mit `/logs/stream`.
  Interrupt via separates `DELETE /inferences/{id}` statt bidirektional.

## Referenzen

- ms-swift docs für DPO/RLHF: <https://swift.readthedocs.io/en/latest/Instruction/RLHF.html>
- presidio (PII): <https://github.com/microsoft/presidio>
- Label Studio API: <https://labelstud.io/api>
- AutoAWQ: <https://github.com/casper-hansen/AutoAWQ>
- Optuna (bereits integriert): <https://optuna.org>
