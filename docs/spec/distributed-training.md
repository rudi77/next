---
feature: distributed-training
status: partial
since: 2026-05-29
last_verified: 2026-06-04
owner:
adr: ROADMAP.md#phase-18
---

# Distributed Training — Multi-Host (DeepSpeed/FSDP)

**Teilweise implementiert (ROADMAP Phase 18).** Single-Host ZeRO ist verdrahtet:
`DistributedConfig` ist Teil der Spec, `swift_builder` emittiert das DeepSpeed-Flag
und surfaced die Multi-Node-Koordination als Env-Variablen. Die eigentliche
Multi-Host-Orchestrierung (SSH-Spawn pro Host, echtes `torchrun`-argv, Lease-
Accounting über mehrere Hosts) liegt heute auf Operator-Ebene und ist noch nicht gebaut.

Wenn ein Single-Host nicht mehr reicht (13B+ full FT oder mehr Throughput),
wird verteiltes Training konfigurierbar. Ziel: eine ExperimentSpec kann
`deepspeed_zero_stage=3` setzen, trainpipe orchestriert bei Bedarf über
mehrere Hosts.

## Capabilities (was der Nutzer tun kann)

- Eine verteilte Konfiguration an einem Experiment setzen (ZeRO-Stage, Knotenzahl, Host-Liste, Master-Addr) — **vorhanden**
- Single-Host-ZeRO (Stage 1/2/3) fahren — **vorhanden**
- Einen Multi-Host-Lauf starten, der über mehrere Knoten skaliert — **Operator-Ebene/geplant**

## Invariants (was immer gelten muss)

- `deepspeed_zero_stage` (1/2/3) ist Teil der Spec; `swift_builder` emittiert
  `--deepspeed_zero<N>` (Stage 0 = aus, emittiert nichts) — **vorhanden**
- Multi-Node-Intent (`num_nodes` > 1) wird als Env durchgereicht — `NNODES`,
  `MASTER_ADDR`, `MASTER_PORT`, `TRAINPIPE_HOST_LIST` — damit der Launcher des
  Betreibers (torchrun/accelerate/SSH-Spawn) sie liest — **vorhanden**
- Public-Feldnamen bleiben stabil; das Flag-Mapping bleibt im swift_builder isoliert

## API surface (der Vertrag für Clients)

- (keine neue Route — erweitert `POST /experiments` um `distributed`-Konfiguration)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `ExperimentSpec.distributed: DistributedConfig` (`deepspeed_zero_stage`,
  `num_nodes`, `host_list`, `master_addr`, `master_port`)

## Extension points (für Plugins / externe Nutzung)

- `training/swift_builder.py` — DeepSpeed-Flag + Multi-Node-Env (Touchpoint für eine
  spätere echte `torchrun`-argv-Generierung)
- GPU-Pool — Multi-Host-Lease-Accounting (Erweiterung der heutigen Single-Host-Logik)

## Tests (müssen existieren und grün sein)

- `tests/test_phase18_distributed.py` — `--deepspeed_zero<N>` für Stage 1/2/3
  (nichts bei Stage 0), Multi-Node-Env-Variablen (NNODES/MASTER_ADDR/…)

## Known gaps

- Kein echtes `torchrun --nproc_per_node --nnodes`-argv und kein Multi-Host-GPU-Pool:
  Multi-Node ist heute als Env-Intent für den Operator-Launcher umgesetzt, nicht als
  von trainpipe selbst gefahrene Orchestrierung.
- Kubernetes-Backend ist bewusst out of scope.

## Cross-references

- related_spec: [training-experiments](training-experiments.md) — erweitert Spec + swift_builder
- related_spec: [platform-foundation](platform-foundation.md) — GPU-Pool/Lease-Accounting
- adr: ROADMAP.md — Phase 18 „Distributed Training (Multi-Host, DeepSpeed/FSDP)"
