---
feature: distributed-training
status: planned
since: 2026-05-29
last_verified: 2026-05-29
owner:
adr: ROADMAP.md#phase-18
---

# Distributed Training — Multi-Host (DeepSpeed/FSDP)

**Geplant (ROADMAP Phase 18) — noch nicht implementiert.**

Wenn ein Single-Host nicht mehr reicht (13B+ full FT oder mehr Throughput),
wird verteiltes Training konfigurierbar. Ziel: eine ExperimentSpec kann
`deepspeed_zero_stage=3` setzen, trainpipe orchestriert bei Bedarf über
mehrere Hosts.

## Capabilities (was der Nutzer tun kann)

- Eine verteilte Konfiguration an einem Experiment setzen (ZeRO-Stage, Knotenzahl, Host-Liste/SSH)
- Einen Multi-Host-Lauf starten, der über mehrere Knoten skaliert

## Invariants (was immer gelten muss)

- `deepspeed_zero_stage` ist Teil der Spec und wird an ms-swift durchgereicht
- `swift_builder` erzeugt `torchrun` mit `--nproc_per_node` + `--nnodes`
- Der Scheduler unterstützt einen Multi-Host-GPU-Pool (Leases über mehrere Hosts)
- Public-Feldnamen bleiben stabil; das Flag-Mapping bleibt im swift_builder isoliert

## API surface (der Vertrag für Clients)

- (keine neue Route — erweitert `POST /experiments` um `distributed`-Konfiguration)

## Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `ExperimentSpec.distributed: DistributedConfig` (zero_stage, num_nodes, host_list / SSH)

## Extension points (für Plugins / externe Nutzung)

- `training/swift_builder.py` — `torchrun`-Generierung für Multi-Node
- GPU-Pool — Multi-Host-Lease-Accounting (Erweiterung der heutigen Single-Host-Logik)

## Tests (müssen existieren und grün sein)

- (geplant) swift_builder erzeugt korrektes `torchrun --nproc_per_node --nnodes`
- (geplant) Multi-Host-Lease-Allokation

## Known gaps

- Gesamtes Feature noch nicht gebaut: kein `distributed`-Feld, keine torchrun-
  Generierung, kein Multi-Host-Pool.
- Kubernetes-Backend ist bewusst out of scope.

## Cross-references

- related_spec: [training-experiments](training-experiments.md) — erweitert Spec + swift_builder
- related_spec: [platform-foundation](platform-foundation.md) — GPU-Pool/Lease-Accounting
- adr: ROADMAP.md — Phase 18 „Distributed Training (Multi-Host, DeepSpeed/FSDP)"
