---
feature: agentic-data-acquisition
status: proposed
since:
last_verified:
owner:
adr: ROADMAP.md
---

# Agentic Data Acquisition — vom Auftrag zum Trainingsset

> **Status: Proposal / Design-Doc.** Noch nicht implementiert. Dieses Dokument
> ist die Diskussionsgrundlage für ein neues Modul, das einen Trainingsdatensatz
> *from-scratch* aus einem natürlichsprachigen Auftrag konstruiert — durch
> Recherche, Web-Akquise realer Daten **und** synthetische Generierung.

## 1. Motivation & Abgrenzung

Der ROADMAP-Nordstern ist „Domänen-LLMs (Buchhalter-DACH, Company-LLMs)".
Heute kann `next` ein Modell *trainieren, sweepen, evaluieren, promoten* — aber
es setzt voraus, dass jemand **bereits einen Datensatz hat**. Genau dort beginnt
in der Praxis die meiste Arbeit. Dieses Modul schließt die Lücke vor dem
Trainings-Loop: aus „erzeuge Trainingsdaten, mit denen ich ein Buchhalter-LLM
für den DACH-Raum trainieren kann" wird ein registriertes `ds:<id>`, das direkt
in `submit → train → eval` fließt.

### Abgrenzung zu `synth` (wichtig)

Das ist **kein** Erweitern von `synth`. Die beiden sind komplementär:

| | `synth` (heute) | `acquisition` (dieses Proposal) |
|---|---|---|
| Eingang | **existierendes** `source_dataset` (Pflicht) | natürlichsprachiger Auftrag, **keine Quelle nötig** |
| Vorgehen | Teacher-LLM expandiert Records 1:1 | Recherche → Web-Akquise → Synthese → Curation |
| Dauer | sekunden–minuten, synchron in `to_thread` | minuten–stunden, langlaufender Phasen-Job |
| Klärung | keine | interaktiv (MCP) **und** async (`awaiting_input`) |
| Risiko | gering | Recht/DSGVO/Lizenz — Compliance ist Pflichtstufe |

`acquisition` *nutzt* `synth` als eine seiner Phasen wieder (die synthetische
Generierung), erweitert es aber nicht in-place. `synth` behält seine saubere,
auditierbare Single-Responsibility.

## 2. Capabilities (was der Nutzer tun kann)

- Einen Akquise-Auftrag in natürlicher Sprache einreichen („Trainingsdaten für
  ein Buchhalter-LLM, DACH, de-DE/de-AT/de-CH, soll Belege buchen und UStVA
  erklären können, soll **keine** Steuerberatung im Rechtssinn geben").
- Vom System **Klärungsfragen** beantworten — interaktiv via MCP-Agent **oder**
  async via `PATCH` auf einen pausierten Run (`awaiting_input`).
- Eine **Deep-Research-Phase** auslösen, die einen Quellenplan erstellt (welche
  Quellen, welche Themen, welche Lizenz/Erlaubnis).
- Reale Daten aus dem Web akquirieren (Search → Fetch → Extract) **und** parallel
  synthetische Beispiele generieren — beides konvergiert in einen Strom.
- Den akquirierten Rohstrom **kuratieren** lassen: Dedup, Sprach-/Qualitätsfilter,
  **PII-Redaction** (`redaction`-Modul), Format-Validierung.
- Das Ergebnis als reguläres Dataset mit vollständiger **Lineage** registrieren
  (`dataset_lineage`) und es direkt trainieren.
- Den Run auflisten, im Detail inspizieren (Phasen-Status, Quellen, Kennzahlen),
  pausieren, fortsetzen, abbrechen.

## 3. Phasen-Architektur

Ein langlaufender Phasen-Job, modelliert exakt nach `pipelines/driver.py` /
`autoresearch/study.py` (async Task, pollt DB-State, Crash-Recovery). Modul:
`trainpipe/acquisition/`.

```
Auftrag (NL)
   │
   ▼
┌──────────┐   Fragen?   ┌──────────────────┐
│ 1 Intake │ ──────────► │ awaiting_input   │◄── PATCH /acquisitions/{id}/answers
│          │ ◄────────── │ (oder MCP-Dialog)│    (oder MCP-Tool im Agenten-Kontext)
└────┬─────┘             └──────────────────┘
     │ AcquisitionSpec (strukturiert)
     ▼
┌──────────────┐   Web-Search + Fetch + Extract
│ 2 Research/  │   → SourcePlan: Quellen + Themen + Lizenzstatus
│   Plan       │
└────┬─────────┘
     ▼
┌─────────────┐        ┌──────────────┐   (parallel)
│ 3a Acquire  │        │ 3b Synthesize│
│  (real web) │        │  (LLM, reuse │
│             │        │   synth)     │
└────┬────────┘        └──────┬───────┘
     └──────────┬─────────────┘
                ▼  roher JSONL-Strom
        ┌───────────────┐
        │ 4 Curate      │  Dedup · Sprachfilter · Qualitätsscore
        │               │  · PII-Redaction · Format-Validierung
        └──────┬────────┘
               ▼
        ┌───────────────┐
        │ 5 Register    │  → ds:<id> + dataset_lineage (quelle→roh→kuriert)
        └───────────────┘
                │
                ▼  fließt in submit → train → eval
```

### Phase 1 — Intake

Ein LLM (über die bestehende `SynthProvider`-Abstraktion) verwandelt den
Freitext-Auftrag in eine strukturierte `AcquisitionSpec`:

```
AcquisitionSpec:
  domain: str                 # "accounting"
  locales: list[str]          # ["de-DE", "de-AT", "de-CH"]
  target_capabilities: list[str]
  out_of_scope: list[str]     # explizit was das LLM NICHT tun soll
  format: Literal["sft", "dpo", "chat", "completion"]
  target_count: int
  quality_bar: ...
  open_questions: list[str]   # ← wenn nichtleer → Phase pausiert
```

Sind `open_questions` nichtleer, geht der Run in `awaiting_input` (siehe §5).

### Phase 2 — Research/Plan

Deep-Research-Loop: Web-Search → relevante URLs → Fetch → Extraktion des
Haupttexts → das LLM destilliert daraus einen `SourcePlan` (Liste von Quellen
mit Thema, geschätztem Wert, **Lizenz-/robots.txt-Status**). Kein Massendownload
hier — nur Planung und Bewertung.

### Phase 3 — Acquire (real) ‖ Synthesize (synthetisch)

Laufen parallel und schreiben in denselben Rohstrom:

- **3a Acquire:** Holt die im `SourcePlan` freigegebenen Quellen, extrahiert
  saubere Textsegmente, formt sie ins Zielformat (z. B. Frage→Antwort-Paare via
  LLM-Transformation des Rohtexts).
- **3b Synthesize:** Ruft das bestehende `synth.runner` mit der `AcquisitionSpec`
  als Instruction — füllt Lücken und Long-Tail-Fälle, die das Web nicht hergibt.

Das Mischungsverhältnis real/synthetisch ist konfigurierbar.

### Phase 4 — Curate (die eigentliche Arbeit)

„So viel wie möglich scrapen" produziert ohne diese Phase Müll. Schritte:
near-dup-Dedup, Sprach-/Domänenfilter, LLM-basierter Qualitätsscore mit
Schwellenwert, **PII-Redaction** über das bestehende `redaction`-Modul
(bei Buchhaltungsdaten nicht optional), Format-Validierung über
`training.dataset_formats.detect_and_validate_info` (wie `synth` es heute am
Ende tut).

### Phase 5 — Register

Schreibt das kuratierte JSONL, dedupe per sha256 (wie `synth`), registriert als
Dataset mit Provenance und schreibt **`dataset_lineage`**: Auftrag → Quellen →
Rohstrom → kuriertes Set. Lückenlos auditierbar.

## 4. Invariants (was immer gelten muss)

- Ein Run durchläuft Phasen monoton; eine Phase startet erst, wenn die vorige
  `completed` ist (wie `PipelineDriver`).
- Angefangene Runs werden nach einem Crash resümiert (Recovery wie Studies/
  Pipelines); ein im Web-Fetch abgestürzter Run setzt nicht doppelt an.
- **Kein Web-Fetch ohne Lizenz-/robots-Check** im `SourcePlan`. Quellen ohne
  Freigabe werden übersprungen und im Lineage als „skipped (license)" vermerkt.
- **PII-Redaction läuft immer** vor Register — sie ist keine optionale Stufe.
- Das Endprodukt ist immer ein format-validiertes, sha256-dedupliziertes Dataset
  mit vollständiger Lineage — oder der Run scheitert sichtbar (kein halbgares
  `ds:<id>`).
- `awaiting_input` blockiert nur Phase 1→2; ein Run kann beliebig lange pausiert
  bleiben, ohne Provider-Requests zu verbrennen.
- Abbruch ist in jeder Phase möglich und lässt bereits geschriebene Teil-Artefakte
  konsistent zurück (oder räumt sie auf, wie `synth` bei `target_count==0`).

## 5. Klärungsfragen — Doppelpfad (MCP **und** async)

Designentscheidung laut Anforderung: **beides**.

- **Interaktiv (MCP):** Im Agenten-Kontext (Claude Code / Desktop) führt der
  Agent den Dialog synchron, bevor der eigentliche Run startet. Ablauf: MCP-Tool
  `plan_acquisition(auftrag)` → liefert `AcquisitionSpec` + `open_questions`
  zurück; der Agent fragt den Menschen, sammelt Antworten, ruft
  `start_acquisition(spec)` mit vervollständigter Spec. Phase 1 läuft hier
  *vor* dem Job-Start, der Job startet bereits geklärt.
- **Async (`awaiting_input`):** Für reine REST-Nutzung ohne Agent. `POST
  /acquisitions` startet sofort; findet Phase 1 offene Fragen, pausiert der Run in
  `awaiting_input` und legt sie als `open_questions` ab. Der Client pollt, sieht
  die Fragen, antwortet per `PATCH /acquisitions/{id}/answers`; der Driver nimmt
  Phase 1 wieder auf. Kein Chat-Kanal nötig.

Beide Pfade münden in dieselbe `AcquisitionSpec` und denselben Driver — der
Unterschied ist nur, *wann* die Fragen beantwortet werden (vor Start vs. mid-run).

## 6. API surface (der Vertrag für Clients)

- `POST /acquisitions` → 201 (Auftrag + optionale Vorab-Spec) · 422 (leerer
  Auftrag)
- `GET /acquisitions` → 200 · `GET /acquisitions/{id}` → 200 · 404
  (Phasen-Status, SourcePlan, Kennzahlen, resultierendes `dataset_id`)
- `PATCH /acquisitions/{id}/answers` → 200 (beantwortet `open_questions`, nimmt
  Phase 1 wieder auf) · 409 (Run nicht in `awaiting_input`)
- `POST /acquisitions/{id}/cancel` → 200 (in jeder Phase)
- Route unter `require_api_key` wie alle Nicht-UI-Router.

## 7. MCP surface

- `plan_acquisition(auftrag)` → `AcquisitionSpec` + `open_questions` (Phase 1 trocken,
  kein Job)
- `start_acquisition(spec | auftrag)` → Run-ID
- `get_acquisition(id)` / `list_acquisitions()` / `cancel_acquisition(id)`
- `answer_acquisition(id, answers)` → für den async-Pfad auch via Agent bedienbar

## 8. Configuration surface (Schlüssel/Env-Vars für Betreiber)

- `AcquisitionConfig`; persistiert in neuen `acquisition_runs`/
  `acquisition_sources`-Tabellen (Migration als **neuer** Eintrag in
  `MIGRATIONS`, `core/db.py` — nie einen bestehenden Eintrag editieren).
- **Neue Dependencies** (heute nur `httpx`): eine Web-Search-API
  (z. B. Tavily / Brave / SerpAPI) hinter einer `SearchProvider`-Abstraktion
  analog zu `SynthProvider`; HTML-Haupttext-Extraktion (z. B. `trafilatura`).
  Als optionales Extra `pip install -e ".[acquisition]"`, damit der Kern schlank
  bleibt.
- Env: `TAVILY_API_KEY` o. ä.; `ACQUISITION_MAX_FETCH`, `ACQUISITION_REAL_RATIO`.
- **Netzwerk-Egress:** Der Server braucht produktiv bewusst freigeschalteten
  Internet-Zugang; in der Web-/Sandbox-Umgebung läuft alles über den Agent-Proxy.

## 9. Extension points (für Plugins / externe Nutzung)

- `SearchProvider` (wie `SynthProvider`): Tavily/Brave/SerpAPI/Mock austauschbar;
  `MockSearchProvider` für Tests ohne Netz.
- `Extractor`: HTML→Text pluggbar (trafilatura ↔ readability ↔ custom).
- `AcquisitionDriver` analog `PipelineDriver` (überwacht Phasen, Recovery).
- Curation-Filter als Kette einzeln zu- und abschaltbar.

## 10. Risiken & offene Entscheidungen

- **Recht/Compliance ist der Knackpunkt, nicht die Technik.** Web-Scraping für
  Trainingsdaten berührt ToS, Urheberrecht, DSGVO — bei Buchhaltungs-/DACH-Daten
  besonders. `compliance`/`redaction` sind Pflichtstufe; pro Quelle Lizenz +
  robots.txt erfassen. **Vor Implementierung zu klären:** welche Quellklassen
  überhaupt erlaubt sind (eigene Daten, CC-lizenziert, behördlich offen, …).
- **Qualität > Menge.** „maximal viele Daten" ist das falsche Ziel; Phase 4
  (Curation) entscheidet über den Trainingserfolg. Kennzahlen (akzeptiert/
  verworfen je Filter) müssen sichtbar sein.
- **Kosten/Laufzeit.** Deep-Research + Fetch + LLM-Curation über tausende
  Records ist teuer; ein `cost-tracking`-Hook (existiert) und ein Budget-Limit
  pro Run sind nötig.
- **Determinismus/Reproduzierbarkeit.** Web-Inhalte ändern sich; der `SourcePlan`
  + Snapshots der Roh-Extrakte gehören ins Lineage, damit ein Set nachvollziehbar
  bleibt.

## 11. Tests (müssen existieren und grün sein)

- `tests/test_acquisition_driver.py` — Phasenfortschritt, `awaiting_input`-
  Pause/Resume, Cancel je Phase, Crash-Recovery.
- Phase-Intake mit `MockProvider` → deterministische `AcquisitionSpec` + Fragen.
- Akquise mit `MockSearchProvider` → keine echten Netz-Calls in CI.
- Curation: Dedup/Filter/Redaction reduzieren einen bekannten Rohstrom korrekt.
- Register: sha256-Dedup, `dataset_lineage`-Einträge, Format-Validierung.

## 12. Known gaps / Phasierung des Aufbaus

Empfohlene Reihenfolge (jede Stufe lauffähig und testbar):

1. **MVP-Gerüst:** Modul + `acquisition_runs`-Migration + Route + MCP-Tool +
   Driver mit Mock-Providern; Phasen 1, 3b (synth-reuse), 4 (Dedup+Validate), 5.
   Noch *ohne* echtes Web.
2. **Async-Klärung:** `awaiting_input` + `PATCH …/answers`.
3. **Research/Acquire real:** `SearchProvider` (Tavily) + Extractor + Lizenz-/
   robots-Checks; `[acquisition]`-Extra.
4. **Härtung:** Redaction-Pflichtstufe, Cost-Budget, Lineage-Snapshots, UI-Tab.

## 13. Cross-references

- related_spec: [synthetic-data](synthetic-data.md) — wird als Synthese-Phase wiederverwendet
- related_spec: [multi-stage-pipelines](multi-stage-pipelines.md) — Phasen-Driver-Muster
- related_spec: [dataset-registry](dataset-registry.md) — Output landet hier
- related_spec: [pii-redaction](pii-redaction.md) — Pflicht-Curation-Stufe (Name ggf. anpassen)
- adr: ROADMAP.md — Nordstern „Domänen-LLMs (Buchhalter-DACH)"
