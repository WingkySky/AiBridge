# AIBridge Implementation Plan

> Date: 2026-07-07
> Based on design: [Design](design.md)
> Implementation approach: Claude fully delegated; Phase 0 sequential, Phase 1+ multi-agent parallel

---

## Overall Implementation Strategy

- **Phase 0 (Foundation)**: Claude builds sequentially to ensure consistency in naming/dependencies/configuration (parallelism tends to produce inconsistencies)
- **Phase 1 (MVP four providers)**: Multi-agent parallel (one agent per provider), with Claude first building the OpenAI-compatible foundation
- **Phase 2 (Remaining adapters)**: Multi-agent parallel, in batches 2a/2b/2c
- **Five-language bindings**: Once aibridge-core and aibridge-ffi are stable, multi-agent parallel (one agent per language)
- Each task carries explicit acceptance criteria, serving as the completion basis for agents and the contract for multi-agent orchestration

---

## Phase 0: Foundation (Sequential, 2–3 weeks)

### 0.1 Cargo workspace skeleton
- Workspace root `Cargo.toml` + 4 crates: aibridge-core / aibridge-ffi / aibridge-python / aibridge-node
- Unified dependency versions (workspace.dependencies)
- rust-toolchain.toml + updated .gitignore
- **Acceptance**: `cargo build -p aibridge-core -p aibridge-ffi` passes

### 0.2 aibridge-core infrastructure
- `error.rs`: AibridgeError enum (thiserror), aligned with the Python v1 error taxonomy
- `config.rs`: ClientOptions + environment variable loading (dotenv)
- `http.rs`: reqwest wrapper (h2, timeouts, connection pool)
- `retry.rs`: retry mechanism (exponential backoff, corresponding to tenacity)
- `util.rs`
- **Acceptance**: Core-layer unit tests pass

### 0.3 Data model layer
- `model/{chat,image,video,audio,common,options}.rs`
- serde struct + Builder derive
- **Acceptance**: Model serialization/deserialization tests pass, cross-checked against Python fixtures

### 0.4 Adapter trait + Client + Router
- `adapter::{Adapter trait, Capabilities, create_adapter}`
- `client::Client`, `router::Router`
- **Acceptance**: trait can be implemented, Client/Router compile successfully

### 0.5 aibridge-ffi C ABI skeleton
- Global tokio runtime (Lazy)
- Handle management (client/stream opaque)
- cbindgen configuration + generation of `aibridge.h`
- Basic functions: client_new/destroy/start, chat, chat_stream, stream_next/destroy, last_error, string/bytes_free
- **Acceptance**: `cargo build -p aibridge-ffi` produces cdylib, aibridge.h generated

### 0.6 Cross-language pipeline verification (openai chat stub)
- aibridge-python: PyO3 module + Client.chat async + streaming AsyncIterator
- aibridge-node: napi module + Client.chat async + streaming AsyncIterable
- bindings/go: CGO calling ffi + chat (goroutine + channel)
- bindings/jvm: JNA calling ffi + chat (CompletableFuture)
- bindings/dotnet: P/Invoke calling ffi + chat (Task)
- **Acceptance**: Five-language hello world (with async + streaming + errors) runs successfully ← **biggest technical risk point**

---

## Phase 1: MVP four providers (multi-agent parallel, 3–4 weeks)

### 1.0 OpenAI-compatible foundation (Claude sequential)
- `adapters/openai_compat.rs`: shared HTTP request construction/response parsing/parameter mapping
- `OPENAI_COMPATIBLE_MAPPING` constant
- **Acceptance**: reusable by sub-adapters

### 1.1–1.4 Four provider adapters (one agent per provider, parallel)
| Task | provider | Protocol | Capabilities |
|---|---|---|---|
| 1.1 | openai | reuse compat | chat/image/embed/list_models |
| 1.2 | agnes | reuse compat | chat/image/video/list_models |
| 1.3 | volcengine_cv | standalone | image/video |
| 1.4 | gemini | standalone | chat/image/embed/list_models |
- **Acceptance**: Each adapter's unit tests pass (shared mock fixtures)

### 1.5 Five-language binding completion (multi-agent parallel, one per language)
- Four providers' capabilities exposed to each language
- Cross-language consistency tests
- **Acceptance**: Five languages × four providers all run successfully ← **milestone M1 where the user's other project can integrate**

---

## Phase 2: Remaining adapters (multi-agent parallel batches, 4–6 weeks)

### 2a OpenAI-compatible family (6 agents parallel)
azure, aggregation_platforms, additional_models, more_models, emerging_models, chinese (compatible portion)

### 2b Standalone protocols (5 agents parallel)
anthropic, stability, runway, pika, kling

### 2c Audio (edge-tts/elevenlabs/cartesia/deepgram/assemblyai, binary payloads)

---

## Phase 3: Release wrap-up (2–3 weeks)

- 3.1 CI matrix (platform × language, cross-compilation)
- 3.2 Publishing packages for each language (PyPI/npm/Maven/NuGet/Go module)
- 3.3 Python v1→v2 migration guide
- 3.4 Documentation website
- 3.5 Archive legacy v1 + tag v2.0.0

---

## Multi-agent Orchestration Principles

- **Adapter migration**: one agent per provider, sharing fixtures and the OpenAiCompatAdapter foundation
- **Language bindings**: one agent per language, launched after core/ffi are stable
- **Cross-language consistency tests**: a dedicated agent to consolidate
- **Dependency order**: Phase 0 sequential; Phase 1.0 foundation sequential; 1.1–1.4 adapters parallel; 1.5 bindings parallel
- **Each agent's task contract**: design document reference + acceptance criteria + fixture path + naming conventions

---

## Milestones

- **M0**: Phase 0 complete, five-language hello world runs successfully (technical risk resolved)
- **M1**: Phase 1 complete, four providers usable across five languages (the user's other project can integrate) ⭐
- **M2**: Phase 2 complete, full adapter migration
- **M3**: Phase 3 complete, v2.0.0 officially released
