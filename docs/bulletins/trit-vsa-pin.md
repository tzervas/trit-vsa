# Interface Bulletin: `trit-vsa/pin`

| Field | Value |
|-------|-------|
| **Bulletin ID** | `trit-vsa/pin` |
| **Producer** | [trit-vsa](https://github.com/tzervas/trit-vsa) |
| **Status** | **DRAFT** (candidate for STABLE after crates.io alignment) |
| **Declared pin** | **`0.2.0`** (exact; matches `[package].version` in this repo) |
| **MSRV** | `1.92` (`rust-version` in `Cargo.toml`) |
| **Last verified** | 2026-07-16 |

## Purpose

Consumers (`vsa-optim-rs`, `memory-gate-rs`, `rust-ai-core`, and other crates in the VectorWeight / rust-ai fleet) need a single declared semver pin for `trit-vsa` so path overrides, `[patch]`, and crates.io resolution stay consistent.

## Consumer dependency (copy-paste)

For **crates.io** consumers aligning to this standalone repository:

```toml
[dependencies]
trit-vsa = "0.2.0"
```

Optional features (see `Cargo.toml`):

```toml
trit-vsa = { version = "0.2.0", features = ["simd"] }
# cuda: requires CUDA toolkit, rust-ai-core, candle-core, cubecl — not part of default local gate
```

## Local gate (producer)

From the repository root:

```bash
./scripts/check.sh
```

Equivalent manual commands:

```bash
cargo check
cargo test
```

**Note:** `cargo check --all-features` / `cuda` is **environment-gated** (CUDA toolkit). The producer gate does not require GPU.

## Consumer pin snapshot (workspace recon, not modified by this bulletin)

| Consumer | Location | Declared `trit-vsa` pin | Notes |
|----------|----------|-------------------------|-------|
| `vsa-optim-rs` | `rust-ai/vsa-optim-rs` | `0.3` (path `../trit-vsa`) | Monorepo copy at **0.3.0** |
| `vsa-optim-rs` | standalone `/vsa-optim-rs` | `0.1` | Awaiting bump |
| `rust-ai-core` | `rust-ai-core` | `0.3` | Orchestration manifest |
| `memory-gate-rs` | `memory-gate-rs` | `0.3.0` (optional) | `vsa-accel` feature |
| `bitnet-quantize` | `bitnet-quantize` | `0.2.0` (path) | Aligned with **0.2.0** |
| `aphelion-framework-rs` | workspace | `0.3.0` | |

**Alignment rule:** This bulletin declares **`0.2.0`** for the **standalone** `tzervas/trit-vsa` repository. Ecosystem crates that pin **`0.3`** depend on the **rust-ai monorepo** `trit-vsa` subtree until a **0.3.x** release is published from this repo (or consumers retarget to `0.2.0`). Producers must not edit consumer `Cargo.toml` files; consumers bump in their own PRs after this bulletin reaches **STABLE**.

## API surface (stability scope for pin)

Pin covers the public crate API documented on [docs.rs/trit-vsa](https://docs.rs/trit-vsa):

- Core types: `Trit`, `Tryte3`, `Word6`, `PackedTritVec`, `SparseVec`
- Module `vsa`: `bind`, `unbind`, `bundle`, similarity helpers
- Module `kernels` / `dispatch`: CPU backends and optional GPU paths behind features

Breaking changes require a semver bump and bulletin revision.

## Evidence

- Tests: `122` unit tests + doc tests green on declared MSRV toolchain (see producer evidence log).
- Repository: `https://github.com/tzervas/trit-vsa`

## Promotion to STABLE

- [ ] crates.io published version matches declared pin **or** bulletin updated to match published version
- [ ] `./scripts/check.sh` green on `main`
- [ ] Consumer owners acknowledge pin in dependent repos (separate PRs)