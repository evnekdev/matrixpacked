# Prompt 04 — Enforce documentation, warning, and package quality gates

Work on https://github.com/evnekdev/matrixpacked

Start only after Prompt 03 is merged.

Create `agent/enforce-quality-gates`. Open a draft PR titled **Enforce warning and documentation quality**. Wait for green CI and finish only with:

**Safe to rebase and merge.**

## Context

The repository already has Linux/OpenBLAS, Windows/MKL, property-test, extended-test, and nalgebra quality CI. Audit and extend existing workflows rather than creating duplicates.

Cargo already has optional `nalgebra-interop` and docs.rs enables it, but crates.io metadata may still be incomplete.

Issue #41 remains deferred. CI should use provider-free checks for isolation and provider-linked tests for execution.

## Goal

Make formatting, compiler warnings, Clippy, missing docs, doctests, examples, tests, feature isolation, Rustdoc, and package contents durable release gates.

## Audit current workflows

Inspect `.github/workflows/` and map:

- Linux/OpenBLAS jobs;
- Windows/MKL jobs;
- fast tests;
- property tests;
- scheduled/manual extended tests;
- nalgebra feature checks;
- Clippy/Rustdoc/examples.

Do not duplicate already reliable commands.

## Required gates

### Formatting

```bash
cargo fmt --all -- --check
```

### Provider-free compile isolation

```bash
cargo check --lib --no-default-features
cargo check --all-targets --features nalgebra-interop
```

Do not require provider-free linking of triangular nalgebra tests until issue #41 is resolved.

### Compiler warnings

Use CI-scoped:

```bash
RUSTFLAGS="-D warnings" cargo check ...
```

Do not add `#![deny(warnings)]` globally.

### Clippy

```bash
cargo clippy --all-targets --features nalgebra-interop -- -D warnings
```

Use provider-linked variants when all targets pull native symbols.

### Tests

Retain:

- ordinary tests;
- nalgebra oracle;
- PR property tests;
- scheduled/manual extended tests;
- Linux primary numerical coverage;
- Windows MKL smoke coverage.

Keep the routine PR matrix affordable.

### Doctests

```bash
cargo test --doc --features nalgebra-interop
```

Add provider only if linked doctest binaries require it.

### Rustdoc

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
```

This must enforce `#![deny(missing_docs)]`, broken links, and Markdown/HTML correctness.

### Examples

Compile every example. Run representative examples on PRs and all examples on schedule/manual CI if runtime is high. Optional nalgebra examples need `nalgebra-interop`; numerical examples need a provider.

### Package verification

In a clean checkout:

```bash
cargo package --list
cargo package
```

Ensure source, README, licenses, user documentation, and useful examples are included.

Exclude generated artifacts, ZIPs, local patches, and prompt recipe folders unless intentionally shipping them. Usually `/prompts_docs`, interoperability prompt folders, and other Codex prompts should be excluded from crates.io packages.

## Cargo metadata

Verify and add as appropriate:

```toml
description = "..."
repository = "https://github.com/evnekdev/matrixpacked"
documentation = "https://docs.rs/matrixpacked"
readme = "README.md"
license = "MIT OR Apache-2.0"
keywords = [...]
categories = [...]
```

Use only licenses present in the repo, at most five valid keywords, and valid crates.io categories. Do not invent a homepage.

Add `rust-version` only after testing the actual MSRV; edition 2024 already sets a lower bound, but do not guess.

## docs.rs

Keep `nalgebra-interop` enabled if provider-free documentation builds. Do not enable static BLAS/MKL providers on docs.rs unless unavoidable.

## Feature policy

Document and test no default provider, optional nalgebra, provider features, and simultaneous-provider behavior. Do not add compile errors without evaluating Cargo feature unification.

## README badges

Add only live CI/license badges now. Add crates.io/docs.rs badges only when those pages exist.

## Validation

Run locally what the environment supports, validate YAML/action versions, and inspect shell syntax on Linux and Windows.

```bash
git diff master...HEAD --stat
git diff master...HEAD
cargo package --list
git diff --check
```

Wait for all CI jobs to be green.

## PR description

Include existing jobs audited, changes made, warning/doc enforcement, test tiers, examples strategy, package metadata/file-list decisions, docs.rs behavior, and explicit deferral of issue #41.
